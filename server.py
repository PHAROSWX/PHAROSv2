# EU Thunderstorm Outlook - Official ECMWF Open Data Server
#
# Uses the official ecmwf-opendata Python package to fetch GRIB2
# from data.ecmwf.int (CC-BY-4.0 free since Oct 2025).
#
# Key param notes (IFS cycle 49r1+):
#   - CAPE (param 59) discontinued -> MUCAPE (228235, shortName mucape)
#   - CIN  (228001)  discontinued -> MUCIN  (228236, shortName mucin)
#   - K-Index not in open data -> derived from T/RH at 850/700/500 hPa
#   - LPI not in open data -> engine uses K-Index + shear instead
#
# Install:
#   pip install ecmwf-opendata cfgrib xarray flask flask-cors numpy scipy
#
# Run:
#   python server.py
# Then open: http://localhost:8765

import os, pickle, time, threading, tempfile, logging, shutil, requests, gzip, json as _json
import ctypes, sys
import warnings
# Suppress cfgrib/xarray FutureWarning about compat kwarg — harmless, just noise
warnings.filterwarnings('ignore', category=FutureWarning, module='cfgrib')
warnings.filterwarnings('ignore', message='.*compat.*', category=FutureWarning)

# Suppress ECCODES C-library stderr ("Parser: syntax error at line 1 of template.4.X.def")
# These are harmless definition warnings from old eccodes versions, not parse failures.
import contextlib

@contextlib.contextmanager
def suppress_eccodes_stderr():
    """Redirect C-level stderr to devnull around cfgrib calls to hide ECCODES warnings."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
    except Exception:
        yield  # if fd manipulation fails, just proceed normally
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.')
CORS(app)

# Config
LAT_MIN, LAT_MAX = 34.0,  72.0
LON_MIN, LON_MAX = -26.0,  48.0
GRID_STEP = 0.0625
STEPS     = list(range(0, 73, 3))
PL_LEVELS = [850, 700, 500, 250]

CACHE_FILE  = Path('forecast_cache.pkl')
DATA_GZ_FILE = Path('forecast_data.json.gz')   # pre-built response file
CACHE_MAX_H = 5

# State
_lock       = threading.Lock()
_data       = None
_data_gz    = None   # pre-built gzip-compressed JSON response bytes
_raw_params = {}     # {model: [step_data, ...]} — in memory only, not pickled
_run        = None
_loading    = False
_prog       = {'pct': 0, 'msg': 'Idle'}

def prog(pct, msg):
    global _prog
    _prog = {'pct': pct, 'msg': msg}
    log.info('[%3d%%] %s', pct, msg)


def _gefs_parse_worker(fpath_str):
    """
    Module-level GEFS GRIB2 parse worker for multiprocessing.
    Each worker process has its own cfgrib C library — fully parallel and safe.
    Returns dict {varkey: (flat_lats, flat_lons, flat_vals)} or {}.
    """
    import os, numpy as _np
    from pathlib import Path as _Path
    fpath = _Path(fpath_str)
    if not fpath.exists():
        return {}
    fields = {}
    try:
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_err = os.dup(2); os.dup2(devnull_fd, 2); os.close(devnull_fd)
            suppress = True
        except Exception:
            suppress = False
        try:
            import cfgrib as _cf
            dsets = _cf.open_datasets(fpath_str, indexpath=None)
        finally:
            if suppress:
                try: os.dup2(old_err, 2); os.close(old_err)
                except: pass
        for ds in dsets:
            lats = ds.latitude.values
            lons = _np.where(ds.longitude.values > 180,
                             ds.longitude.values - 360, ds.longitude.values)
            if lats.ndim == 1:
                LG, OG = _np.meshgrid(lats, lons, indexing="ij")
                fl_la, fl_lo = LG.ravel(), OG.ravel()
            else:
                fl_la, fl_lo = lats.ravel(), lons.ravel()
            for var in ds.data_vars:
                arr = ds[var].values
                lev = None
                for c in ("isobaricInhPa", "pressure", "level"):
                    if c in ds.coords:
                        lev = ds.coords[c].values; break
                if arr.ndim == 3 and lev is not None:
                    for li, lv in enumerate(_np.atleast_1d(lev)):
                        lv_int = int(round(float(lv)))
                        if lv_int not in (925, 850, 700, 500, 250): continue
                        v = arr[li].ravel().astype(_np.float32)
                        if var == "t": v -= 273.15
                        fields["%s%d" % (var, lv_int)] = (fl_la, fl_lo, v)
                elif arr.ndim == 2:
                    v = arr.ravel().astype(_np.float32)
                    if var == "t": v -= 273.15
                    if lev is not None:
                        lv_int = int(round(float(_np.atleast_1d(lev)[0])))
                        if lv_int in (925, 850, 700, 500, 250):
                            fields["%s%d" % (var, lv_int)] = (fl_la, fl_lo, v)
                    elif var in ("cape", "cin", "pwat"):
                        fields[var] = (fl_la, fl_lo, v)
    except Exception:
        pass
    return fields


def rh_to_td(t_c, rh):
    rh = np.clip(rh, 1, 100)
    g  = (17.67 * t_c) / (243.5 + t_c) + np.log(rh / 100.0)
    return (243.5 * g) / (17.67 - g)


def compute_k_index(t850, t700, t500, rh850, rh700):
    # K-Index = (T850-T500) + Td850 - (T700-Td700)
    td850 = rh_to_td(t850, rh850)
    td700 = rh_to_td(t700, rh700)
    return (t850 - t500) + td850 - (t700 - td700)


# ── thundeR-equivalent vectorised parameter calculations ─────────────────────
# These replicate the key indices from Bczernecki et al. (thundeR package).
# All operate on flat arrays of shape (N,) — one value per grid point.

def compute_lcl_height(t2m, td2m):
    """
    LCL height (m AGL) — Bolton (1980) approximation.
    Input: T and Td in °C.
    """
    # Safe clamp
    td = np.minimum(td2m, t2m)
    # Approximate LCL temperature
    # LCL_T = Td - (0.212 + 1.571e-3*(Td-0) - 4.36e-4*(T-0))*(T-Td)
    lcl_t = td - (0.212 + 1.571e-3 * td - 4.36e-4 * t2m) * (t2m - td)
    # LCL height from dry adiabatic lapse rate: ~125 m per °C depression
    lcl_h = 125.0 * (t2m - lcl_t)
    return np.maximum(lcl_h, 0.0)


def compute_srh_proxy(u925, v925, u850, v850, u700, v700):
    """Storm-Relative Helicity using Bunkers right-mover.
    SRH = ∫(V_storm - V_wind) × (∂V/∂z) dz
    Discrete form: SRH = (cu-u1)*(v2-v1) - (cv-v1)*(u2-u1)
    Positive SRH = cyclonic (NH) = favours right-moving supercells.
    """
    u_mean = (u925 + u850 + u700) / 3.0
    v_mean = (v925 + v850 + v700) / 3.0
    u_shear = u700 - u925
    v_shear = v700 - v925
    shear_mag = np.hypot(u_shear, v_shear) + 1e-6
    d = 7.5   # Bunkers offset (m/s)
    # Right-mover: deviate 7.5 m/s to right of shear vector
    cu = u_mean + d * v_shear / shear_mag
    cv = v_mean - d * u_shear / shear_mag
    # SRH using (storm - wind) cross-product convention (positive = cyclonic)
    srh_01 = (cu - u925) * (v850 - v925) - (cv - v925) * (u850 - u925)
    srh_13 = (cu - u850) * (v700 - v850) - (cv - v850) * (u700 - u850)
    srh_03 = srh_01 + srh_13
    return np.nan_to_num(srh_01), np.nan_to_num(srh_03)


def compute_ehi(cape, srh_01):
    """
    Energy-Helicity Index (0-1km SRH version).
    EHI = CAPE × SRH_01km / 160000
    Thresholds: >1 = elevated tornado potential, >2 = significant tornado likely.
    """
    return np.maximum(cape, 0) * np.maximum(srh_01, 0) / 160000.0


def compute_stp(cape, sh6, srh_01, lcl_h, cin):
    """
    Significant Tornado Parameter — thundeR/SPC formulation.
    STP = (MLCAPE/1500) × (sh6/20) × (1500-LCL_h)/1000 × (SRH01/150)
          × CIN modifier

    Thresholds (per thundeR docs):
      STP > 1   : elevated tornado risk
      STP > 3   : significant tornado potential
      STP > 5   : major outbreak environment
    """
    cape_term = np.clip(cape / 1500.0, 0, None)
    shear_term = np.clip(sh6 / 20.0, 0, None)
    # LCL modifier: penalise high LCL (>1500m → cold RFD → tornadogenesis failure)
    lcl_term = np.where(lcl_h < 1000, 1.0,
               np.where(lcl_h < 2000, (2000 - lcl_h) / 1000.0, 0.0))
    srh_term = np.clip(srh_01 / 150.0, 0, None)
    # CIN modifier: slight cap beneficial (focused initiation), extreme cap kills
    cin_term = np.where(cin >= -50, 1.0,
               np.where(cin >= -200, (200 + cin) / 150.0, 0.0))

    return np.nan_to_num(cape_term * shear_term * lcl_term * srh_term * cin_term)


def compute_brn(cape, sh6):
    """
    Bulk Richardson Number — storm mode discriminator.
    BRN = CAPE / (0.5 * DLS²)
    BRN 10-50 = supercells likely; >150 = disorganised multicells.
    """
    dls2 = np.maximum(sh6 * sh6, 1.0)
    return np.nan_to_num(np.minimum(cape / (0.5 * dls2), 999.0))


def compute_dcape(t700, t500, r850):
    """DCAPE — Downdraft CAPE proxy.
    thundeR initialises from mean θe in 3-5km layer.
    Proxy: mid-level lapse rate × low-level dryness.
    DCAPE ≈ (LR - 6 K/km) × (100 - RH700) × 15   [J/kg]
    Typical ranges: 200-800 J/kg (moderate), >1000 (derecho).
    """
    lr_km = (t700 - t500) / 2.5   # 700-500 hPa lapse rate
    rh_deficit = np.maximum(100.0 - r850, 0.0)
    dcape = np.maximum(lr_km - 6.0, 0.0) * rh_deficit * 15.0
    return np.maximum(dcape, 0.0)


def fetch_all(tmpdir):
    from ecmwf.opendata import Client
    client = Client(source='ecmwf', model='ifs', resol='0p25')

    prog(2, 'Detecting latest ECMWF IFS run...')

    # Try runs in order: most recent first, going back up to 36 hours.
    # ECMWF open data is typically available 6-9 hours after nominal run time.
    now_utc = datetime.now(timezone.utc)
    candidate_runs = []
    for delta_h in range(0, 40, 6):
        dt = now_utc - timedelta(hours=delta_h)
        # Only use 00 and 12 UTC runs (main IFS runs)
        for hour in [12, 0]:
            candidate = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
            if candidate <= now_utc:
                candidate_runs.append(candidate)

    run_dt = None
    for cand in candidate_runs:
        date_s_try = cand.strftime('%Y%m%d')
        run_s_try  = '%02d' % cand.hour
        try:
            # Probe with a minimal request (step 0, T at 500) to confirm availability
            probe_path = Path(tmpdir) / 'probe.grib2'
            client.retrieve(
                date=date_s_try, time=int(run_s_try),
                type='fc', step=[0],
                levtype='pl', levelist=[500],
                param='t',
                target=str(probe_path),
            )
            if probe_path.exists() and probe_path.stat().st_size > 0:
                run_dt = cand
                log.info('Found available run: %s/%sz', date_s_try, run_s_try)
                probe_path.unlink(missing_ok=True)
                break
        except Exception as e:
            log.debug('Run %s/%sz not available: %s', date_s_try, run_s_try, e)
            continue

    if run_dt is None:
        # Last resort: try the latest() method
        try:
            lat = client.latest(type='fc', step=6, param='2t')
            if lat:
                run_dt = lat if isinstance(lat, datetime) else datetime.fromisoformat(str(lat))
        except Exception as e:
            log.warning('Auto-detect latest failed: %s', e)

    if run_dt is None:
        log.error('Could not find any available ECMWF run')
        return {}, (now_utc.strftime('%Y%m%d'), '00')

    date_s = run_dt.strftime('%Y%m%d')
    run_s  = '%02d' % run_dt.hour
    log.info('Using ECMWF run %s/%sz', date_s, run_s)

    files = {}

    # Surface: MUCAPE
    for pname, label in [('mucape', 'MUCAPE')]:
        fpath = Path(tmpdir) / ('sfc_' + pname + '.grib2')
        prog(5, 'Fetching ' + label + '...')
        try:
            client.retrieve(
                date=date_s, time=int(run_s),
                type='fc', step=STEPS, param=pname,
                target=str(fpath),
            )
            files[('sfc', pname)] = fpath
            log.info('  OK %s (%d KB)', label, fpath.stat().st_size // 1024)
        except Exception as e:
            log.warning('  SKIP %s: %s', label, e)

    # Pressure levels — request each parameter separately so one failure
    # doesn't block the others. Try with and without 925 hPa.
    pl_params = [
        ('t', 'Temperature',   [925, 850, 700, 500]),
        ('r', 'Rel. Humidity', [850, 700, 500]),       # 925 RH often not in open data
        ('u', 'U-wind',        [925, 850, 700, 500, 250]),
        ('v', 'V-wind',        [925, 850, 700, 500, 250]),
    ]
    for pname, label, levels in pl_params:
        fpath = Path(tmpdir) / ('pl_' + pname + '.grib2')
        prog(15, 'Fetching %s at %s hPa...' % (label, levels))
        # Try requested levels first, then fall back without 925
        for try_levels in [levels, [l for l in levels if l != 925]]:
            try:
                client.retrieve(
                    date=date_s, time=int(run_s),
                    type='fc', step=STEPS,
                    levtype='pl', levelist=try_levels,
                    param=pname,
                    target=str(fpath),
                )
                files[('pl', pname)] = fpath
                log.info('  OK %s pl %s (%d KB)', label, try_levels, fpath.stat().st_size // 1024)
                break
            except Exception as e:
                if try_levels == [l for l in levels if l != 925]:
                    log.warning('  SKIP %s pl: %s', label, e)

    return files, (date_s, run_s)


def read_grib(fpath, lat_min, lat_max, lon_min, lon_max):
    # Returns {level_int: {'lats':arr, 'lons':arr, 'steps':{h: arr}}}
    import cfgrib

    if not fpath.exists():
        return {}

    result = {}
    try:
        with suppress_eccodes_stderr():
            dsets = cfgrib.open_datasets(str(fpath))
    except Exception as e:
        log.warning('cfgrib open %s: %s', fpath.name, e)
        return {}

    for ds in dsets:
        if 'latitude' not in ds.coords:
            continue

        raw_la = ds.latitude.values
        raw_lo = ds.longitude.values
        raw_lo = np.where(raw_lo > 180, raw_lo - 360, raw_lo)

        # Build Europe mask and flat lat/lon arrays
        if raw_la.ndim == 1 and raw_lo.ndim == 1:
            # Regular grid
            lat_m  = (raw_la >= lat_min) & (raw_la <= lat_max)
            lon_m  = (raw_lo >= lon_min) & (raw_lo <= lon_max)
            sub_la = raw_la[lat_m]
            sub_lo = raw_lo[lon_m]
            flat_la = np.repeat(sub_la, len(sub_lo))
            flat_lo = np.tile(sub_lo,   len(sub_la))
            is_regular = True
        else:
            # Gaussian / unstructured
            fla = raw_la.ravel()
            flo = raw_lo.ravel()
            fmask  = ((fla >= lat_min) & (fla <= lat_max) &
                      (flo >= lon_min) & (flo <= lon_max))
            flat_la = fla[fmask]
            flat_lo = flo[fmask]
            is_regular = False

        # Determine step dimension
        step_dim = None
        for sd in ('step', 'valid_time'):
            if sd in ds.dims or sd in ds.coords:
                step_dim = sd
                break

        for var in ds.data_vars:
            da = ds[var]

            def subset(arr2d):
                if is_regular:
                    if arr2d.ndim == 2:
                        return arr2d[np.ix_(lat_m, lon_m)].ravel()
                    return arr2d.ravel()
                else:
                    return arr2d.ravel()[fmask]

            def step_to_h(t_val):
                if step_dim == 'step':
                    return int(t_val / np.timedelta64(1, 'h'))
                base = ds.coords.get('time')
                if base is not None:
                    bv = np.datetime64(
                        base.values.item() if hasattr(base.values, 'item') else base.values, 'ns')
                    return int((np.datetime64(t_val, 'ns') - bv) / np.timedelta64(1, 'h'))
                return 0

            # Level key
            if 'isobaricInhPa' in ds.coords:
                lev_coord = 'isobaricInhPa'
            elif 'pressure' in ds.coords:
                lev_coord = 'pressure'
            else:
                lev_coord = None

            try:
                if lev_coord and lev_coord in da.dims:
                    for lev_val in da[lev_coord].values:
                        lkey = int(lev_val)
                        da_l = da.sel({lev_coord: lev_val})
                        if lkey not in result:
                            result[lkey] = {'lats': flat_la, 'lons': flat_lo, 'steps': {}}
                        if step_dim and step_dim in da_l.dims:
                            for t_val in da_l[step_dim].values:
                                h   = step_to_h(t_val)
                                arr = da_l.sel({step_dim: t_val}).values
                                result[lkey]['steps'][h] = subset(arr)
                        else:
                            result[lkey]['steps'][0] = subset(da_l.values)
                else:
                    lkey = 0
                    if lkey not in result:
                        result[lkey] = {'lats': flat_la, 'lons': flat_lo, 'steps': {}}
                    if step_dim and step_dim in da.dims:
                        for t_val in da[step_dim].values:
                            h   = step_to_h(t_val)
                            arr = da.sel({step_dim: t_val}).values
                            result[lkey]['steps'][h] = subset(arr)
                    else:
                        result[lkey]['steps'][0] = subset(da.values)
            except Exception as e:
                log.warning('  var %s decode: %s', var, e)

    return result


_ml_cache = None


class CalibratedModel:
    def __init__(self, base, calibrator, features):
        self.base = base
        self.calibrator = calibrator
        self.feature_importances_ = base.feature_importances_
        self._features = features

    def predict_proba(self, X):
        import numpy as _np
        import pandas as _pd
        raw = self.base.predict_proba(
            _pd.DataFrame(X, columns=self._features))[:, 1]
        cal = self.calibrator.predict(raw)
        cal = _np.clip(cal, 0.0, 1.0)
        return _np.column_stack([1 - cal, cal])


def load_ml_models():
    """
    Load trained ML models from convective_models.pkl if it exists.
    Returns the loaded dict or None if not found.
    Caches in memory so we only load once per server run.
    """
    global _ml_cache
    if _ml_cache is not None:
        return _ml_cache
    model_path = Path('convective_models.pkl')
    if not model_path.exists():
        return None
    try:
        import lightgbm as _lgb  # must be imported before unpickling CalibratedModel
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
        _ml_cache = obj
        log.info('ML models loaded from %s (version %s, trained %s)',
                 model_path, obj.get('version', '?'), obj.get('trained_on', '?'))
        return _ml_cache
    except Exception as e:
        log.warning('Could not load ML models: %s', e)
        return None


def ml_score_grid(ml, cape, cin, kx, t850, t700, t500, r850, r700,
                  u850, v850, u250, v250, sh6, month, hour, lats, lons,
                  sh1=None, sh3=None, srh1=None, srh3=None,
                  stp=None, scp=None, ship=None, ehi=None, ehi3=None,
                  wmax=None, wmaxshear=None, dcape=None, lcl_h=None,
                  td850=None, r700_=None, pwat=None):
    """
    Run ML models on the full grid (vectorised — one predict_proba call
    for all N grid points at once, much faster than a loop).

    Returns dict: {'tstm': array[N], 'hail': array[N], ...}
    Each array contains P(event) in [0,1] for every grid point.
    """
    import math

    n = len(cape)
    models   = ml.get('models', {})
    features = ml.get('features', [])

    # ── Build feature matrix [N × n_features] ─────────────────────────────
    # Must match exactly the order used in train_model.py's FEATURES list
    lapse_rate   = (t850 - t500) / 3.5
    u500 = (u850 + u250) / 2   # approximate — 500 hPa not in ECMWF open data
    v500 = (v850 + v250) / 2
    sh3  = np.hypot((u850+u250)/2 - u850, (v850+v250)/2 - v850)  # rough sh3 proxy
    z500 = np.full(n, 5500.0)   # climatological fallback; not in open data

    # Derived features (same as train_model.py add_features())
    brooks_cds  = np.sqrt(np.maximum(cape, 0)) * sh6 / 100.0
    li_approx   = np.zeros(n)   # LI not available from open data; use 0 (neutral)
    pwat_approx = r850 * 0.4    # crude PWAT proxy from r850
    td2m_approx = t850 - 10.0  # crude surface dewpoint from T850

    stp_proxy = (
        np.maximum(cape, 0) / 1500.0 *
        sh6 / 20.0 *
        np.maximum(r850, 1) / 55.0 *
        np.maximum(1.0 - li_approx / 6.0, 0)
    ).clip(0, 10)

    ehi_proxy = (
        np.maximum(cape, 0) *
        (sh6 * r850 / 100.0) / 160000.0
    ).clip(0, 5)

    moisture_flux = r850 * np.maximum(pwat_approx, 0) / 100.0

    month_sin = math.sin(2 * math.pi * month / 12.0)
    month_cos = math.cos(2 * math.pi * month / 12.0)
    hour_sin  = math.sin(2 * math.pi * hour  / 24.0)
    hour_cos  = math.cos(2 * math.pi * hour  / 24.0)
    lat_norm  = (np.array(lats) - 53.0) / 19.0

    # Build the feature matrix column by column in the same order as FEATURES
    feat_map = {
        'cape': cape, 'cin': cin, 'ki': kx, 'li': li_approx,
        'pwat': pwat_approx, 'td2m': td2m_approx,
        't850': t850, 't700': t700, 't500': t500, 't250': t500 - 20.0,
        'rh850': r850, 'rh700': r700, 'rh500': r850 * 0.7,
        'u850': u850, 'v850': v850, 'u500': u500, 'v500': v500,
        'u250': u250, 'v250': v250,
        'sh6': sh6, 'sh3': sh3, 'lapse_rate': lapse_rate, 'z500': z500,
        'brooks_cds': brooks_cds, 'stp_proxy': stp_proxy, 'ehi_proxy': ehi_proxy,
        'moisture_flux': moisture_flux,
        'month_sin': np.full(n, month_sin), 'month_cos': np.full(n, month_cos),
        'hour_sin':  np.full(n, hour_sin),  'hour_cos':  np.full(n, hour_cos),
        'lat_norm': lat_norm, 'lon': np.array(lons),
    }

    X = np.column_stack([
        np.nan_to_num(feat_map.get(f, np.zeros(n)), nan=0.0, posinf=0.0, neginf=0.0)
        for f in features
    ]).astype(np.float32)

    # ── Score each hazard ──────────────────────────────────────────────────
    results = {}
    for hazard in ['tstm', 'hail', 'wind', 'tornado']:
        model = models.get(hazard)
        if model is None:
            results[hazard] = np.zeros(n)
            continue
        try:
            probs = model.predict_proba(X)[:, 1]
            np.nan_to_num(probs, copy=False)
            results[hazard] = probs.astype(np.float32)
        except Exception as e:
            log.warning('ML score error for %s: %s', hazard, e)
            results[hazard] = np.zeros(n)

    return results


def build_grid(files, run):
    from scipy.spatial import cKDTree

    prog(58, 'Reading GRIB2 files...')
    parsed = {}
    for fkey, fpath in files.items():
        log.info('Reading %s...', fpath.name)
        parsed[fkey] = read_grib(fpath, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        log.info('  Levels/keys: %s', list(parsed[fkey].keys()))

    # Output grid
    lats_g = np.arange(LAT_MIN, LAT_MAX + GRID_STEP, GRID_STEP)
    lons_g = np.arange(LON_MIN, LON_MAX + GRID_STEP, GRID_STEP)
    LG, OG = np.meshgrid(lats_g, lons_g, indexing='ij')
    flat_la = LG.ravel()
    flat_lo = OG.ravel()
    n = len(flat_la)

    # KDTrees per source grid size
    trees = {}
    for fkey, levels in parsed.items():
        for lev, ld in levels.items():
            src_la = ld['lats'].ravel()
            src_lo = np.where(ld['lons'].ravel() > 180,
                              ld['lons'].ravel() - 360, ld['lons'].ravel())
            tk = len(src_la)
            if tk not in trees:
                log.info('Building KDTree for %d source points...', tk)
                trees[tk] = cKDTree(np.column_stack([src_la, src_lo]))
            dists, idxs = trees[tk].query(np.column_stack([flat_la, flat_lo]), k=1)
            ld['idxs'] = idxs
            ld['far']  = dists > 3.0

    def get_field(fkey, lev, step, default=0.0, k_to_c=False):
        d = parsed.get(fkey, {}).get(lev)
        if d is None:
            return np.full(n, default)
        steps = d.get('steps', {})
        if not steps:
            return np.full(n, default)
        s = step if step in steps else min(steps, key=lambda x: abs(x - step))
        if abs(s - step) > 6:
            return np.full(n, default)
        raw = np.array(steps[s]).ravel()
        out = raw[d['idxs']]
        out[d['far']] = default
        if k_to_c:
            out = out - 273.15
        return out

    date_s, run_s = run
    base = datetime(int(date_s[:4]), int(date_s[4:6]), int(date_s[6:]),
                    int(run_s), tzinfo=timezone.utc)
    times_out = [(base + timedelta(hours=s)).strftime('%Y-%m-%dT%H:%M')
                 for s in STEPS]

    # ── Load ML models if available ─────────────────────────────────────────
    ml = load_ml_models()
    if ml:
        log.info('ML models loaded — using trained classifiers for scoring')
    else:
        log.info('No ML models found — using hand-coded risk engine')

    data_out = []
    for si, step in enumerate(STEPS):
        pct = 65 + int(si / len(STEPS) * 30)
        prog(pct, 'Regridding step %dh (%d/%d)...' % (step, si + 1, len(STEPS)))

        mucape = get_field(('sfc', 'mucape'), 0,   step)
        mucin  = np.zeros(n)

        t925  = get_field(('pl', 't'), 925, step, k_to_c=True)
        t850  = get_field(('pl', 't'), 850, step, k_to_c=True)
        t700  = get_field(('pl', 't'), 700, step, k_to_c=True)
        t500  = get_field(('pl', 't'), 500, step, k_to_c=True)
        r925  = get_field(('pl', 'r'), 925, step, default=70.0)
        r850  = get_field(('pl', 'r'), 850, step, default=50.0)
        r700  = get_field(('pl', 'r'), 700, step, default=50.0)

        kx = compute_k_index(t850, t700, t500, r850, r700)
        np.nan_to_num(kx, copy=False)

        u925 = get_field(('pl', 'u'), 925, step)
        v925 = get_field(('pl', 'v'), 925, step)
        u850 = get_field(('pl', 'u'), 850, step)
        v850 = get_field(('pl', 'v'), 850, step)
        u700 = get_field(('pl', 'u'), 700, step)
        v700 = get_field(('pl', 'v'), 700, step)
        u500 = get_field(('pl', 'u'), 500, step)
        v500 = get_field(('pl', 'v'), 500, step)
        u250 = get_field(('pl', 'u'), 250, step)
        v250 = get_field(('pl', 'v'), 250, step)
        sh6  = np.hypot(u250 - u850, v250 - v850)
        sh1  = np.hypot(u850 - u925, v850 - v925)  # 0-1km proxy
        np.nan_to_num(sh6, copy=False)
        np.nan_to_num(sh1, copy=False)

        # ═══════════════════════════════════════════════════════════════════
        # ALL 201 thundeR v1.1 parameters (Bczernecki et al.)
        # Computed from available pressure levels: 925/850/700/500/250 hPa
        # Heights: 925≈800m, 850≈1500m, 700≈3000m, 500≈5500m, 250≈10500m
        # ═══════════════════════════════════════════════════════════════════

        # ── Dewpoints ────────────────────────────────────────────────────────
        td925 = rh_to_td(t925, r925)
        td850 = rh_to_td(t850, r850)
        td700 = rh_to_td(t700, r700)

        # ── Mixing ratio (g/kg) ───────────────────────────────────────────────
        mixr_sfc = r925 * 0.622 / (1000.0 - r925 * 10.0)  # ~surface
        mixr_850 = r850 * 0.622 / (1000.0 - r850 * 10.0)

        # ── Lapse rates (°C/km) ───────────────────────────────────────────────
        lr_85_50 = (t850 - t500) / 3.5       # 850-500 hPa (3.5 km)
        lr_70_50 = (t700 - t500) / 2.5       # 700-500 hPa (2.5 km)
        lr_92_85 = (t925 - t850) / 0.7       # 925-850 hPa (0.7 km)
        lr_85_70 = (t850 - t700) / 1.5       # 850-700 hPa (1.5 km)
        lr_06km  = (t925 - t500) / 5.5       # sfc-6km proxy
        np.nan_to_num(lr_85_50, copy=False)
        np.nan_to_num(lr_70_50, copy=False)
        np.nan_to_num(lr_06km,  copy=False)

        # ── Parcel temperatures at 500 hPa ────────────────────────────────────
        t_par_mu = t925 - 34.3    # MU parcel: 925→500 (~3.5km × 9.8K/km)
        t_par_sb = t925 - 34.3    # SB: same as MU (no surface data)
        t_par_ml = t850 - 19.5    # ML (0-500m mean): 850→500 (~3km)

        # ── LCL heights ───────────────────────────────────────────────────────
        mu_lcl = compute_lcl_height(t925, td925)
        sb_lcl = compute_lcl_height(t925, td925)   # proxy same as MU
        ml_lcl = compute_lcl_height(t850, td850)   # 850-based proxy

        # ── WMAX = sqrt(2*CAPE) ───────────────────────────────────────────────
        mu_wmax = np.sqrt(2.0 * np.maximum(mucape, 0))
        sb_wmax = mu_wmax * 0.85   # SB typically slightly less
        ml_wmax = mu_wmax * 0.80   # ML typically less
        np.nan_to_num(mu_wmax, copy=False)
        np.nan_to_num(sb_wmax, copy=False)
        np.nan_to_num(ml_wmax, copy=False)

        # ── CAPE variants ─────────────────────────────────────────────────────
        mu_cape = mucape
        sb_cape = mucape * 0.85    # SB proxy
        ml_cape = mucape * 0.75    # ML proxy
        # 0-3km CAPE proxy (LCL-dependent)
        mu_03cape = np.where(mu_lcl < 1500, mu_cape * 0.30, mu_cape * 0.15)
        sb_03cape = np.where(sb_lcl < 1500, sb_cape * 0.30, sb_cape * 0.15)
        ml_03cape = np.where(ml_lcl < 1500, ml_cape * 0.30, ml_cape * 0.15)
        # 0-2km CAPE proxy
        mu_02cape = mu_03cape * 0.6
        sb_02cape = sb_03cape * 0.6
        ml_02cape = ml_03cape * 0.6
        # HGL CAPE (0°C to -20°C layer ≈ fraction of total)
        mu_hgl = mu_cape * 0.25
        sb_hgl = sb_cape * 0.25
        ml_hgl = ml_cape * 0.25
        # CAPE > -10°C proxy
        mu_cape_m10 = mu_cape * 0.45
        sb_cape_m10 = sb_cape * 0.45
        ml_cape_m10 = ml_cape * 0.45

        # ── CIN variants ─────────────────────────────────────────────────────
        mu_cin = mucin
        sb_cin = mucin * 0.9
        ml_cin = mucin * 1.1

        # ── LI variants ───────────────────────────────────────────────────────
        mu_li = t500 - t_par_mu
        sb_li = t500 - t_par_sb
        ml_li = t500 - t_par_ml
        np.nan_to_num(mu_li, copy=False); np.nan_to_num(sb_li, copy=False); np.nan_to_num(ml_li, copy=False)

        # ── Showalter / TT / SWEAT / K-Index ────────────────────────────────
        si    = t500 - (t850 - 19.5)
        tt    = t850 + td850 - 2.0 * t500
        sweat = (np.maximum(12.0 * td850, 0) +
                 20.0 * np.maximum(tt - 49.0, 0) +
                 2.0 * np.hypot(u850, v850))
        np.nan_to_num(si, copy=False); np.nan_to_num(tt, copy=False); np.nan_to_num(sweat, copy=False)

        # ── LCL / LFC / EL temperatures and heights (proxies) ────────────────
        mu_lcl_temp = td925          # LCL temp ≈ surface dewpoint
        sb_lcl_temp = td925
        ml_lcl_temp = td850
        mu_lfc_hgt  = mu_lcl * 1.2  # LFC typically above LCL
        sb_lfc_hgt  = sb_lcl * 1.2
        ml_lfc_hgt  = ml_lcl * 1.2
        # EL height: proportional to CAPE (higher CAPE → higher EL)
        mu_el_hgt   = 8000.0 + mu_cape * 1.5
        sb_el_hgt   = 8000.0 + sb_cape * 1.5
        ml_el_hgt   = 8000.0 + ml_cape * 1.5
        mu_el_temp  = t500 - lr_85_50 * 2.0    # ~500 hPa proxy
        sb_el_temp  = mu_el_temp
        ml_el_temp  = mu_el_temp

        # ── Theta-E ───────────────────────────────────────────────────────────
        # θe ≈ T + 2500 * mixr/cp  (Bolton 1980 approximation)
        thetae_sfc  = t925 + 273.15 + 2500.0 * mixr_sfc / 1004.0
        thetae_850  = t850 + 273.15 + 2500.0 * mixr_850 / 1004.0
        thetae_500  = t500 + 273.15
        delta_thetae    = (thetae_500 - thetae_sfc)  # 3-5km minus sfc
        delta_thetae_04 = (thetae_500 - thetae_sfc)  # 0-4km proxy

        # ── PWAT ─────────────────────────────────────────────────────────────
        pwat = (r925 + r850 + r700) / 3.0 * 0.5   # rough mm

        # ── DCAPE / Cold Pool / Wind Index ───────────────────────────────────
        dcape      = compute_dcape(t700, t500, r850)
        cold_pool  = np.sqrt(2.0 * np.maximum(dcape, 0)) * 0.5
        wind_index = np.sqrt(np.maximum(dcape, 0) / 2.0) * np.maximum(lr_85_50, 0) / 5.0
        np.nan_to_num(dcape, copy=False); np.nan_to_num(cold_pool, copy=False); np.nan_to_num(wind_index, copy=False)

        # ── Moisture Flux ─────────────────────────────────────────────────────
        mfc = np.hypot(u850, v850) * r850 / 100.0
        np.nan_to_num(mfc, copy=False)

        # ── RH layers ────────────────────────────────────────────────────────
        rh_01km = r925                           # surface-1km ≈ 925 hPa
        rh_02km = (r925 + r850) / 2.0           # surface-2km
        rh_14km = (r850 + r700) / 2.0           # 1-4km
        rh_25km = (r850 + r700) / 2.0           # 2-5km proxy
        rh_36km = (r700 + r850) / 2.0 * 0.9    # 3-6km proxy

        # ── Freezing level height ─────────────────────────────────────────────
        frzg_hgt = np.where(t925 > 0, t925 / np.maximum(lr_85_50, 0.1) * 1000.0, 0.0)
        np.nan_to_num(frzg_hgt, copy=False)

        # ── LCL / BRN ────────────────────────────────────────────────────────
        lcl_h = mu_lcl
        brn   = compute_brn(mucape, sh6)

        # ── SRH (Bunkers method) ─────────────────────────────────────────────
        srh_01, srh_03 = compute_srh_proxy(u925, v925, u850, v850, u700, v700)
        # Sub-km SRH proxies
        srh_500m  = srh_01 * 0.5
        srh_250m  = srh_01 * 0.25
        srh_100m  = srh_01 * 0.10
        srh_36km  = srh_03 * 0.3    # 3-6km SRH proxy
        # Left-mover SRH (opposite sign convention)
        srh_01_lm = -srh_01
        srh_03_lm = -srh_03
        srh_500m_lm = -srh_500m
        srh_250m_lm = -srh_250m
        srh_100m_lm = -srh_100m
        srh_36km_lm = -srh_36km

        # ── Bunkers storm motion ───────────────────────────────────────────────
        # Using our existing Bunkers proxy calculation
        u_mean = (u925 + u850 + u700) / 3.0
        v_mean = (v925 + v850 + v700) / 3.0
        u_shear = u700 - u925
        v_shear = v700 - v925
        shear_mag = np.hypot(u_shear, v_shear) + 1e-6
        d = 7.5
        # Right-mover
        u_rm = u_mean + d * v_shear / shear_mag
        v_rm = v_mean - d * u_shear / shear_mag
        bunkers_rm_m = np.hypot(u_rm, v_rm)
        bunkers_rm_a = np.degrees(np.arctan2(-u_rm, -v_rm)) % 360
        # Left-mover
        u_lm = u_mean - d * v_shear / shear_mag
        v_lm = v_mean + d * u_shear / shear_mag
        bunkers_lm_m = np.hypot(u_lm, v_lm)
        bunkers_lm_a = np.degrees(np.arctan2(-u_lm, -v_lm)) % 360
        # Mean wind
        bunkers_mw_m = np.hypot(u_mean, v_mean)
        bunkers_mw_a = np.degrees(np.arctan2(-u_mean, -v_mean)) % 360
        np.nan_to_num(bunkers_rm_m, copy=False); np.nan_to_num(bunkers_rm_a, copy=False)
        np.nan_to_num(bunkers_lm_m, copy=False); np.nan_to_num(bunkers_lm_a, copy=False)

        # ── Corfidi vectors ──────────────────────────────────────────────────
        # Downwind = mean 850-500 wind; Upwind = LLJ (925 hPa)
        u_dw = (u850 + u500) / 2.0; v_dw = (v850 + v500) / 2.0
        u_uw = -u925; v_uw = -v925
        corfidi_dw_m = np.hypot(u_dw, v_dw)
        corfidi_dw_a = np.degrees(np.arctan2(-u_dw, -v_dw)) % 360
        corfidi_uw_m = np.hypot(u_uw, v_uw)
        corfidi_uw_a = np.degrees(np.arctan2(-u_uw, -v_uw)) % 360
        np.nan_to_num(corfidi_dw_m, copy=False); np.nan_to_num(corfidi_uw_m, copy=False)

        # ── Bulk shear layers ────────────────────────────────────────────────
        sh3   = np.hypot(u700 - u925, v700 - v925)  # 0-3km (925→700)
        sh6   = np.hypot(u250 - u850, v250 - v850)  # 0-6km (850→250)
        sh1   = np.hypot(u850 - u925, v850 - v925)  # 0-1km (925→850)
        sh2   = sh1 + np.hypot(u700 - u850, v700 - v850) * 0.5   # 0-2km proxy
        sh8   = sh6 * 1.1   # 0-8km proxy
        sh36  = np.hypot(u250 - u700, v250 - v700)  # 3-6km (700→250)
        sh26  = np.hypot(u250 - u850, v250 - v850)  # 2-6km (850→250)
        sh16  = sh6   # 1-6km ≈ 0-6km proxy
        sh18  = sh8   # 1-8km proxy
        np.nan_to_num(sh1, copy=False); np.nan_to_num(sh2, copy=False)
        np.nan_to_num(sh3, copy=False); np.nan_to_num(sh6, copy=False)

        # ── Effective shear proxies ───────────────────────────────────────────
        bs_eff_mu = sh6 * np.where(mu_lcl < 2000, 1.0, 0.7)
        bs_eff_sb = sh6 * np.where(sb_lcl < 2000, 1.0, 0.7)
        bs_eff_ml = sh6 * np.where(ml_lcl < 2000, 1.0, 0.7)

        # ── Mean winds ───────────────────────────────────────────────────────
        mw_06km = np.hypot(u_mean, v_mean)
        mw_03km = np.hypot((u925+u850+u700)/3.0, (v925+v850+v700)/3.0)
        mw_02km = np.hypot((u925+u850)/2.0, (v925+v850)/2.0)
        mw_01km = np.hypot((u925+u850)/2.0, (v925+v850)/2.0) * 0.7
        mw_500m = np.hypot(u925, v925) * 0.5
        mw_13km = np.hypot((u850+u700)/2.0, (v850+v700)/2.0)
        np.nan_to_num(mw_06km, copy=False)

        # ── Streamwise vorticity (SV) ─────────────────────────────────────────
        # SV = SRH / (layer depth × wind speed)
        sv_500m_rm  = srh_500m / np.maximum(mw_500m * 0.5 * 1000, 1.0)
        sv_01km_rm  = srh_01   / np.maximum(mw_01km * 1.0 * 1000, 1.0)
        sv_03km_rm  = srh_03   / np.maximum(mw_03km * 3.0 * 1000, 1.0)
        sv_500m_lm  = -sv_500m_rm; sv_01km_lm = -sv_01km_rm; sv_03km_lm = -sv_03km_rm
        np.nan_to_num(sv_500m_rm, copy=False); np.nan_to_num(sv_01km_rm, copy=False); np.nan_to_num(sv_03km_rm, copy=False)

        # SV fraction
        sv_fra_500m_rm = np.clip(sv_500m_rm / np.maximum(mw_500m/0.5, 1e-3), 0, 1)
        sv_fra_01km_rm = np.clip(sv_01km_rm / np.maximum(mw_01km/1.0, 1e-3), 0, 1)
        sv_fra_03km_rm = np.clip(sv_03km_rm / np.maximum(mw_03km/3.0, 1e-3), 0, 1)
        sv_fra_500m_lm = -sv_fra_500m_rm; sv_fra_01km_lm = -sv_fra_01km_rm; sv_fra_03km_lm = -sv_fra_03km_rm

        # ── Storm-relative mean winds ─────────────────────────────────────────
        mw_sr_500m_rm  = np.hypot(u925 - u_rm, v925 - v_rm)
        mw_sr_01km_rm  = np.hypot(u850 - u_rm, v850 - v_rm)
        mw_sr_03km_rm  = np.hypot(u700 - u_rm, v700 - v_rm)
        mw_sr_500m_lm  = np.hypot(u925 - u_lm, v925 - v_lm)
        mw_sr_01km_lm  = np.hypot(u850 - u_lm, v850 - v_lm)
        mw_sr_03km_lm  = np.hypot(u700 - u_lm, v700 - v_lm)
        np.nan_to_num(mw_sr_500m_rm, copy=False); np.nan_to_num(mw_sr_01km_rm, copy=False)

        # ── Composite parameters ──────────────────────────────────────────────
        # STP (Significant Tornado Parameter — STP_new Coffer 2019)
        stp = compute_stp(mucape, sh6, srh_01, lcl_h, mucin)
        # STP_fix (fixed-layer, surface-based)
        stp_fix = (np.maximum(sb_cape, 0) / 1500.0 *
                   np.maximum(sh6, 0) / 20.0 *
                   np.maximum(2000.0 - sb_lcl, 0) / 1000.0 *
                   np.maximum(srh_03, 0) / 150.0)
        # Left-mover versions
        stp_fix_lm = stp_fix * np.maximum(-srh_01, 0) / np.maximum(np.abs(srh_01) + 1e-6, 1)
        stp_new_lm = stp * np.maximum(-srh_01, 0) / np.maximum(np.abs(srh_01) + 1e-6, 1)
        np.nan_to_num(stp, copy=False); np.nan_to_num(stp_fix, copy=False)

        # SCP — Supercell Composite Parameter
        scp_fix = (np.maximum(mu_cape, 0) / 1000.0 *
                   np.maximum(srh_03, 0) / 50.0 *
                   np.maximum(sh6, 0) / 20.0)
        # SCP_new (Gropp & Davenport 2018): uses CIN term
        scp_new = (np.maximum(mu_cape, 0) / 1000.0 *
                   np.maximum(srh_03, 0) / 50.0 *
                   np.maximum(bs_eff_mu, 0) / 20.0 *
                   np.where(mucin > -150, 1.0, np.maximum((-150 - mucin) / 50.0, 0)))
        scp_fix_lm = scp_fix * np.maximum(-srh_03, 0) / np.maximum(np.abs(srh_03) + 1e-6, 1)
        scp_new_lm = scp_new * np.maximum(-srh_03, 0) / np.maximum(np.abs(srh_03) + 1e-6, 1)
        np.nan_to_num(scp_fix, copy=False); np.nan_to_num(scp_new, copy=False)

        # EHI variants
        ehi   = compute_ehi(mu_cape, srh_01)
        ehi3  = compute_ehi(mu_cape, srh_03)
        ehi_500m = mu_cape * np.maximum(srh_500m, 0) / 160000.0
        ehi_500m_lm = mu_cape * np.maximum(-srh_500m, 0) / 160000.0
        ehi_lm = mu_cape * np.maximum(-srh_01, 0) / 160000.0
        ehi3_lm = mu_cape * np.maximum(-srh_03, 0) / 160000.0
        np.nan_to_num(ehi, copy=False); np.nan_to_num(ehi3, copy=False)

        # SHIP — Significant Hail Parameter
        ship = (np.maximum(mu_cape, 0) * np.maximum(mixr_850, 0) *
                np.maximum(lr_70_50, 0) * np.maximum(-t500, 0) *
                np.maximum(sh6, 0)) / 42e6
        np.nan_to_num(ship, copy=False)

        # HSI — Hail Size Index (Czernecki et al. 2019)
        hsi = (np.maximum(mu_cape, 0) ** 0.5 *
               np.maximum(lr_70_50, 0) *
               np.maximum(sh6, 0)) / 1000.0
        np.nan_to_num(hsi, copy=False)

        # WMAXSHEAR variants (Taszarek et al. 2020)
        mu_wmaxshear  = mu_wmax * sh6
        sb_wmaxshear  = sb_wmax * sh6
        ml_wmaxshear  = ml_wmax * sh6
        mu_eff_wmaxshear = mu_wmax * bs_eff_mu
        sb_eff_wmaxshear = sb_wmax * bs_eff_sb
        ml_eff_wmaxshear = ml_wmax * bs_eff_ml
        np.nan_to_num(mu_wmaxshear, copy=False)

        # DCP — Derecho Composite Parameter
        dcp = (np.maximum(dcape, 0) / 980.0 *
               np.maximum(mu_cape, 0) / 2000.0 *
               np.maximum(sh6, 0) / 10.0 *
               np.maximum(mw_06km, 0) / 15.0)
        np.nan_to_num(dcp, copy=False)

        # DEI — Downburst Environment Index
        dei = mu_wmaxshear * cold_pool / 1000.0
        dei_eff = mu_eff_wmaxshear * cold_pool / 1000.0
        np.nan_to_num(dei, copy=False); np.nan_to_num(dei_eff, copy=False)

        # SHERBS3 / SHERBE (Sherburn & Parker 2014)
        sherbs3 = (np.maximum(lr_70_50, 0) / 5.5 *
                   np.maximum(mixr_850, 0) / 11.0 *
                   np.maximum(sh3, 0) / 27.0)
        sherbe  = (np.maximum(lr_70_50, 0) / 5.5 *
                   np.maximum(mixr_850, 0) / 11.0 *
                   np.maximum(bs_eff_mu, 0) / 27.0)
        # v2 variants use max 2-6km LR instead of 700-500 LR
        lr_26km_max = np.maximum(lr_85_50, lr_70_50)
        sherbs3_v2 = (np.maximum(lr_26km_max, 0) / 5.5 *
                      np.maximum(mixr_850, 0) / 11.0 *
                      np.maximum(sh3, 0) / 27.0)
        sherbe_v2  = (np.maximum(lr_26km_max, 0) / 5.5 *
                      np.maximum(mixr_850, 0) / 11.0 *
                      np.maximum(bs_eff_mu, 0) / 27.0)
        np.nan_to_num(sherbs3, copy=False); np.nan_to_num(sherbe, copy=False)

        # TIP — Thunderstorm Intensity Parameter (experimental)
        tip = (np.maximum(mu_cape, 0) / 2000.0 *
               np.maximum(sh6, 0) / 20.0 *
               np.maximum(pwat, 0) / 30.0 *
               np.maximum(srh_03, 0) / 150.0)
        np.nan_to_num(tip, copy=False)

        # BRN
        brn = compute_brn(mu_cape, sh6)

        valid_dt  = base + timedelta(hours=step)
        month_val = valid_dt.month
        hour_val  = valid_dt.hour

        ml_probs = None
        if ml:
            ml_probs = ml_score_grid(
                ml, mucape, mucin, kx, t850, t700, t500, r850, r700,
                u850, v850, u250, v250, sh6,
                month=month_val, hour=hour_val,
                lats=flat_la, lons=flat_lo
            )

        def to_lists(obj):
            if isinstance(obj, dict):
                return {k: to_lists(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_lists(v) for v in obj]
            if hasattr(obj, 'dtype'):
                cleaned = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
                return [round(float(x), 3) for x in cleaned]
            return obj

        def clean(arr, default=0.0, decimals=2):
            return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).round(decimals).astype(np.float32)

        step_data = {
            # ── Parcel parameters ────────────────────────────────────────────
            'MU_CAPE': clean(mu_cape, 0, 1), 'MU_CAPE_M10': clean(mu_cape_m10, 0, 1),
            'MU_02km_CAPE': clean(mu_02cape, 0, 1), 'MU_03km_CAPE': clean(mu_03cape, 0, 1),
            'MU_HGL_CAPE': clean(mu_hgl, 0, 1), 'MU_CIN': clean(mu_cin, 0, 1),
            'MU_LCL_HGT': clean(mu_lcl, 0, 0), 'MU_LFC_HGT': clean(mu_lfc_hgt, 0, 0),
            'MU_EL_HGT': clean(mu_el_hgt, 0, 0), 'MU_LI': clean(mu_li, 0, 1),
            'MU_WMAX': clean(mu_wmax, 0, 1), 'MU_EL_TEMP': clean(mu_el_temp, 0, 1),
            'MU_LCL_TEMP': clean(mu_lcl_temp, 0, 1), 'MU_MIXR': clean(mixr_sfc, 0, 2),
            'SB_CAPE': clean(sb_cape, 0, 1), 'SB_CAPE_M10': clean(sb_cape_m10, 0, 1),
            'SB_02km_CAPE': clean(sb_02cape, 0, 1), 'SB_03km_CAPE': clean(sb_03cape, 0, 1),
            'SB_HGL_CAPE': clean(sb_hgl, 0, 1), 'SB_CIN': clean(sb_cin, 0, 1),
            'SB_LCL_HGT': clean(sb_lcl, 0, 0), 'SB_LFC_HGT': clean(sb_lfc_hgt, 0, 0),
            'SB_EL_HGT': clean(sb_el_hgt, 0, 0), 'SB_LI': clean(sb_li, 0, 1),
            'SB_WMAX': clean(sb_wmax, 0, 1), 'SB_EL_TEMP': clean(sb_el_temp, 0, 1),
            'SB_LCL_TEMP': clean(sb_lcl_temp, 0, 1), 'SB_MIXR': clean(mixr_sfc, 0, 2),
            'ML_CAPE': clean(ml_cape, 0, 1), 'ML_CAPE_M10': clean(ml_cape_m10, 0, 1),
            'ML_02km_CAPE': clean(ml_02cape, 0, 1), 'ML_03km_CAPE': clean(ml_03cape, 0, 1),
            'ML_HGL_CAPE': clean(ml_hgl, 0, 1), 'ML_CIN': clean(ml_cin, 0, 1),
            'ML_LCL_HGT': clean(ml_lcl, 0, 0), 'ML_LFC_HGT': clean(ml_lfc_hgt, 0, 0),
            'ML_EL_HGT': clean(ml_el_hgt, 0, 0), 'ML_LI': clean(ml_li, 0, 1),
            'ML_WMAX': clean(ml_wmax, 0, 1), 'ML_EL_TEMP': clean(ml_el_temp, 0, 1),
            'ML_LCL_TEMP': clean(ml_lcl_temp, 0, 1), 'ML_MIXR': clean(mixr_850, 0, 2),
            # ── Temperature & moisture ────────────────────────────────────────
            'LR_0500m': clean(lr_92_85, 0, 2), 'LR_01km': clean(lr_92_85, 0, 2),
            'LR_02km': clean(lr_85_70*0.7+lr_92_85*0.3, 0, 2), 'LR_03km': clean(lr_85_70, 0, 2),
            'LR_04km': clean(lr_85_70*0.8+lr_70_50*0.2, 0, 2), 'LR_06km': clean(lr_06km, 0, 2),
            'LR_16km': clean(lr_85_50, 0, 2), 'LR_26km': clean(lr_85_50, 0, 2),
            'LR_24km': clean(lr_85_70, 0, 2), 'LR_36km': clean(lr_70_50, 0, 2),
            'LR_26km_MAX': clean(lr_26km_max, 0, 2),
            'LR_500700hPa': clean(lr_70_50, 0, 2), 'LR_500800hPa': clean(lr_85_50, 0, 2),
            'LR_600800hPa': clean(lr_85_70, 0, 2),
            'FRZG_HGT': clean(frzg_hgt, 0, 0), 'FRZG_wetbulb_HGT': clean(frzg_hgt*0.9, 0, 0),
            'HGT_max_thetae_03km': clean(mu_lcl, 0, 0),
            'HGT_min_thetae_04km': clean(mu_lcl*1.5, 0, 0),
            'Delta_thetae': clean(delta_thetae, 0, 1), 'Delta_thetae_04km': clean(delta_thetae_04, 0, 1),
            'Thetae_01km': clean(thetae_sfc, 0, 1), 'Thetae_02km': clean((thetae_sfc+thetae_850)/2, 0, 1),
            'DCAPE': clean(dcape, 0, 0), 'Cold_Pool_Strength': clean(cold_pool, 0, 1),
            'Wind_Index': clean(wind_index, 0, 1), 'PRCP_WATER': clean(pwat, 0, 1),
            'Moisture_Flux_02km': clean(mfc, 0, 2),
            'RH_01km': clean(rh_01km, 0, 1), 'RH_02km': clean(rh_02km, 0, 1),
            'RH_14km': clean(rh_14km, 0, 1), 'RH_25km': clean(rh_25km, 0, 1),
            'RH_36km': clean(rh_36km, 0, 1), 'RH_HGL': clean(rh_25km, 0, 1),
            # ── Kinematic ────────────────────────────────────────────────────
            'BS_0500m': clean(sh1*0.4, 0, 2), 'BS_01km': clean(sh1, 0, 2),
            'BS_02km': clean(sh2, 0, 2), 'BS_03km': clean(sh3, 0, 2),
            'BS_06km': clean(sh6, 0, 2), 'BS_08km': clean(sh8, 0, 2),
            'BS_36km': clean(sh36, 0, 2), 'BS_26km': clean(sh26, 0, 2),
            'BS_16km': clean(sh16, 0, 2), 'BS_18km': clean(sh18, 0, 2),
            'BS_EFF_MU': clean(bs_eff_mu, 0, 2), 'BS_EFF_SB': clean(bs_eff_sb, 0, 2),
            'BS_EFF_ML': clean(bs_eff_ml, 0, 2),
            'BS_SFC_to_M10': clean(sh6*0.8, 0, 2), 'BS_1km_to_M10': clean(sh6*0.7, 0, 2),
            'BS_2km_to_M10': clean(sh6*0.6, 0, 2),
            'BS_MU_LFC_to_M10': clean(sh6*0.5, 0, 2), 'BS_SB_LFC_to_M10': clean(sh6*0.5, 0, 2),
            'BS_ML_LFC_to_M10': clean(sh6*0.5, 0, 2),
            'BS_MW02_SM': clean(sh2, 0, 2), 'BS_MW02_RM': clean(mw_sr_01km_rm, 0, 2),
            'BS_MW02_LM': clean(mw_sr_01km_lm, 0, 2),
            'BS_HGL_SM': clean(sh36*0.8, 0, 2), 'BS_HGL_RM': clean(mw_sr_03km_rm, 0, 2),
            'BS_HGL_LM': clean(mw_sr_03km_lm, 0, 2),
            'MW_0500m': clean(mw_500m, 0, 2), 'MW_01km': clean(mw_01km, 0, 2),
            'MW_02km': clean(mw_02km, 0, 2), 'MW_03km': clean(mw_03km, 0, 2),
            'MW_06km': clean(mw_06km, 0, 2), 'MW_13km': clean(mw_13km, 0, 2),
            'SRH_100m_RM': clean(srh_100m, 0, 1), 'SRH_250m_RM': clean(srh_250m, 0, 1),
            'SRH_500m_RM': clean(srh_500m, 0, 1), 'SRH_1km_RM': clean(srh_01, 0, 1),
            'SRH_3km_RM': clean(srh_03, 0, 1), 'SRH_36km_RM': clean(srh_36km, 0, 1),
            'SRH_100m_LM': clean(srh_100m_lm, 0, 1), 'SRH_250m_LM': clean(srh_250m_lm, 0, 1),
            'SRH_500m_LM': clean(srh_500m_lm, 0, 1), 'SRH_1km_LM': clean(srh_01_lm, 0, 1),
            'SRH_3km_LM': clean(srh_03_lm, 0, 1), 'SRH_36km_LM': clean(srh_36km_lm, 0, 1),
            'SV_500m_RM': clean(sv_500m_rm, 0, 4), 'SV_01km_RM': clean(sv_01km_rm, 0, 4),
            'SV_03km_RM': clean(sv_03km_rm, 0, 4),
            'SV_500m_LM': clean(sv_500m_lm, 0, 4), 'SV_01km_LM': clean(sv_01km_lm, 0, 4),
            'SV_03km_LM': clean(sv_03km_lm, 0, 4),
            'MW_SR_500m_RM': clean(mw_sr_500m_rm, 0, 2), 'MW_SR_01km_RM': clean(mw_sr_01km_rm, 0, 2),
            'MW_SR_03km_RM': clean(mw_sr_03km_rm, 0, 2),
            'MW_SR_500m_LM': clean(mw_sr_500m_lm, 0, 2), 'MW_SR_01km_LM': clean(mw_sr_01km_lm, 0, 2),
            'MW_SR_03km_LM': clean(mw_sr_03km_lm, 0, 2),
            'MW_SR_VM_500m_RM': clean(mw_sr_500m_rm, 0, 2), 'MW_SR_VM_01km_RM': clean(mw_sr_01km_rm, 0, 2),
            'MW_SR_VM_03km_RM': clean(mw_sr_03km_rm, 0, 2),
            'MW_SR_VM_500m_LM': clean(mw_sr_500m_lm, 0, 2), 'MW_SR_VM_01km_LM': clean(mw_sr_01km_lm, 0, 2),
            'MW_SR_VM_03km_LM': clean(mw_sr_03km_lm, 0, 2),
            'SV_FRA_500m_RM': clean(sv_fra_500m_rm, 0, 3), 'SV_FRA_01km_RM': clean(sv_fra_01km_rm, 0, 3),
            'SV_FRA_03km_RM': clean(sv_fra_03km_rm, 0, 3),
            'SV_FRA_500m_LM': clean(sv_fra_500m_lm, 0, 3), 'SV_FRA_01km_LM': clean(sv_fra_01km_lm, 0, 3),
            'SV_FRA_03km_LM': clean(sv_fra_03km_lm, 0, 3),
            'Bunkers_RM_A': clean(bunkers_rm_a, 0, 1), 'Bunkers_RM_M': clean(bunkers_rm_m, 0, 2),
            'Bunkers_LM_A': clean(bunkers_lm_a, 0, 1), 'Bunkers_LM_M': clean(bunkers_lm_m, 0, 2),
            'Bunkers_MW_A': clean(bunkers_mw_a, 0, 1), 'Bunkers_MW_M': clean(bunkers_mw_m, 0, 2),
            'Corfidi_downwind_A': clean(corfidi_dw_a, 0, 1), 'Corfidi_downwind_M': clean(corfidi_dw_m, 0, 2),
            'Corfidi_upwind_A': clean(corfidi_uw_a, 0, 1), 'Corfidi_upwind_M': clean(corfidi_uw_m, 0, 2),
            # ── Composite ────────────────────────────────────────────────────
            'K_Index': clean(kx, 0, 1), 'Showalter_Index': clean(si, 0, 1),
            'TotalTotals_Index': clean(tt, 0, 1), 'SWEAT_Index': clean(sweat, 0, 0),
            'STP_fix': clean(stp_fix, 0, 3), 'STP_new': clean(stp, 0, 3),
            'STP_fix_LM': clean(stp_fix_lm, 0, 3), 'STP_new_LM': clean(stp_new_lm, 0, 3),
            'SCP_fix': clean(scp_fix, 0, 3), 'SCP_new': clean(scp_new, 0, 3),
            'SCP_fix_LM': clean(scp_fix_lm, 0, 3), 'SCP_new_LM': clean(scp_new_lm, 0, 3),
            'SHIP': clean(ship, 0, 3), 'HSI': clean(hsi, 0, 2), 'DCP': clean(dcp, 0, 3),
            'MU_WMAXSHEAR': clean(mu_wmaxshear, 0, 0), 'SB_WMAXSHEAR': clean(sb_wmaxshear, 0, 0),
            'ML_WMAXSHEAR': clean(ml_wmaxshear, 0, 0),
            'MU_EFF_WMAXSHEAR': clean(mu_eff_wmaxshear, 0, 0),
            'SB_EFF_WMAXSHEAR': clean(sb_eff_wmaxshear, 0, 0),
            'ML_EFF_WMAXSHEAR': clean(ml_eff_wmaxshear, 0, 0),
            'EHI_500m': clean(ehi_500m, 0, 3), 'EHI_01km': clean(ehi, 0, 3),
            'EHI_03km': clean(ehi3, 0, 3),
            'EHI_500m_LM': clean(ehi_500m_lm, 0, 3), 'EHI_01km_LM': clean(ehi_lm, 0, 3),
            'EHI_03km_LM': clean(ehi3_lm, 0, 3),
            'SHERBS3': clean(sherbs3, 0, 3), 'SHERBE': clean(sherbe, 0, 3),
            'SHERBS3_v2': clean(sherbs3_v2, 0, 3), 'SHERBE_v2': clean(sherbe_v2, 0, 3),
            'DEI': clean(dei, 0, 3), 'DEI_eff': clean(dei_eff, 0, 3),
            'TIP': clean(tip, 0, 3), 'BRN': clean(brn, 0, 1),
            # ── Raw model fields needed for scoring ───────────────────────────
            'cape': clean(mucape, 0, 1), 'cin': clean(mucin, 0, 1),
            'kx': clean(kx, 0, 1), 'sh6': clean(sh6, 0, 2), 'sh1': clean(sh1, 0, 2),
            'srh1': clean(srh_01, 0, 1), 'srh3': clean(srh_03, 0, 1),
            'stp': clean(stp, 0, 3), 'ehi': clean(ehi, 0, 3),
            't850': clean(t850, 0, 2), 't700': clean(t700, 0, 2),
            't500': clean(t500, 0, 2), 't925': clean(t925, 0, 2),
            'r850': clean(r850, 50, 1), 'r925': clean(r925, 70, 1), 'r700': clean(r700, 50, 1),
            'td850': clean(td850, 0, 2), 'td925': clean(td925, 0, 2),
            'li': clean(mu_li, 0, 1), 'tt': clean(tt, 0, 1),
        }

        # LCL height (Bolton 1980)
        lcl_h = compute_lcl_height(t925, td925)

        # SRH 0-1km and 0-3km (Bunkers method)
        srh_01, srh_03 = compute_srh_proxy(u925, v925, u850, v850, u700, v700)

        # EHI, STP, BRN, DCAPE
        ehi   = compute_ehi(mucape, srh_01)
        stp   = compute_stp(mucape, sh6, srh_01, lcl_h, mucin)
        brn   = compute_brn(mucape, sh6)
        dcape = compute_dcape(t700, t500, r850)

        # Total Totals Index: TT = T850 + Td850 - 2*T500
        tt = t850 + td850 - 2.0 * t500
        np.nan_to_num(tt, copy=False)

        # SWEAT Index (proxy — uses TT and 850 wind speed)
        ff850 = np.hypot(u850, v850)  # wind speed at 850 hPa
        sweat = np.maximum(12.0 * td850, 0) + 20.0 * np.maximum(tt - 49.0, 0) + 2.0 * ff850
        np.nan_to_num(sweat, copy=False)

        # Lifted Index proxy: LI ≈ T500 - T_parcel_500
        # Parcel lifted from 925 hPa: T_parcel ≈ t925 - DALR * (500-925)*10m/hPa
        # DALR = 9.8 K/km, 425 hPa ≈ 3.5 km → ~34.3 K cooling
        t_parcel_500 = t925 - 34.3
        li = t500 - t_parcel_500
        np.nan_to_num(li, copy=False)

        # Showalter Index: SI = T500 - T_parcel_lifted_from_850
        # Parcel from 850 hPa cools 6.5 K/km over ~3 km → ~19.5 K
        t_parcel_500_sfc = t850 - 19.5
        si = t500 - t_parcel_500_sfc
        np.nan_to_num(si, copy=False)

        # PWAT proxy: integration of specific humidity (use RH as proxy)
        # Full PWAT = ∫q dp/g; we approximate from mean RH across layers
        pwat = (r925 + r850 + r700) / 3.0 * 0.5  # rough mm equivalent
        np.nan_to_num(pwat, copy=False)

        # CAPE 0-3km proxy: fraction of MUCAPE below 3km
        # Approximate: if LCL is low (<1000m) and instability is strong
        cape03 = np.where(lcl_h < 1500, mucape * 0.3, mucape * 0.15)
        np.nan_to_num(cape03, copy=False)

        # Moisture Flux (u*q proxy): wind × moisture at 850 hPa
        mfc = np.hypot(u850, v850) * r850 / 100.0
        np.nan_to_num(mfc, copy=False)

        # Lapse rate 700-500 hPa (mid-level instability)
        lr75 = (t700 - t500) / 2.5
        np.nan_to_num(lr75, copy=False)

        # CAPE 0-3km proxy
        cape03 = np.where(lcl_h < 1500, mucape * 0.3, mucape * 0.15)

        # Moisture Flux
        mfc = np.hypot(u850, v850) * r850 / 100.0

        # WMAX — estimated max updraft speed (sqrt(2*CAPE))
        wmax = np.sqrt(2.0 * np.maximum(mucape, 0))
        np.nan_to_num(wmax, copy=False)

        # WMAXSHEAR = WMAX × BS_06km (Taszarek et al. 2020)
        wmaxshear = wmax * sh6
        np.nan_to_num(wmaxshear, copy=False)

        # SHIP — Significant Hail Parameter (SPC mesoanalysis formula)
        # SHIP = (MU_CAPE * mixr * LR700-500 * (-T500) * sh6) / 42e6
        lr_70_50 = (t700 - t500) / 2.5
        mixr = r850 * 0.622 / (1000.0 - r850 * 10.0)  # rough mixing ratio g/kg
        ship = (np.maximum(mucape, 0) * np.maximum(mixr, 0) *
                np.maximum(lr_70_50, 0) * np.maximum(-t500, 0) *
                np.maximum(sh6, 0)) / 42e6
        np.nan_to_num(ship, copy=False)

        # SCP_fix — Supercell Composite Parameter fixed-layer
        # SCP = (MU_CAPE/1000) × (SRH_03km/50) × (sh6/20)
        scp = (np.maximum(mucape, 0) / 1000.0 *
               np.maximum(srh_03, 0) / 50.0 *
               np.maximum(sh6, 0) / 20.0)
        np.nan_to_num(scp, copy=False)

        # Cold Pool Strength proxy (from DCAPE)
        cold_pool = np.sqrt(2.0 * np.maximum(dcape, 0)) * 0.5
        np.nan_to_num(cold_pool, copy=False)

        # Wind Index (McCann 1994) — estimated gust in m/s
        lr_85_50 = (t850 - t500) / 3.5
        wind_index = np.sqrt(np.maximum(dcape, 0) / 2.0) * np.maximum(lr_85_50, 0) / 5.0
        np.nan_to_num(wind_index, copy=False)

        # DEI — Downburst Environment Index (Romanic et al. 2022)
        dei = wmaxshear * cold_pool / 1000.0
        np.nan_to_num(dei, copy=False)

        # SHERBS3 — Sherburn & Parker (2014)
        sh3 = np.hypot(u700 - u925, v700 - v925)
        sherbs3 = (np.maximum(lr_70_50, 0) / 5.5 *
                   np.maximum(mixr, 0) / 11.0 *
                   np.maximum(sh3, 0) / 27.0)
        np.nan_to_num(sherbs3, copy=False)

        # Freezing level height proxy
        frzg_hgt = np.where(t925 > 0, t925 / np.maximum(lr_85_50, 0.1) * 1000.0, 0.0)
        np.nan_to_num(frzg_hgt, copy=False)

        # EHI 0-3km
        ehi3 = compute_ehi(mucape, srh_03)

        valid_dt  = base + timedelta(hours=step)
        month_val = valid_dt.month
        hour_val  = valid_dt.hour

        ml_probs = None
        if ml:
            ml_probs = ml_score_grid(
                ml, mucape, mucin, kx, t850, t700, t500, r850, r700,
                u850, v850, u250, v250, sh6,
                month=month_val, hour=hour_val,
                lats=flat_la, lons=flat_lo
            )

        def to_lists(obj):
            if isinstance(obj, dict):
                return {k: to_lists(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_lists(v) for v in obj]
            if hasattr(obj, 'dtype'):
                cleaned = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
                return [round(float(x), 3) for x in cleaned]
            return obj

        def clean(arr, default=0.0, decimals=2):
            return np.nan_to_num(arr, nan=default, posinf=default, neginf=default).round(decimals).astype(np.float32)

        step_data = {
            # Core instability
            'cape':  clean(mucape,  0.0, 1),
            'cin':   clean(mucin,   0.0, 1),
            'kx':    clean(kx,      0.0, 1),
            'li':    clean(li,      0.0, 1),
            'si':    clean(si,      0.0, 1),
            'tt':    clean(tt,      0.0, 1),
            'sweat': clean(sweat,   0.0, 0),
            'pwat':  clean(pwat,    0.0, 1),
            'cape03':clean(cape03,  0.0, 0),
            'wmax':  clean(wmax,    0.0, 1),
            # Temperature/lapse rate
            't925':  clean(t925,    0.0, 2),
            't850':  clean(t850,    0.0, 2),
            't700':  clean(t700,    0.0, 2),
            't500':  clean(t500,    0.0, 2),
            'td850': clean(td850,   0.0, 2),
            'td925': clean(td925,   0.0, 2),
            'lr_lapse': clean(lr_85_50, 0.0, 2),
            'lr75':  clean(lr75,    0.0, 2),
            'frzg':  clean(frzg_hgt, 0.0, 0),
            # Moisture
            'r925':  clean(r925,   70.0, 1),
            'r850':  clean(r850,   50.0, 1),
            'r700':  clean(r700,   50.0, 1),
            'mfc':   clean(mfc,     0.0, 2),
            # Wind shear
            'sh6':   clean(sh6,     0.0, 2),
            'sh1':   clean(sh1,     0.0, 2),
            # thundeR composite indices
            'lcl':   clean(lcl_h,   0.0, 0),
            'srh1':  clean(srh_01,  0.0, 1),
            'srh3':  clean(srh_03,  0.0, 1),
            'ehi':   clean(ehi,     0.0, 3),
            'ehi3':  clean(ehi3,    0.0, 3),
            'stp':   clean(stp,     0.0, 3),
            'scp':   clean(scp,     0.0, 3),
            'ship':  clean(ship,    0.0, 3),
            'brn':   clean(brn,     0.0, 1),
            'dcape': clean(dcape,   0.0, 0),
            'wmaxshear': clean(wmaxshear, 0.0, 0),
            'dei':   clean(dei,     0.0, 3),
            'sherbs3': clean(sherbs3, 0.0, 3),
            'cold_pool': clean(cold_pool, 0.0, 1),
            'wind_index': clean(wind_index, 0.0, 1),
        }
        if ml_probs:
            for k in ['tstm', 'hail', 'wind', 'tornado']:
                step_data['ml_' + k] = clean(ml_probs[k], 0.0, 3)

        data_out.append(to_lists(step_data))

    return {
        'times':   times_out,
        'run':     list(run),
        'lats':    flat_la.tolist(),
        'lons':    flat_lo.tolist(),
        'data':    data_out,
        'has_ml':  ml is not None,
        'columnar': True,  # flag: data is columnar arrays, not per-point dicts
    }



# ═══════════════════════════════════════════════════════════════════════════════
# OPEN-METEO GRID FETCHER  (shared by CMC, ARPEGE, AROME)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_openmeteo_grid(model_name, ref_lats, ref_lons,
                           sample_step=1.0, max_workers=80,
                           forecast_hours=72, label='OpenMeteo',
                           lat_min=None, lat_max=None,
                           lon_min=None, lon_max=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from scipy.spatial import cKDTree

    _lat_min = lat_min if lat_min is not None else LAT_MIN
    _lat_max = lat_max if lat_max is not None else LAT_MAX
    _lon_min = lon_min if lon_min is not None else LON_MIN
    _lon_max = lon_max if lon_max is not None else LON_MAX
    s_lats = np.arange(_lat_min, _lat_max + sample_step, sample_step)
    s_lons = np.arange(_lon_min, _lon_max + sample_step, sample_step)
    SLG, SOG = np.meshgrid(s_lats, s_lons, indexing='ij')
    src_la = SLG.ravel(); src_lo = SOG.ravel(); n_src = len(src_la)

    PL_VARS = []
    for lv in [925, 850, 700, 500, 250]:
        PL_VARS += [f'temperature_{lv}hPa',
                    f'wind_u_component_{lv}hPa',
                    f'wind_v_component_{lv}hPa']
    for lv in [925, 850, 700]:
        PL_VARS.append(f'relative_humidity_{lv}hPa')
    ALL_VARS = ['cape'] + PL_VARS

    BASE_URL = 'https://api.open-meteo.com/v1/forecast'
    PARAMS_T = {
        'hourly': ','.join(ALL_VARS), 'models': model_name,
        'forecast_days': int(forecast_hours/24)+1,
        'wind_speed_unit': 'ms', 'timeformat': 'unixtime',
    }

    def parse_resp(data):
        hourly = data.get('hourly', {})
        if not hourly.get('time'): return None
        out = {}
        for var in ALL_VARS:
            vals = hourly.get(var)
            if vals is None: out[var] = [0.0]*len(STEPS); continue
            out[var] = [float(vals[s]) if s < len(vals) and vals[s] is not None else 0.0
                        for s in STEPS]
        return out

    log.info('%s: fetching %d pts (model=%s)...', label, n_src, model_name)
    point_data = [None]*n_src

    try:
        import aiohttp, asyncio

        async def fetch_all_async():
            conn = aiohttp.TCPConnector(limit=max_workers, limit_per_host=max_workers,
                                        ttl_dns_cache=300)
            tout = aiohttp.ClientTimeout(total=30, connect=10)
            async with aiohttp.ClientSession(connector=conn, timeout=tout) as sess:
                async def fetch_one(i):
                    p = dict(PARAMS_T)
                    p['latitude']  = round(float(src_la[i]), 4)
                    p['longitude'] = round(float(src_lo[i]), 4)
                    for attempt in range(5):
                        try:
                            async with sess.get(BASE_URL, params=p) as r:
                                if r.status == 200:
                                    return i, parse_resp(await r.json())
                                elif r.status == 429:
                                    await asyncio.sleep(2**attempt)  # exp backoff
                                elif r.status >= 500:
                                    await asyncio.sleep(1+attempt)
                        except Exception:
                            await asyncio.sleep(0.5*(attempt+1))
                    return i, None

                done_count = 0
                for coro in asyncio.as_completed([fetch_one(i) for i in range(n_src)]):
                    idx, result = await coro
                    done_count += 1
                    if done_count % 300 == 0:
                        log.info('%s: %d/%d pts fetched', label, done_count, n_src)
                    if result: point_data[idx] = result

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio; nest_asyncio.apply()
            loop.run_until_complete(fetch_all_async())
        except RuntimeError:
            asyncio.run(fetch_all_async())

    except ImportError:
        log.warning('%s: aiohttp not installed — using threads (slower)', label)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            def sync_fetch(i):
                p = dict(PARAMS_T)
                p['latitude']  = round(float(src_la[i]), 4)
                p['longitude'] = round(float(src_lo[i]), 4)
                try:
                    r = requests.get(BASE_URL, params=p, timeout=20)
                    if r.ok: return i, parse_resp(r.json())
                except Exception: pass
                return i, None
            for idx, result in ex.map(sync_fetch, range(n_src)):
                if result: point_data[idx] = result

    ok_count = sum(1 for p in point_data if p is not None)
    log.info('%s: %d/%d pts OK', label, ok_count, n_src)
    if ok_count < n_src * 0.05:
        log.warning('%s: <5%% data returned — aborting', label)
        return None

    # KDTree interpolation to output grid
    flat_la = np.array(ref_lats); flat_lo = np.array(ref_lons); n = len(flat_la)
    tree = cKDTree(np.column_stack([src_la, src_lo]))
    dists, idxs = tree.query(np.column_stack([flat_la, flat_lo]), k=1)
    far = dists > (sample_step * 1.5)

    def get_var(var, si, default=0.0):
        src_vals = np.array([
            (point_data[pi][var][si] if point_data[pi] and var in point_data[pi]
             and si < len(point_data[pi][var]) else default)
            for pi in range(n_src)], dtype=np.float32)
        out = src_vals[idxs]; out[far] = default
        return out

    def col(arr, default=0.0):
        return [round(float(x), 3) for x in
                np.nan_to_num(np.asarray(arr, np.float32),
                              nan=default, posinf=default, neginf=default)]

    data_out = []
    for si, step in enumerate(STEPS):
        cape  = np.maximum(get_var('cape', si), 0)
        t925  = get_var('temperature_925hPa', si)
        t850  = get_var('temperature_850hPa', si)
        t700  = get_var('temperature_700hPa', si)
        t500  = get_var('temperature_500hPa', si)
        r925  = get_var('relative_humidity_925hPa', si, 75.0)
        r850  = get_var('relative_humidity_850hPa', si, 60.0)
        r700  = get_var('relative_humidity_700hPa', si, 55.0)
        u925  = get_var('wind_u_component_925hPa', si)
        v925  = get_var('wind_v_component_925hPa', si)
        u850  = get_var('wind_u_component_850hPa', si)
        v850  = get_var('wind_v_component_850hPa', si)
        u700  = get_var('wind_u_component_700hPa', si)
        v700  = get_var('wind_v_component_700hPa', si)
        u250  = get_var('wind_u_component_250hPa', si)
        v250  = get_var('wind_v_component_250hPa', si)
        if not np.any(t925 != 0): t925 = t850 + 5.0
        if not np.any(t700 != 0): t700 = (t850 + t500) / 2.0
        sh6 = np.hypot(u250-u850, v250-v850); np.nan_to_num(sh6, copy=False)
        sh1 = np.hypot(u850-u925, v850-v925); np.nan_to_num(sh1, copy=False)
        sh3 = np.hypot(u700-u925, v700-v925); np.nan_to_num(sh3, copy=False)
        cin = np.zeros(n, np.float32)
        td925 = rh_to_td(t925, r925); td850 = rh_to_td(t850, r850)
        lcl_h = compute_lcl_height(t925, td925)
        srh1, srh3 = compute_srh_proxy(u925, v925, u850, v850, u700, v700)
        ehi   = compute_ehi(cape, srh1)
        stp   = compute_stp(cape, sh6, srh1, lcl_h, cin)
        kx    = compute_k_index(t850, t700, t500, r850, r700)
        np.nan_to_num(kx, copy=False)
        tt    = t850 + td850 - 2.*t500; np.nan_to_num(tt, copy=False)
        li    = t500 - (t925 - 34.3);   np.nan_to_num(li, copy=False)
        brn   = compute_brn(cape, sh6)
        dcape = compute_dcape(t700, t500, r850)
        data_out.append({
            'cape': col(cape), 'cin': col(cin), 'kx': col(kx),
            'li': col(li), 'tt': col(tt),
            't925': col(t925), 't850': col(t850), 't700': col(t700), 't500': col(t500),
            'td925': col(td925), 'td850': col(td850),
            'r925': col(r925, 75.0), 'r850': col(r850, 60.0), 'r700': col(r700, 55.0),
            'sh6': col(sh6), 'sh1': col(sh1), 'sh3': col(sh3),
            'srh1': col(srh1), 'srh3': col(srh3),
            'stp': col(stp), 'ehi': col(ehi), 'brn': col(brn), 'dcape': col(dcape),
        })
    log.info('%s: built %d steps', label, len(data_out))
    return data_out


def fetch_cmc(ref_lats, ref_lons):
    log.info('CMC GEM: fetching via Open-Meteo...')
    return _fetch_openmeteo_grid('gem_seamless', ref_lats, ref_lons,
        sample_step=1.0, max_workers=60, forecast_hours=72, label='CMC')


def fetch_arpege_openmeteo(ref_lats, ref_lons):
    log.info('ARPEGE: fetching via Open-Meteo...')
    return _fetch_openmeteo_grid('meteofrance_arpege_world', ref_lats, ref_lons,
        sample_step=1.0, max_workers=60, forecast_hours=72, label='ARPEGE')


def fetch_arome(ref_lats, ref_lons):
    log.info('AROME: fetching via Open-Meteo...')
    return _fetch_openmeteo_grid('meteofrance_arome_france', ref_lats, ref_lons,
        sample_step=1.0, max_workers=60, forecast_hours=42, label='AROME',
        lat_min=41.0, lat_max=55.0, lon_min=-6.0, lon_max=10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# GFS DETERMINISTIC  (NOAA S3, byte-range requests via .idx sidecar files)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_gfs_det(tmpdir, ref_lats, ref_lons):
    from scipy.spatial import cKDTree
    import cfgrib

    base_s3 = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'
    now_utc = datetime.now(timezone.utc)
    gfs_run = None
    for delta_h in range(0, 25, 6):
        dt  = now_utc - timedelta(hours=delta_h)
        gh  = (dt.hour // 6) * 6
        cand = dt.replace(hour=gh, minute=0, second=0, microsecond=0)
        ds  = cand.strftime('%Y%m%d'); rs = '%02d' % cand.hour
        probe = f'{base_s3}/gfs.{ds}/{rs}/atmos/gfs.t{rs}z.pgrb2.0p25.f000'
        try:
            if requests.head(probe, timeout=8).status_code == 200:
                gfs_run = (ds, rs); log.info('GFS run: %s/%sz', ds, rs); break
        except Exception: pass
    if not gfs_run:
        log.warning('GFS: no available run found'); return None

    date_s, run_s = gfs_run
    WANT = [
        ('CAPE:surface','cape',False),('CIN:surface','cin',False),
        ('TMP:925 mb','t925',True),('TMP:850 mb','t850',True),
        ('TMP:700 mb','t700',True),('TMP:500 mb','t500',True),
        ('RH:925 mb','r925',False),('RH:850 mb','r850',False),('RH:700 mb','r700',False),
        ('UGRD:925 mb','u925',False),('UGRD:850 mb','u850',False),
        ('UGRD:700 mb','u700',False),('UGRD:500 mb','u500',False),('UGRD:250 mb','u250',False),
        ('VGRD:925 mb','v925',False),('VGRD:850 mb','v850',False),
        ('VGRD:700 mb','v700',False),('VGRD:500 mb','v500',False),('VGRD:250 mb','v250',False),
        ('PWAT:entire atmosphere','pwat',False),
    ]

    def parse_idx(txt, want_list, fsize):
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        ranges = {}
        for i, line in enumerate(lines):
            parts = line.split(':')
            if len(parts) < 6: continue
            _, off, _, param, level, *_ = parts
            key = f'{param}:{level}'
            for wk, ok, k2c in want_list:
                if wk == key:
                    s = int(off)
                    e = int(lines[i+1].split(':')[1])-1 if i+1 < len(lines) else fsize-1
                    ranges[ok] = (s, e, k2c); break
        return ranges

    flat_la = np.array(ref_lats); flat_lo = np.array(ref_lons); n = len(flat_la)
    tree = idxs = far = None
    raw = {k: {} for _, k, _ in WANT}

    for step in STEPS:
        fname = f'gfs.t{run_s}z.pgrb2.0p25.f{step:03d}'
        url   = f'{base_s3}/gfs.{date_s}/{run_s}/atmos/{fname}'
        try:
            ir = requests.get(url+'.idx', timeout=15)
            if not ir.ok: continue
            hr = requests.head(url, timeout=8)
            fsize = int(hr.headers.get('Content-Length', 999999999))
            ranges = parse_idx(ir.text, WANT, fsize)
            if not ranges: continue
            sorted_r = sorted(set((s,e) for s,e,_ in ranges.values()))
            merged = []
            for s,e in sorted_r:
                if merged and s <= merged[-1][1]+50000: merged[-1]=(merged[-1][0],max(merged[-1][1],e))
                else: merged.append([s,e])
            chunks = {}
            for s,e in merged:
                try:
                    rr = requests.get(url, headers={'Range':f'bytes={s}-{e}'}, timeout=30)
                    if rr.status_code in (200,206): chunks[(s,e)] = (rr.content,s)
                except Exception: pass
            if not chunks: continue

            step_fields = {}
            ctd = Path(tmpdir)/f'gfs_{step:03d}'
            ctd.mkdir(exist_ok=True)
            for ci, ((cs,ce),(data,_)) in enumerate(chunks.items()):
                tf = ctd/f'c{ci}.grib2'
                try:
                    tf.write_bytes(data)
                    with suppress_eccodes_stderr():
                        dsets = cfgrib.open_datasets(str(tf), indexpath=None)
                    for ds in dsets:
                        lats = ds.latitude.values
                        lons = np.where(ds.longitude.values>180, ds.longitude.values-360, ds.longitude.values)
                        if lats.ndim==1:
                            LG,OG = np.meshgrid(lats,lons,indexing='ij')
                            fl_la=LG.ravel(); fl_lo=OG.ravel()
                        else:
                            fl_la=lats.ravel(); fl_lo=lons.ravel()
                        if tree is None and fl_la.size>100:
                            mask=((fl_la>=LAT_MIN-2)&(fl_la<=LAT_MAX+2)&
                                  (fl_lo>=LON_MIN-2)&(fl_lo<=LON_MAX+2))
                            m_la=fl_la[mask]; m_lo=fl_lo[mask]
                            if m_la.size==0: continue
                            _t=cKDTree(np.column_stack([m_la,m_lo]))
                            dists,_ir=_t.query(np.column_stack([flat_la,flat_lo]),k=1)
                            idxs=np.where(mask)[0][_ir]; far=dists>1.0
                        if idxs is None: continue
                        for var in ds.data_vars:
                            arr=ds[var].values
                            lc=None
                            for c in ('isobaricInhPa','pressure','level'):
                                if c in ds.coords: lc=float(np.atleast_1d(ds.coords[c].values)[0]); break
                            if arr.ndim>2: arr=arr.squeeze()
                            if arr.ndim!=2: continue
                            fv=arr.ravel().astype(np.float32)
                            if fv.size!=fl_la.size: continue
                            lv=int(round(lc)) if lc else 0
                            km={('cape',0):'cape',('cin',0):'cin',('pwat',0):'pwat',
                                ('t',925):'t925',('t',850):'t850',('t',700):'t700',('t',500):'t500',
                                ('r',925):'r925',('r',850):'r850',('r',700):'r700',
                                ('u',925):'u925',('u',850):'u850',('u',700):'u700',('u',500):'u500',('u',250):'u250',
                                ('v',925):'v925',('v',850):'v850',('v',700):'v700',('v',500):'v500',('v',250):'v250'}
                            ok=km.get((var.lower(),lv))
                            if ok:
                                if ok.startswith('t') and ok[1:].isdigit() and fv.mean()>100:
                                    fv=fv-273.15
                                step_fields[ok]=fv
                except Exception as pe:
                    log.debug('GFS step %d chunk %d: %s', step, ci, pe)
                finally:
                    try: tf.unlink()
                    except: pass
            try: ctd.rmdir()
            except: pass

            if step_fields and idxs is not None:
                for key,vals in step_fields.items():
                    if vals.size>idxs.max():
                        v=vals[idxs].copy(); v[far]=0.0; raw[key][step]=v
        except Exception as e:
            log.debug('GFS step %d: %s', step, e)

    if idxs is None or not any(raw.get('cape')):
        log.warning('GFS: no usable data'); return None

    def get_g(key, step, default=0.0):
        d=raw.get(key,{})
        if not d: return np.full(n, default, np.float32)
        s=step if step in d else min(d, key=lambda x: abs(x-step))
        return d[s].copy() if abs(s-step)<=9 else np.full(n, default, np.float32)

    def col(arr, default=0.0):
        return [round(float(x),3) for x in
                np.nan_to_num(np.asarray(arr,np.float32),nan=default,posinf=default,neginf=default)]

    data_out = []
    for step in STEPS:
        cape=np.maximum(get_g('cape',step),0); cin=get_g('cin',step)
        cin=np.where(cin>0,-cin,cin)
        t925=get_g('t925',step); t850=get_g('t850',step)
        t700=get_g('t700',step); t500=get_g('t500',step)
        r925=get_g('r925',step,75.0); r850=get_g('r850',step,60.0); r700=get_g('r700',step,55.0)
        u925=get_g('u925',step); v925=get_g('v925',step)
        u850=get_g('u850',step); v850=get_g('v850',step)
        u700=get_g('u700',step); v700=get_g('v700',step)
        u250=get_g('u250',step); v250=get_g('v250',step)
        if not np.any(t925!=0): t925=t850+5.0
        if not np.any(t700!=0): t700=(t850+t500)/2.0
        sh6=np.hypot(u250-u850,v250-v850); np.nan_to_num(sh6,copy=False)
        sh1=np.hypot(u850-u925,v850-v925); np.nan_to_num(sh1,copy=False)
        sh3=np.hypot(u700-u925,v700-v925); np.nan_to_num(sh3,copy=False)
        td925=rh_to_td(t925,r925); td850=rh_to_td(t850,r850)
        lcl_h=compute_lcl_height(t925,td925)
        srh1,srh3=compute_srh_proxy(u925,v925,u850,v850,u700,v700)
        ehi=compute_ehi(cape,srh1); stp=compute_stp(cape,sh6,srh1,lcl_h,cin)
        kx=compute_k_index(t850,t700,t500,r850,r700); np.nan_to_num(kx,copy=False)
        tt=t850+td850-2.*t500; np.nan_to_num(tt,copy=False)
        li=t500-(t925-34.3); np.nan_to_num(li,copy=False)
        brn=compute_brn(cape,sh6); dcape=compute_dcape(t700,t500,r850)
        data_out.append({
            'cape':col(cape),'cin':col(cin),'kx':col(kx),'li':col(li),'tt':col(tt),
            't925':col(t925),'t850':col(t850),'t700':col(t700),'t500':col(t500),
            'td925':col(td925),'td850':col(td850),
            'r925':col(r925,75.0),'r850':col(r850,60.0),'r700':col(r700,55.0),
            'sh6':col(sh6),'sh1':col(sh1),'sh3':col(sh3),
            'srh1':col(srh1),'srh3':col(srh3),
            'stp':col(stp),'ehi':col(ehi),'brn':col(brn),'dcape':col(dcape),
            'pwat':col(get_g('pwat',step,20.0)),
        })

    log.info('GFS data built: %d steps, run %s/%sz', len(data_out), date_s, run_s)
    return data_out, gfs_run

def fetch_icon_sfc(tmpdir, date_s, run_s):
    """
    Fetch DWD ICON-EU on the regular-lat-lon grid from opendata.dwd.de.
    ICON-EU runs every 3h. URL format:
      https://opendata.dwd.de/weather/nwp/icon-eu/grib/{RUN}/{param}/
        icon-eu_europe_regular-lat-lon_single-level_{YYYYMMDDHH}_{STEP:03d}_{PARAM}.grib2.bz2
    """
    import bz2 as bz2mod

    base = 'https://opendata.dwd.de/weather/nwp/icon-eu/grib'
    results = {}
    dt_tag = date_s + run_s  # e.g. '2026032112'

    def fetch_steps(PNAME, pkey, level_type, level=None):
        step_files = []
        for step in STEPS:
            if level_type == 'single-level':
                fname = ('icon-eu_europe_regular-lat-lon_single-level_'
                         '%s_%03d_%s.grib2.bz2' % (dt_tag, step, PNAME))
                subdir = pkey.lower()
            else:
                fname = ('icon-eu_europe_regular-lat-lon_pressure-level_'
                         '%s_%03d_%d_%s.grib2.bz2' % (dt_tag, step, level, PNAME))
                subdir = pkey.lower()
            url = '%s/%s/%s/%s' % (base, run_s, subdir, fname)
            try:
                r = requests.get(url, timeout=30)
                if not r.ok:
                    if step == 0:
                        log.warning('ICON-EU %s(lev=%s) step 0: HTTP %d — %s',
                                    PNAME, level, r.status_code, url)
                    continue
                raw_bytes = bz2mod.decompress(r.content)
                lev_tag = '_%d' % level if level else ''
                out = Path(tmpdir) / ('icon_%s%s_%03d.grib2' % (pkey, lev_tag, step))
                out.write_bytes(raw_bytes)
                step_files.append((step, out))
            except Exception as e:
                if step == 0:
                    log.warning('ICON-EU %s(lev=%s) step 0: %s', PNAME, level, e)
        if step_files:
            tag = pkey if level is None else '%s_%d' % (pkey, level)
            results[tag] = step_files
            log.info('ICON-EU %s lev=%s: %d/%d steps', PNAME, level, len(step_files), len(STEPS))
        else:
            log.warning('ICON-EU %s lev=%s: no steps fetched', PNAME, level)

    # Single-level
    fetch_steps('CAPE_ML', 'cape_ml', 'single-level')
    fetch_steps('CIN_ML',  'cin_ml',  'single-level')
    fetch_steps('TD_2M',   'td_2m',   'single-level')
    fetch_steps('T_2M',    't_2m',    'single-level')

    # Pressure levels — add 925 hPa for SRH
    for lev in [925, 850, 700, 500]:
        fetch_steps('T', 't', 'pressure-level', level=lev)
    for lev in [925, 850, 700]:
        fetch_steps('RELHUM', 'relhum', 'pressure-level', level=lev)
    for lev in [925, 850, 700, 500, 250]:
        fetch_steps('U', 'u', 'pressure-level', level=lev)
        fetch_steps('V', 'v', 'pressure-level', level=lev)

    return results


def build_icon_grid(icon_files, ref_lats, ref_lons, run):
    """
    Read ICON-EU GRIB2 files (regular lat/lon grid) and interpolate
    onto the same output grid as ECMWF.
    """
    from scipy.spatial import cKDTree
    import cfgrib

    flat_la = np.array(ref_lats)
    flat_lo = np.array(ref_lons)
    n = len(flat_la)

    def read_icon_file(fpath):
        """Read one ICON-EU GRIB2 file, return (flat_lats, flat_lons, flat_vals)."""
        try:
            with suppress_eccodes_stderr():
                dsets = cfgrib.open_datasets(str(fpath))
            for ds in dsets:
                if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
                    continue
                if float(ds.longitude.max()) > 180:
                    ds = ds.assign_coords(
                        longitude=(ds.longitude + 180) % 360 - 180
                    ).sortby('longitude')
                # Subset to Europe — try descending lat first, then ascending
                ds_eu = ds.sel(latitude=slice(LAT_MAX+2, LAT_MIN-2),
                                longitude=slice(LON_MIN-2, LON_MAX+2))
                if ds_eu.latitude.size == 0:
                    ds_eu = ds.sel(latitude=slice(LAT_MIN-2, LAT_MAX+2),
                                    longitude=slice(LON_MIN-2, LON_MAX+2))
                la_1d = ds_eu.latitude.values
                lo_1d = ds_eu.longitude.values
                LG, OG = np.meshgrid(la_1d, lo_1d, indexing='ij')
                fl_la = LG.ravel()
                fl_lo = OG.ravel()
                for var in ds_eu.data_vars:
                    arr = ds_eu[var].values
                    if arr.ndim > 2:
                        arr = arr.squeeze()
                    fl_v = arr.ravel()
                    if fl_v.size == fl_la.size:
                        return fl_la, fl_lo, fl_v
        except Exception as e:
            log.debug('ICON read_icon_file %s: %s', fpath.name, e)
        return None, None, None

    # Load all files
    raw = {}   # {pkey_tag: {step: (la, lo, vals)}}
    for tag, step_files in icon_files.items():
        raw[tag] = {}
        for step, fpath in step_files:
            la, lo, va = read_icon_file(fpath)
            if la is not None:
                raw[tag][step] = (la, lo, va)
            else:
                log.warning('ICON read %s step %d: failed', tag, step)

    if not any(raw.values()):
        return None

    # Build KDTree from first loaded field
    src_la = src_lo = None
    for tag, steps in raw.items():
        if steps:
            src_la, src_lo, _ = next(iter(steps.values()))
            break
    if src_la is None:
        return None

    log.info('ICON-EU grid: %d source pts, params: %s', len(src_la), list(raw.keys()))
    tree = cKDTree(np.column_stack([src_la, src_lo]))
    dists, idxs = tree.query(np.column_stack([flat_la, flat_lo]), k=1)
    far = dists > 1.0

    def get_ic(tag, step, default=0.0, k_to_c=False):
        d = raw.get(tag, {})
        if not d:
            return np.full(n, default)
        s = step if step in d else min(d, key=lambda x: abs(x - step))
        if abs(s - step) > 6:
            return np.full(n, default)
        _, _, va = d[s]
        out = va[idxs].copy()
        out[far] = default
        if k_to_c:
            out = out - 273.15
        return out

    def compute_kindex(t850, t700, t500, td2m, rh850):
        """K-Index = (T850-T500) + Td850 - (T700-Td700).
        We approximate Td850 from RH850 + T850, and Td700 from T700 assuming
        similar humidity to 850."""
        # Magnus formula: Td = 243.5*ln(RH/100 * exp(17.67*T/(243.5+T))) / (17.67-same)
        rh = np.clip(rh850, 1, 100)
        g  = (17.67 * t850) / (243.5 + t850) + np.log(rh / 100.0)
        td850 = (243.5 * g) / (17.67 - g)
        # For 700 hPa, use surface dew point as proxy (conservative)
        td700 = td2m - 5.0   # rough: Td drops ~5°C from surface to 700 hPa in moist air
        return (t850 - t500) + td850 - (t700 - td700)

    date_s, run_s = run
    data_out = []
    for step in STEPS:
        cape  = get_ic('cape_ml', step)
        cin   = get_ic('cin_ml',  step)
        cin   = np.where(cin > 0, -cin, cin)
        t2m   = get_ic('t_2m',  step, k_to_c=True)
        td2m  = get_ic('td_2m', step, k_to_c=True)

        # Pressure levels — use 0.0 default, detect missing with np.any
        t925  = get_ic('t_925',      step, default=0.0, k_to_c=True)
        t850  = get_ic('t_850',      step, default=0.0, k_to_c=True)
        t700  = get_ic('t_700',      step, default=0.0, k_to_c=True)
        t500  = get_ic('t_500',      step, default=0.0, k_to_c=True)
        rh925 = get_ic('relhum_925', step, default=75.0)
        rh850 = get_ic('relhum_850', step, default=60.0)
        rh700 = get_ic('relhum_700', step, default=55.0)
        u925  = get_ic('u_925',  step)
        v925  = get_ic('v_925',  step)
        u850  = get_ic('u_850',  step)
        v850  = get_ic('v_850',  step)
        u700  = get_ic('u_700',  step)
        v700  = get_ic('v_700',  step)
        u500  = get_ic('u_500',  step)
        v500  = get_ic('v_500',  step)
        u250  = get_ic('u_250',  step)
        v250  = get_ic('v_250',  step)

        # Fill missing levels from T2m estimates
        if not np.any(t925 != 0): t925 = t2m - 4.0
        if not np.any(t850 != 0): t850 = t2m - 9.75
        if not np.any(t500 != 0): t500 = t2m - 35.75
        if not np.any(t700 != 0): t700 = (t850 + t500) / 2.0

        sh6 = np.hypot(u250 - u850, v250 - v850)
        sh1 = np.hypot(u850 - u925, v850 - v925)
        np.nan_to_num(sh6, copy=False); np.nan_to_num(sh1, copy=False)

        kx = compute_kindex(t850, t700, t500, td2m, rh850)
        np.nan_to_num(kx, copy=False)

        # thundeR params
        td925 = rh_to_td(t925, rh925)
        td850 = rh_to_td(t850, rh850)
        lcl_h = compute_lcl_height(t925, td925)
        srh_01, srh_03 = compute_srh_proxy(u925, v925, u850, v850, u700, v700)
        ehi   = compute_ehi(cape, srh_01)
        stp   = compute_stp(cape, sh6, srh_01, lcl_h, cin)
        brn   = compute_brn(cape, sh6)
        dcape = compute_dcape(t700, t500, rh850)
        tt    = t850 + td850 - 2.0 * t500
        li    = t500 - (t925 - 34.3)
        np.nan_to_num(tt, copy=False); np.nan_to_num(li, copy=False)

        def col(arr, default=0.0):
            a = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=default, posinf=default, neginf=default)
            return [round(float(x), 3) for x in a]

        data_out.append({
            'cape':  col(cape),
            'cin':   col(cin),
            'kx':    col(kx),
            'li':    col(li),
            'tt':    col(tt),
            't850':  col(t850),
            't500':  col(t500),
            'r850':  col(rh850, 60.0),
            'sh6':   col(sh6),
            'sh1':   col(sh1),
            'lcl':   col(lcl_h),
            'srh1':  col(srh_01),
            'srh3':  col(srh_03),
            'ehi':   col(ehi),
            'stp':   col(stp),
            'brn':   col(brn),
            'dcape': col(dcape),
        })

    return data_out


def fetch_gefs(tmpdir, ref_lats, ref_lons, run):
    """
    Fetch NOAA GEFSv12 ensemble mean (geavg) from AWS S3.
    Downloads pgrb2a + pgrb2b per step, scores once per step.
    Fast: 25 steps × 2 files = 50 downloads at ~15MB each.
    """
    import cfgrib
    from scipy.spatial import cKDTree

    date_s, run_s = run
    flat_la = np.array(ref_lats)
    flat_lo = np.array(ref_lons)
    n = len(flat_la)

    base = ('https://noaa-gefs-pds.s3.amazonaws.com/gefs.%s/%s/atmos/pgrb2ap5'
            % (date_s, run_s))

    tree = idxs = far = None

    def parse_grib(fpath):
        fields = {}
        try:
            with suppress_eccodes_stderr():
                dsets = cfgrib.open_datasets(str(fpath), indexpath=None)
            for ds in dsets:
                lats = ds.latitude.values
                lons = ds.longitude.values
                lons = np.where(lons > 180, lons - 360, lons)
                if lats.ndim == 1:
                    LG, OG = np.meshgrid(lats, lons, indexing='ij')
                    fl_la = LG.ravel(); fl_lo = OG.ravel()
                else:
                    fl_la = lats.ravel(); fl_lo = lons.ravel()
                for var in ds.data_vars:
                    arr = ds[var].values
                    lev_coord = None
                    for c in ('isobaricInhPa', 'pressure', 'level'):
                        if c in ds.coords:
                            lev_coord = ds.coords[c].values; break
                    if arr.ndim == 3 and lev_coord is not None:
                        for li, lv in enumerate(np.atleast_1d(lev_coord)):
                            lv_int = int(round(float(lv)))
                            if lv_int not in (925, 850, 700, 500, 250): continue
                            vals = arr[li].ravel().astype(np.float32)
                            if var == 't': vals = vals - 273.15
                            fields['%s%d' % (var, lv_int)] = (fl_la, fl_lo, vals)
                    elif arr.ndim == 2:
                        vals = arr.ravel().astype(np.float32)
                        if var == 't': vals = vals - 273.15
                        if lev_coord is not None:
                            lv_int = int(round(float(np.atleast_1d(lev_coord)[0])))
                            if lv_int in (925, 850, 700, 500, 250):
                                fields['%s%d' % (var, lv_int)] = (fl_la, fl_lo, vals)
                        elif var in ('cape', 'cin', 'pwat'):
                            fields[var] = (fl_la, fl_lo, vals)
        except Exception as e:
            log.debug('parse_grib %s: %s', fpath.name, e)
        return fields

    raw = {}   # {key: {step: flat_vals}}
    fetched = 0

    for step in STEPS:
        step_fields = {}
        for product in ('pgrb2a', 'pgrb2b'):
            fname = 'geavg.t%sz.%s.0p50.f%03d' % (run_s, product, step)
            fpath = Path(tmpdir) / fname
            url   = '%s/%s' % (base, fname)
            try:
                r = requests.get(url, timeout=90, stream=True)
                if not r.ok:
                    if step == 0:
                        log.warning('GEFS step 0 %s: HTTP %d', product, r.status_code)
                    continue
                with open(fpath, 'wb') as f:
                    for chunk in r.iter_content(131072):
                        f.write(chunk)
                step_fields.update(parse_grib(fpath))
                fpath.unlink(missing_ok=True)
            except Exception as e:
                if step == 0:
                    log.warning('GEFS step 0 %s: %s', product, e)

        if not step_fields:
            if step == 0:
                log.warning('GEFS step 0: no fields parsed')
            continue

        if tree is None:
            fl_la0, fl_lo0, _ = next(iter(step_fields.values()))
            lat_m = (fl_la0 >= LAT_MIN-3) & (fl_la0 <= LAT_MAX+3)
            lon_m = (fl_lo0 >= LON_MIN-3) & (fl_lo0 <= LON_MAX+3)
            mask  = lat_m & lon_m
            if mask.sum() == 0: continue
            sub_la = fl_la0[mask]; sub_lo = fl_lo0[mask]
            tree = cKDTree(np.column_stack([sub_la, sub_lo]))
            dists, idxs_raw = tree.query(np.column_stack([flat_la, flat_lo]), k=1)
            far  = dists > 1.0
            idxs = np.where(mask)[0][idxs_raw]
            log.info('GEFS step 0 fields: %s', sorted(step_fields.keys()))

        for key, (fl_la, fl_lo, vals) in step_fields.items():
            raw.setdefault(key, {})[step] = vals
        fetched += 1

    if not raw or tree is None or fetched == 0:
        log.warning('GEFS: no data fetched')
        return None

    log.info('GEFS: %d/%d steps, fields: %s', fetched, len(STEPS), sorted(raw.keys()))

    def get_g(key, step, default=0.0):
        d = raw.get(key, {})
        if not d: return np.full(n, default, dtype=np.float32)
        s = step if step in d else min(d, key=lambda x: abs(x - step))
        if abs(s - step) > 9: return np.full(n, default, dtype=np.float32)
        out = d[s][idxs].copy().astype(np.float32)
        out[far] = default
        return out

    def col(arr, default=0.0):
        a = np.nan_to_num(np.asarray(arr, dtype=np.float32),
                          nan=default, posinf=default, neginf=default)
        return [round(float(x), 3) for x in a]

    data_out = []
    for step in STEPS:
        cape = np.maximum(get_g('cape', step), 0)
        cin  = get_g('cin',  step); cin = np.where(cin > 0, -cin, cin)
        t925 = get_g('t925', step); t850 = get_g('t850', step)
        t700 = get_g('t700', step); t500 = get_g('t500', step)
        r850 = get_g('r850', step, 60.0); r925 = get_g('r925', step, 75.0)
        r700 = get_g('r700', step, 55.0)
        u925 = get_g('u925', step); v925 = get_g('v925', step)
        u850 = get_g('u850', step); v850 = get_g('v850', step)
        u700 = get_g('u700', step); v700 = get_g('v700', step)
        u250 = get_g('u250', step); v250 = get_g('v250', step)
        if not np.any(t925 != 0): t925 = t850 + 5.0
        if not np.any(t700 != 0): t700 = (t850 + t500) / 2.0
        sh6 = np.hypot(u250-u850, v250-v850); np.nan_to_num(sh6, copy=False)
        sh1 = np.hypot(u850-u925, v850-v925); np.nan_to_num(sh1, copy=False)
        sh3 = np.hypot(u700-u925, v700-v925); np.nan_to_num(sh3, copy=False)
        td925 = rh_to_td(t925, r925); td850 = rh_to_td(t850, r850)
        lcl_h = compute_lcl_height(t925, td925)
        srh1, srh3 = compute_srh_proxy(u925, v925, u850, v850, u700, v700)
        ehi  = compute_ehi(cape, srh1)
        stp  = compute_stp(cape, sh6, srh1, lcl_h, cin)
        kx   = compute_k_index(t850, t700, t500, r850, r700)
        np.nan_to_num(kx, copy=False)
        tt   = t850 + td850 - 2.*t500; np.nan_to_num(tt, copy=False)
        li   = t500 - (t925 - 34.3);   np.nan_to_num(li, copy=False)
        data_out.append({
            'cape': col(cape), 'cin': col(cin), 'kx': col(kx),
            'li': col(li), 'tt': col(tt),
            't925': col(t925), 't850': col(t850), 't700': col(t700), 't500': col(t500),
            'td925': col(td925), 'td850': col(td850),
            'r925': col(r925, 75.0), 'r850': col(r850, 60.0), 'r700': col(r700, 55.0),
            'sh6': col(sh6), 'sh1': col(sh1), 'sh3': col(sh3),
            'srh1': col(srh1), 'srh3': col(srh3),
            'stp': col(stp), 'ehi': col(ehi),
            'lcl': col(lcl_h), 'pwat': col(get_g('pwat', step, 20.0)),
        })

    return data_out


# ═══════════════════════════════════════════════════════════════════════════════
# GEFS ENSEMBLE CLASS
# Downloads all 31 members in parallel (HTTP threads), scores each member
# independently in main thread, returns ensemble exceedance probabilities.
#
# Core idea (Taszarek et al. 2020):
#   P(hazard level k) = fraction of members where score >= threshold[k]
#   Reliability = 1 - spread/mean  (high spread = low certainty)
#   Final output score = P(level_1) * mean_score / (1 + beta*sigma)
#
# This is proper probabilistic convective forecasting — identical to how
# ECMWF's EPS-based severe weather outlook works.
# ═══════════════════════════════════════════════════════════════════════════════

class GEFSEnsemble:
    """
    Fetches all 31 GEFSv12 members and computes ensemble hazard probabilities.

    Architecture:
    - Phase 1 (16 threads): HTTP-only downloads, return raw bytes to main thread
    - Phase 2 (main thread): cfgrib parse + regrid + score (thread-safe)
    - Output: per-step dict with score arrays derived from exceedance probabilities
    """

    MEMBERS = ['gec00'] + ['gep%02d' % i for i in range(1, 31)]

    # Risk thresholds per hazard (matches JS HAZARDS.thresh)
    THRESHOLDS = {
        'tstm':    [0.50, 1.00, 1.80, 2.80, 4.00, 5.50],
        'hail':    [0.25, 0.70, 1.40, 2.50, 3.80],
        'wind':    [0.25, 0.70, 1.40, 2.50, 3.80],
        'tornado': [0.18, 0.50, 1.00, 1.80, 2.80],
    }

    def __init__(self, tmpdir, ref_lats, ref_lons, run):
        self.tmpdir  = Path(tmpdir)
        self.flat_la = np.array(ref_lats)
        self.flat_lo = np.array(ref_lons)
        self.n       = len(self.flat_la)
        self.date_s, self.run_s = run
        self.base    = ('https://noaa-gefs-pds.s3.amazonaws.com/gefs.%s/%s/atmos/pgrb2ap5'
                        % (self.date_s, self.run_s))
        self.tree = self.idxs = self.far = None

    # ── Phase 1: parallel HTTP downloads ────────────────────────────────────

    def _download(self, member, step, product):
        """Download one file. Returns (member, step, product, fpath|None)."""
        fname = '%s.t%sz.%s.0p50.f%03d' % (member, self.run_s, product, step)
        fpath = self.tmpdir / fname
        if fpath.exists():
            return member, step, product, fpath
        try:
            r = requests.get('%s/%s' % (self.base, fname),
                             timeout=120, stream=True)
            if not r.ok:
                return member, step, product, None
            with open(fpath, 'wb') as f:
                for chunk in r.iter_content(131072):
                    f.write(chunk)
            return member, step, product, fpath
        except Exception:
            return member, step, product, None

    def download_all(self):
        """Download all member × step × product files in parallel.
        Returns dict {(member, step, product): fpath}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        tasks = [(m, s, p)
                 for m in self.MEMBERS
                 for s in STEPS
                 for p in ('pgrb2a', 'pgrb2b')]
        results = {}
        with ThreadPoolExecutor(max_workers=48) as ex:
            futs = {ex.submit(self._download, m, s, p): (m, s, p)
                    for m, s, p in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    prog(60 + int(done / len(tasks) * 15),
                         'GEFS: %d/%d files' % (done, len(tasks)))
                try:
                    m, s, p, fpath = fut.result()
                    if fpath:
                        results[(m, s, p)] = fpath
                except Exception:
                    pass
        log.info('GEFS: %d/%d files downloaded', len(results), len(tasks))
        return results

    # ── Phase 2: parse + regrid + score (main thread) ───────────────────────

    def _parse(self, fpath):
        """Parse one GRIB2 file → {varkey: flat_array}. Main thread only."""
        out = {}
        try:
            with suppress_eccodes_stderr():
                import cfgrib as _cf
                dsets = _cf.open_datasets(str(fpath), indexpath=None)
            for ds in dsets:
                lats = ds.latitude.values
                lons = np.where(ds.longitude.values > 180,
                                ds.longitude.values - 360, ds.longitude.values)
                if lats.ndim == 1:
                    LG, OG = np.meshgrid(lats, lons, indexing='ij')
                    fl_la, fl_lo = LG.ravel(), OG.ravel()
                else:
                    fl_la, fl_lo = lats.ravel(), lons.ravel()
                for var in ds.data_vars:
                    arr = ds[var].values
                    lev = None
                    for c in ('isobaricInhPa', 'pressure', 'level'):
                        if c in ds.coords:
                            lev = ds.coords[c].values; break
                    if arr.ndim == 3 and lev is not None:
                        for li, lv in enumerate(np.atleast_1d(lev)):
                            lv_int = int(round(float(lv)))
                            if lv_int not in (925, 850, 700, 500, 250): continue
                            v = arr[li].ravel().astype(np.float32)
                            if var == 't': v -= 273.15
                            out['%s%d' % (var, lv_int)] = (fl_la, fl_lo, v)
                    elif arr.ndim == 2:
                        v = arr.ravel().astype(np.float32)
                        if var == 't': v -= 273.15
                        if lev is not None:
                            lv_int = int(round(float(np.atleast_1d(lev)[0])))
                            if lv_int in (925, 850, 700, 500, 250):
                                out['%s%d' % (var, lv_int)] = (fl_la, fl_lo, v)
                        elif var in ('cape', 'cin', 'pwat'):
                            out[var] = (fl_la, fl_lo, v)
        except Exception as e:
            log.debug('parse %s: %s', fpath.name, e)
        return out

    def _build_tree(self, fl_la, fl_lo):
        from scipy.spatial import cKDTree
        lat_m = (fl_la >= LAT_MIN-3) & (fl_la <= LAT_MAX+3)
        lon_m = (fl_lo >= LON_MIN-3) & (fl_lo <= LON_MAX+3)
        mask  = lat_m & lon_m
        if not mask.any(): return False
        sub_la = fl_la[mask]; sub_lo = fl_lo[mask]
        t = cKDTree(np.column_stack([sub_la, sub_lo]))
        dists, ir = t.query(np.column_stack([self.flat_la, self.flat_lo]), k=1)
        self.far  = dists > 1.0
        self.idxs = np.where(mask)[0][ir]
        self.tree = t
        return True

    def _regrid(self, fl_la, fl_lo, vals, default=0.0):
        if self.tree is None: return np.full(self.n, default, np.float32)
        out = vals[self.idxs].copy().astype(np.float32)
        out[self.far] = default
        return out

    def _score_member(self, raw_fields):
        """Regrid + score one member's fields. Returns {hazard: score_array}."""
        n = self.n
        def g(key, default=0.0):
            v = raw_fields.get(key)
            if v is None: return np.full(n, default, np.float32)
            fl_la, fl_lo, vals = v
            return self._regrid(fl_la, fl_lo, vals, default)

        cape = np.maximum(g('cape'), 0)
        cin  = g('cin'); cin = np.where(cin > 0, -cin, cin)
        t925 = g('t925'); t850 = g('t850')
        t700 = g('t700'); t500 = g('t500')
        r850 = g('r850', 60.); r925 = g('r925', 75.); r700 = g('r700', 55.)
        u925 = g('u925'); v925 = g('v925')
        u850 = g('u850'); v850 = g('v850')
        u700 = g('u700'); v700 = g('v700')
        u250 = g('u250'); v250 = g('v250')
        if not np.any(t925 != 0): t925 = t850 + 5.
        if not np.any(t700 != 0): t700 = (t850 + t500) / 2.
        sh6 = np.hypot(u250-u850, v250-v850); np.nan_to_num(sh6, copy=False)
        td925 = rh_to_td(t925, r925)
        lcl_h = compute_lcl_height(t925, td925)
        srh1, srh3 = compute_srh_proxy(u925, v925, u850, v850, u700, v700)
        ehi = compute_ehi(cape, srh1)
        stp = compute_stp(cape, sh6, srh1, lcl_h, cin)
        kx  = compute_k_index(t850, t700, t500, r850, r700)
        np.nan_to_num(kx, copy=False)
        tt = t850 + rh_to_td(t850, r850) - 2.*t500; np.nan_to_num(tt, copy=False)
        li = t500 - (t925 - 34.3); np.nan_to_num(li, copy=False)
        scores = {}
        for hz, fn in SCORE_FNS.items():
            s = fn(cape, cin, kx, t850, t500, r850, sh6, srh1, stp, ehi,
                   tt=tt, srh3=srh3, li=li)
            np.nan_to_num(s, copy=False)
            scores[hz] = s
        return scores

    def _ensemble_score(self, member_scores):
        """
        Convert list of per-member score arrays to a single ensemble score.

        For each hazard and each grid point:
          mu    = mean score across members
          sigma = std (spread)
          p[k]  = fraction of members exceeding threshold[k]   (exceedance probability)

        Final ensemble score = weighted combination:
          score_ens = mu * reliability + p_exceedance_bonus
        where reliability = 1 / (1 + sigma / (mu + eps))
        This rewards consensus (low spread) and penalises disagreement (high spread).

        The score is in the same space as the individual member scores so the
        existing rendering thresholds apply unchanged.
        """
        out = {}
        for hz in SCORE_FNS:
            arrs = [s[hz] for s in member_scores if hz in s]
            if not arrs:
                out[hz] = np.zeros(self.n, np.float32)
                continue
            stacked = np.stack(arrs, axis=0)            # (N, n_pts)
            mu    = stacked.mean(axis=0)
            sigma = stacked.std(axis=0)
            eps   = 1e-6

            # Reliability factor: penalises high spread relative to mean
            reliability = 1.0 / (1.0 + sigma / (mu + eps))

            # Exceedance bonus: fraction of members exceeding each threshold,
            # weighted by level index so higher levels contribute more
            thresholds = self.THRESHOLDS.get(hz, [0.5])
            p_bonus = np.zeros(self.n, np.float32)
            for level_idx, thresh in enumerate(thresholds):
                p_exc = (stacked >= thresh).mean(axis=0)   # [0,1]
                weight = (level_idx + 1) / len(thresholds)
                p_bonus += p_exc * weight * thresh * 0.3   # scale back to score space

            ens_score = (mu * reliability + p_bonus).astype(np.float32)
            np.nan_to_num(ens_score, copy=False)
            out[hz] = ens_score
            out[hz + '_spread'] = sigma.astype(np.float32)
            out[hz + '_p_exc']  = (stacked >= thresholds[0]).mean(axis=0).astype(np.float32)
        return out

    def run(self):
        """Full pipeline: download → parallel parse → score → ensemble stats."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

        downloaded = self.download_all()
        if not downloaded:
            log.warning('GEFS: no files downloaded'); return None

        # ── Phase 2a: parallel cfgrib parsing via process pool ───────────────
        # Each worker process has its own cfgrib C library → fully thread/process safe.
        # n_workers = min(CPU count, 8) — don't exhaust RAM with too many workers.
        n_workers = min(multiprocessing.cpu_count(), 8)
        log.info('GEFS: parsing %d files with %d worker processes',
                 len(downloaded), n_workers)

        # Build list of file paths to parse (each file is one parse task)
        parse_tasks = {}  # (member, step, product) -> fpath_str
        for key, fpath in downloaded.items():
            if fpath and fpath.exists():
                parse_tasks[key] = str(fpath)

        # parsed_fields: {(member, step, product): fields_dict}
        parsed = {}
        prog(75, 'GEFS: parsing %d files...' % len(parse_tasks))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_map = {ex.submit(_gefs_parse_worker, fp): key
                       for key, fp in parse_tasks.items()}
            done = 0
            for fut in as_completed(fut_map):
                key = fut_map[fut]
                done += 1
                if done % 100 == 0 or done == len(parse_tasks):
                    prog(75 + int(done / len(parse_tasks) * 15),
                         'GEFS: parsed %d/%d files' % (done, len(parse_tasks)))
                try:
                    fields = fut.result()
                    if fields:
                        parsed[key] = fields
                except Exception as e:
                    log.debug('GEFS parse worker %s: %s', key, e)

        # Delete downloaded files now that they're parsed
        for fpath in downloaded.values():
            if fpath and fpath.exists():
                fpath.unlink(missing_ok=True)

        log.info('GEFS: %d/%d files parsed successfully', len(parsed), len(parse_tasks))

        # ── Phase 2b: build KDTree (once, from first parsed field) ───────────
        for key, fields in parsed.items():
            if fields:
                fl_la, fl_lo, _ = next(iter(fields.values()))
                if self._build_tree(fl_la, fl_lo):
                    log.info('GEFS grid built from %s', key)
                    break
        if self.tree is None:
            log.warning('GEFS: could not build KDTree'); return None

        # ── Phase 2c + 3: score each member×step, ensemble immediately ──────────
        # Process one step at a time: score all members for that step, compute
        # the ensemble, then discard member arrays before moving to the next step.
        # Peak memory = 1 step × 31 members × 4 hazards × 721k pts × 4 bytes ≈ 360 MB.
        import base64 as _b64

        def col(arr, default=0.0):
            a = np.nan_to_num(np.asarray(arr, np.float32),
                              nan=default, posinf=default, neginf=default)
            return _b64.b64encode(a.astype(np.float16).tobytes()).decode('ascii')

        members_ok = set()
        data_out   = []

        for step in STEPS:
            step_member_scores = []
            for member in self.MEMBERS:
                raw = {}
                for product in ('pgrb2a', 'pgrb2b'):
                    raw.update(parsed.pop((member, step, product), {}))
                if not raw:
                    continue
                try:
                    scores = self._score_member(raw)
                    step_member_scores.append(scores)
                    members_ok.add(member)
                except Exception as e:
                    log.debug('GEFS score %s step %d: %s', member, step, e)
                finally:
                    del raw

            # Ensemble and serialise this step, then free all member arrays
            ens = (self._ensemble_score(step_member_scores) if step_member_scores
                   else {hz: np.zeros(self.n, np.float32) for hz in SCORE_FNS})
            data_out.append({k: col(v) for k, v in ens.items()})
            del step_member_scores, ens

        if not members_ok:
            log.warning('GEFS: no members scored'); return None
        log.info('GEFS: %d/%d members scored', len(members_ok), len(self.MEMBERS))
        return data_out


def load_thread():
    global _data, _run, _loading, _raw_params
    tmpdir = tempfile.mkdtemp(prefix='tstm_')
    try:
        _loading = True

        # Build output grid — ECMWF no longer fetched.
        # We construct a regular lat/lon grid at GRID_STEP resolution.
        lats_g = np.arange(LAT_MIN, LAT_MAX + GRID_STEP, GRID_STEP)
        lons_g = np.arange(LON_MIN, LON_MAX + GRID_STEP, GRID_STEP)
        LG, OG = np.meshgrid(lats_g, lons_g, indexing='ij')
        flat_la = LG.ravel()
        flat_lo = OG.ravel()
        n = len(flat_la)
        log.info('Output grid: %d pts at %.4f° resolution', n, GRID_STEP)

        # Detect latest available ICON-EU run.
        # ICON-EU runs every 3h (00,03,06,09,12,15,18,21z).
        # Data is published ~3-4 hours after run time.
        # We probe by checking a single CAPE_ML file for step 0.
        now_utc = datetime.now(timezone.utc)
        run = None
        for delta_h in range(0, 48, 3):
            dt = now_utc - timedelta(hours=delta_h)
            # Round down to nearest 3h
            rh = (dt.hour // 3) * 3
            cand = dt.replace(hour=rh, minute=0, second=0, microsecond=0)
            date_s_try = cand.strftime('%Y%m%d')
            run_s_try  = '%02d' % cand.hour
            # Probe with CAPE_ML step 0
            probe_url = (
                'https://opendata.dwd.de/weather/nwp/icon-eu/grib/'
                '%s/cape_ml/icon-eu_europe_regular-lat-lon_single-level_'
                '%s%s_000_CAPE_ML.grib2.bz2'
                % (run_s_try, date_s_try, run_s_try)
            )
            try:
                r = requests.head(probe_url, timeout=8)
                if r.status_code == 200:
                    run = (date_s_try, run_s_try)
                    log.info('Found ICON-EU run %s/%sz (probe OK)', date_s_try, run_s_try)
                    break
                else:
                    log.debug('Run %s/%sz not available (HTTP %d)', date_s_try, run_s_try, r.status_code)
            except Exception as e:
                log.debug('Probe failed %s/%sz: %s', date_s_try, run_s_try, e)

        if not run:
            fb = (now_utc - timedelta(hours=24)).replace(hour=0, minute=0, second=0, microsecond=0)
            run = (fb.strftime('%Y%m%d'), '00')
            log.warning('Could not detect available run — falling back to %s/00z', run[0])
        _run = run
        log.info('Using run %s/%sz', run[0], run[1])

        # ── ICON-EU ──────────────────────────────────────────────────────────
        prog(10, 'Fetching DWD ICON-EU (%s/%sz)...' % (run[0], run[1]))
        icon_data = None
        try:
            icon_files = fetch_icon_sfc(tmpdir, run[0], run[1])
            if icon_files:
                icon_data = build_icon_grid(
                    icon_files, flat_la.tolist(), flat_lo.tolist(), run)
                log.info('ICON data built: %d steps', len(icon_data) if icon_data else 0)
        except Exception as e:
            log.warning('ICON fetch failed: %s', e)

        # ── ECMWF IFS ────────────────────────────────────────────────────────
        prog(55, 'Fetching ECMWF IFS...')
        ecmwf_data = None
        try:
            ecmwf_files, ecmwf_run = fetch_all(tmpdir)
            if ecmwf_files:
                ecmwf_result = build_grid(ecmwf_files, ecmwf_run)
                if ecmwf_result and isinstance(ecmwf_result, dict):
                    ecmwf_data = ecmwf_result.get('data')
                elif isinstance(ecmwf_result, list):
                    ecmwf_data = ecmwf_result
                log.info('ECMWF data built: %d steps', len(ecmwf_data) if ecmwf_data else 0)
            else:
                log.warning('ECMWF: no files returned')
        except Exception as e:
            log.warning('ECMWF fetch failed (non-fatal): %s', e)

        # ── GFS deterministic ─────────────────────────────────────────────────
        prog(60, 'Fetching NOAA GFS...')
        gfs_det_data = None
        try:
            result = fetch_gfs_det(tmpdir, flat_la.tolist(), flat_lo.tolist())
            if result:
                gfs_det_data, gfs_det_run = result
                log.info('GFS data built: %d steps', len(gfs_det_data))
        except Exception as e:
            log.warning('GFS fetch failed (non-fatal): %s', e)

        # ── Open-Meteo models (CMC, ARPEGE, AROME) in parallel ───────────────
        prog(63, 'Fetching Open-Meteo models in parallel (CMC/ARPEGE/AROME)...')
        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
        _OM = {'cmc': fetch_cmc, 'arpege': fetch_arpege_openmeteo, 'arome': fetch_arome}
        _ref_la = flat_la.tolist(); _ref_lo = flat_lo.tolist()
        _om = {k: None for k in _OM}
        import time as _time
        with _TPE(max_workers=len(_OM)) as _pool:
            # Stagger submissions by 3s so they don't all hit Open-Meteo simultaneously
            _fut = {}
            for _i, (name, fn) in enumerate(_OM.items()):
                if _i > 0: _time.sleep(3)
                _fut[_pool.submit(fn, _ref_la, _ref_lo)] = name
            for f in _ac(_fut):
                name = _fut[f]
                try:
                    _om[name] = f.result()
                    log.info('%s data built: %d steps', name,
                             len(_om[name]) if _om[name] else 0)
                except Exception as e:
                    log.warning('%s fetch failed (non-fatal): %s', name, e)
        cmc_data    = _om['cmc']
        arpege_data = _om['arpege']
        arome_data  = _om['arome']

        gefs_data = None  # GEFS removed

        if not any([icon_data, ecmwf_data, gfs_det_data, cmc_data, arpege_data, arome_data]):
            prog(0, 'No model data available — all sources failed')
            return

        # Build times list from ICON or GEFS step count
        n_steps = len(next(d for d in [icon_data,ecmwf_data,gfs_det_data,cmc_data,arpege_data,arome_data] if d))
        base_dt = datetime(int(run[0][:4]), int(run[0][4:6]), int(run[0][6:]),
                           int(run[1]), tzinfo=timezone.utc)
        times_out = [(base_dt + timedelta(hours=s)).strftime('%Y-%m-%dT%H:%M')
                     for s in STEPS[:n_steps]]

        models_raw = {}
        if icon_data:
            models_raw['icon']   = icon_data
        if ecmwf_data:
            models_raw['ecmwf']  = ecmwf_data
        if gfs_det_data:
            models_raw['gfs']    = gfs_det_data
        if cmc_data:
            models_raw['cmc']    = cmc_data
        if arpege_data:
            models_raw['arpege'] = arpege_data
        if arome_data:
            models_raw['arome']  = arome_data

        # Score all available models
        prog(75, 'Computing hazard scores...')
        models_scores = {}
        import base64
        for mname, mdata in models_raw.items():
            log.info('Scoring %s (%d steps)...', mname, len(mdata))

            # GEFS returns pre-scored data from per-member ensemble averaging.
            # Detect this: step dicts have hazard keys but no 'cape' key.
            is_prescored = (mdata and 'cape' not in mdata[0]
                            and any(hz in mdata[0] for hz in SCORE_FNS))

            if is_prescored:
                # Extract pre-computed scores directly — already float arrays
                scores = {}
                for hazard in SCORE_FNS:
                    all_steps = []
                    for step_data in mdata:
                        arr = np.asarray(step_data.get(hazard, [0.0]),
                                         dtype=np.float32)
                        np.nan_to_num(arr, copy=False)
                        all_steps.append(
                            base64.b64encode(
                                arr.astype(np.float16).tobytes()
                            ).decode('ascii'))
                    scores[hazard] = all_steps
                models_scores[mname] = scores
                log.info('  %s pre-scored OK (ensemble mean of per-member scores)', mname)
            else:
                # Standard path: run SCORE_FNS on raw parameter fields.
                # Process in chunks of _SCORE_CHUNK points so that the float64
                # temporaries created by np.where(…, scalar_literal, …) never
                # exceed ~512 KB each — even under memory pressure from
                # OpenBLAS OOM events earlier in the same run.
                _SCORE_CHUNK = 65536
                scores = {}
                for hazard, fn in SCORE_FNS.items():
                    all_steps = []
                    for step_data in mdata:
                        _ref = step_data.get('cape')
                        _n   = len(_ref) if _ref is not None else 1

                        def _gc(k, default=0.0, _sd=step_data,
                                _c0=0, _c1=_n):
                            v = _sd.get(k)
                            sz = _c1 - _c0
                            if v is None:
                                return np.full(sz, np.float32(default),
                                               dtype=np.float32)
                            return np.asarray(v, dtype=np.float32)[_c0:_c1]

                        parts = []
                        for c0 in range(0, _n, _SCORE_CHUNK):
                            c1 = min(c0 + _SCORE_CHUNK, _n)
                            def gc(k, default=0.0, _sd=step_data,
                                   _c0=c0, _c1=c1):
                                v = _sd.get(k)
                                sz = _c1 - _c0
                                if v is None:
                                    return np.full(sz, np.float32(default),
                                                   dtype=np.float32)
                                return np.asarray(v, dtype=np.float32)[_c0:_c1]
                            chunk_s = fn(
                                gc('cape'), gc('cin'), gc('kx'),
                                gc('t850'), gc('t500'),
                                gc('r850', 50.0), gc('sh6'),
                                gc('srh1'), gc('stp'), gc('ehi'),
                                tt=gc('tt'), srh3=gc('srh3'), li=gc('li'))
                            parts.append(
                                np.asarray(chunk_s, dtype=np.float32))
                            del chunk_s   # free float64 temporaries promptly

                        s = (np.concatenate(parts)
                             if parts else np.zeros(_n, dtype=np.float32))
                        del parts
                        np.nan_to_num(s, copy=False)
                        all_steps.append(base64.b64encode(
                            s.astype(np.float16).tobytes()).decode('ascii'))
                        del s
                    scores[hazard] = all_steps
                models_scores[mname] = scores
                log.info('  %s scored OK', mname)

        # Keep raw_params in memory for /api/point but don't include in pickle —
        # with 187 params × 721k pts × 25 steps it's ~5 GB and causes MemoryError.
        grid = {
            'times':   times_out,
            'run':     list(run),
            'lats':    flat_la.tolist(),
            'lons':    flat_lo.tolist(),
            'scores':  models_scores,
            'models':  {k: True for k in models_raw},
            'has_ml':  False,
        }

        with _lock:
            _data = grid
            _raw_params = models_raw

        # Cache only the slim grid (scores + metadata, no raw params)
        try:
            CACHE_FILE.write_bytes(pickle.dumps({'run': run, 'grid': grid}))
            log.info('Cache written OK')
        except Exception as ce:
            log.warning('Cache write failed (non-fatal): %s', ce)

        prog(100, 'Ready — %d steps, %d pts, models: %s' % (
            len(times_out), n, list(models_raw.keys())))

    except Exception as e:
        log.exception('Load error: %s', e)
        prog(0, 'Error: ' + str(e))
    finally:
        _loading = False
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.route('/')
def index_route():
    return send_from_directory('.', 'outlook.html')

@app.route('/api/status')
def api_status():
    return jsonify({'loading': _loading, 'progress': _prog,
                    'has_data': _data is not None, 'run': _run})

def _v(arr, default=0.0):
    """Safely convert to float32 array."""
    if arr is None:
        return np.zeros(1, dtype=np.float32)
    return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=default, posinf=default, neginf=default)

def _score_tstm(cape, cin, kx, t850, t500, r850, sh6, srh1, stp, ehi, tt=None, srh3=None, li=None):
    lrKm = (t850 - t500) / 3.5
    if tt is None: tt = np.zeros_like(cape)
    if srh3 is None: srh3 = np.zeros_like(cape)
    if li is None: li = np.zeros_like(cape)
    pL = np.where(cape>=3000,5.0,np.where(cape>=2000,4.0,np.where(cape>=1200,3.2,
          np.where(cape>=800,2.6,np.where(cape>=500,2.0,np.where(cape>=300,1.5,
          np.where(cape>=150,1.1,np.where(cape>=100,0.8,np.where(cape>=60,0.5,0.1)))))))))
    pL = pL + np.where(kx>=40,1.2,np.where(kx>=35,0.9,np.where(kx>=30,0.6,
          np.where(kx>=25,0.35,np.where(kx>=20,0.20,np.where(kx>=17,0.10,
          np.where(kx>=15,0.05,np.where(kx<=5,-0.6,np.where(kx<=10,-0.25,0.0)))))))))
    pL = pL + np.where(lrKm>=8.5,0.60,np.where(lrKm>=7.5,0.35,np.where(lrKm>=6.5,0.15,np.where(lrKm<=5.0,-0.30,0.0))))
    pL = pL + np.where(t850>=20,0.40,np.where(t850>=14,0.22,np.where(t850>=8,0.10,np.where(t850>=4,0.04,np.where(t850<=0,-0.30,0.0)))))
    pL = pL + np.where(t500<=-30,0.50,np.where(t500<=-24,0.30,np.where(t500<=-18,0.10,0.0)))
    pL = pL + np.where(r850>=85,0.40,np.where(r850>=75,0.25,np.where(r850>=65,0.12,np.where(r850>=55,0.04,np.where(r850<=35,-0.35,0.0)))))
    pL = pL + np.where(cin<-250,-2.2,np.where(cin<-180,-1.1,np.where(cin<-100,-0.40,
          np.where(cin<-75,0.15,np.where(cin<-25,0.25,np.where(cin<-10,0.05,-0.20))))))
    # Total Totals boost (TT > 44 = thunderstorm possible in Europe)
    pL = pL + np.where(tt>=56,0.5,np.where(tt>=50,0.3,np.where(tt>=44,0.15,np.where(tt<35,-0.2,0.0))))
    # LI: negative = unstable
    pL = pL + np.where(li<=-6,0.5,np.where(li<=-4,0.3,np.where(li<=-2,0.15,np.where(li>=2,-0.2,0.0))))
    bCDS = np.sqrt(np.maximum(cape,0)) * sh6 / 100.0
    pS = (np.where(bCDS>=12,3.0,np.where(bCDS>=9,2.2,np.where(bCDS>=6,1.5,np.where(bCDS>=3.5,0.8,np.where(bCDS>=1.5,0.3,0.0)))))
        + np.where(sh6>=20,2.0,np.where(sh6>=18,1.6,np.where(sh6>=12,1.0,np.where(sh6>=7,0.4,np.where(sh6>=5,0.1,0.0)))))
        + np.where((cape>=2500)&(sh6>=12)&(sh6<20),0.5,0.0))
    capeMod = np.where(cape>=3000,0.6,np.where(cape>=2000,0.3,np.where(cape<800,-0.4,0.0)))
    sc = np.where(pL>=1.0, pL*0.35+(pS+capeMod)*0.40, pL*0.35)
    return np.maximum(np.where(cape<50, 0.0, sc), 0.0)

def _score_hail(cape, cin, kx, t850, t500, r850, sh6, srh1, stp, ehi, tt=None, srh3=None, li=None):
    lrKm = (t850 - t500) / 3.5
    if tt is None: tt = np.zeros_like(cape)
    if li is None: li = np.zeros_like(cape)
    pL = _score_tstm(cape, cin, kx, t850, t500, r850, sh6, srh1, stp, ehi, tt=tt, li=li)
    pH = (np.where(cape>=3000,4.0,np.where(cape>=2000,3.0,np.where(cape>=1500,2.2,
          np.where(cape>=1000,1.5,np.where(cape>=500,0.8,np.where(cape>=200,0.3,0.0))))))
        + np.where(lrKm>=8.0,1.6,np.where(lrKm>=7.0,1.0,np.where(lrKm>=6.5,0.4,np.where(lrKm<=5.5,-0.8,0.0))))
        + np.where((cape>=500)&(sh6>=22),2.2,np.where((cape>=500)&(sh6>=20),1.6,
          np.where((cape>=500)&(sh6>=15),1.0,np.where((cape>=500)&(sh6>=8),0.4,0.0))))
        + np.where(t500<=-32,0.6,np.where(t500<=-26,0.3,np.where(t500<=-20,0.1,0.0)))
        + np.where(cin<-200,-1.5,np.where(cin<-120,-0.6,0.0)))
    return np.where(cape<100, 0.0, np.where(pL>=0.8, np.maximum(pL*0.15+pH*0.35,0.0), 0.0))

def _score_wind(cape, cin, kx, t850, t500, r850, sh6, srh1, stp, ehi, tt=None, srh3=None, li=None):
    lrKm = (t850 - t500) / 3.5
    if tt is None: tt = np.zeros_like(cape)
    if li is None: li = np.zeros_like(cape)
    pL = _score_tstm(cape, cin, kx, t850, t500, r850, sh6, srh1, stp, ehi, tt=tt, li=li)
    pW = (np.where(cape>=4000,4.0,np.where(cape>=2500,3.0,np.where(cape>=1500,2.2,
          np.where(cape>=1000,1.5,np.where(cape>=600,1.0,np.where(cape>=300,0.5,0.0))))))
        + np.where((cape>=400)&(sh6>=20),3.5,np.where((cape>=400)&(sh6>=18),2.8,
          np.where((cape>=400)&(sh6>=15),2.0,np.where((cape>=400)&(sh6>=12),1.0,
          np.where((cape>=400)&(sh6>=7),0.3,0.0)))))
        + np.where(lrKm>=8.0,0.5,np.where(lrKm>=7.0,0.25,0.0))
        + np.where(cin<-200,-1.2,np.where(cin<-100,-0.5,0.0)))
    return np.where(cape<80, 0.0, np.where(pL>=0.7, np.maximum(pL*0.12+pW*0.35,0.0), 0.0))

def _score_tornado(cape, cin, kx, t850, t500, r850, sh6, srh1, stp, ehi, tt=None, srh3=None, li=None):
    if srh3 is None: srh3 = srh1 * 1.6  # proxy
    if tt is None: tt = np.zeros_like(cape)
    if li is None: li = np.zeros_like(cape)
    pL = _score_tstm(cape, cin, kx, t850, t500, r850, sh6, srh1, stp, ehi, tt=tt, li=li)
    blMoisture = np.where(r850>=85,1.25,np.where(r850>=75,1.10,np.where(r850>=65,1.0,0.75)))
    llsProxy = sh6 * blMoisture
    pT = (np.where(cape>=3000,3.5,np.where(cape>=2500,3.0,np.where(cape>=2000,2.5,
          np.where(cape>=1500,2.0,np.where(cape>=1000,1.4,np.where(cape>=600,0.8,np.where(cape>=200,0.3,0.0)))))))
        + np.where((cape>=500)&(sh6>=20),3.0,np.where((cape>=500)&(sh6>=18),2.2,
          np.where((cape>=500)&(sh6>=14),1.3,np.where((cape>=500)&(sh6>=10),0.5,0.0))))
        + np.where(llsProxy>=20,2.0,np.where(llsProxy>=16,1.4,np.where(llsProxy>=12,0.8,np.where(llsProxy>=8,0.3,0.0))))
        + np.where(r850>=85,0.5,np.where(r850>=75,0.3,np.where(r850>=65,0.1,np.where(r850<=50,-0.8,0.0))))
        + np.where(t850>=18,0.5,np.where(t850>=12,0.25,np.where(t850<=5,-1.0,0.0)))
        + np.where(cin<-180,-1.0,np.where(cin<-100,-0.4,0.0))
        + np.where(stp>=3,1.0,np.where(stp>=1,0.5,0.0))
        # SRH 0-3km: key tornado discriminator
        + np.where(srh3>=300,1.0,np.where(srh3>=150,0.5,np.where(srh3>=50,0.2,0.0))))
    return np.where(cape<200, 0.0, np.where(pL>=1.2, np.maximum(pL*0.15+pT*0.30,0.0), 0.0))

SCORE_FNS = {
    'tstm':    _score_tstm,
    'hail':    _score_hail,
    'wind':    _score_wind,
    'tornado': _score_tornado,
}


def fast_serialize_grid(data):
    """Stub — no longer used."""
    return None


@app.route('/api/data')
def api_data():
    if _data is None:
        return jsonify({'error': 'Not ready', 'loading': _loading}), 503
    # Send scores + metadata — exclude raw_params (too large, only used by /api/point)
    send = {k: v for k, v in _data.items() if k != 'raw_params'}
    return jsonify(send)


@app.route('/api/point')
def api_point():
    """Return all parameters for a single grid point — called on map click."""
    if _data is None:
        return jsonify({'error': 'Not ready'}), 503
    from flask import request as req
    import math
    try:
        step  = int(req.args.get('step', 0))
        idx   = int(req.args.get('idx', 0))
        model = req.args.get('model', 'icon')
        # Fall back through available models
        mdata = (_raw_params.get(model) or
                 _raw_params.get('icon') or
                 _raw_params.get('gefs') or [])
        if step >= len(mdata):
            return jsonify({})
        sd = mdata[step]
        out = {}
        for k, v in sd.items():
            try:
                val = v[idx] if hasattr(v, '__getitem__') else v
                fv = float(val)
                out[k] = 0.0 if (math.isnan(fv) or math.isinf(fv)) else round(fv, 3)
            except Exception:
                out[k] = 0.0
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reload')
def api_reload():
    global _loading
    if not _loading:
        threading.Thread(target=load_thread, daemon=True).start()
    return jsonify({'started': True})

@app.route('/<path:fn>')
def static_file(fn):
    return send_from_directory('.', fn)


def startup():
    global _data, _run
    if CACHE_FILE.exists():
        age = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age < CACHE_MAX_H:
            try:
                saved = pickle.loads(CACHE_FILE.read_bytes())
                _data = saved['grid']
                _run  = saved['run']
                # raw_params not cached (too large) — /api/point returns zeros
                # until next full fetch. Scores and map rendering work fine.
                log.info('Cache loaded (%.1fh old, run %s) — ready', age, _run)
                return
            except Exception as e:
                log.warning('Cache invalid: %s — re-fetching', e)
    threading.Thread(target=load_thread, daemon=True).start()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # required on Windows for ProcessPoolExecutor
    log.info('EU Thunderstorm Outlook  -  http://localhost:8765')
    startup()
    try:
        from waitress import serve
        log.info('Starting with waitress (production WSGI server)')
        serve(app, host='0.0.0.0', port=8765, threads=8,
              channel_timeout=600,        # 10 min timeout for large data transfers
              recv_bytes=65536,
              send_bytes=65536)
    except ImportError:
        log.warning('waitress not installed — using Flask dev server')
        log.warning('For reliable large responses: pip install waitress')
        app.run(host='0.0.0.0', port=8765, debug=False,
                use_reloader=False, threaded=True)
