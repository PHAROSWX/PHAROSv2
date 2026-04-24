#!/usr/bin/env python3
"""
generate_static.py
==================
Headless pipeline runner for the EU Convective Outlook.

Imports the processing functions from server.py, runs the full
ICON-EU + GEFS data fetch, and writes two static files:

  docs/forecast_data.json   — the full API response (what /api/data returns)
  docs/meta.json            — run metadata for the page header

Usage:
  python generate_static.py [--icon-only] [--gefs-only] [--resolution 0.25]

Environment variables:
  ICON_ONLY=1          Skip GEFS (faster, ~15 min instead of ~90 min)
  GRID_STEP=0.25       Output resolution in degrees (default: 0.25)
  SKIP_CACHE_CHECK=1   Always regenerate even if run hasn't changed
"""

import sys
import os
import json
import gzip
import argparse
import logging
import shutil
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger(__name__)

DOCS_DIR = Path('docs')
META_FILE = DOCS_DIR / 'meta.json'
DATA_FILE = DOCS_DIR / 'forecast_data.json'

# ── Resolution config ─────────────────────────────────────────────────────────
# Default 0.25° gives ~45k grid points over Europe — a good balance of
# detail vs. file size. The original server uses 0.0625° (~720k pts)
# which would produce a ~200 MB JSON unsuitable for GitHub Pages.
DEFAULT_GRID_STEP = float(os.environ.get('GRID_STEP', '0.25'))


def check_icon_eu_available():
    """
    Probe DWD open data for the most recent available ICON-EU run.
    ICON-EU runs every 3h (00,03,06,09,12,15,18,21 UTC).
    Returns (date_str, run_str) or None if nothing found in the past 12h.
    """
    now_utc = datetime.now(timezone.utc)
    for delta_h in range(0, 13, 3):
        dt = now_utc - timedelta(hours=delta_h)
        rh = (dt.hour // 3) * 3
        cand = dt.replace(hour=rh, minute=0, second=0, microsecond=0)
        date_s = cand.strftime('%Y%m%d')
        run_s  = '%02d' % cand.hour
        probe = (
            f'https://opendata.dwd.de/weather/nwp/icon-eu/grib/'
            f'{run_s}/cape_ml/'
            f'icon-eu_europe_regular-lat-lon_single-level_'
            f'{date_s}{run_s}_000_CAPE_ML.grib2.bz2'
        )
        try:
            r = requests.head(probe, timeout=10)
            if r.status_code == 200:
                log.info('ICON-EU available: %s/%sz', date_s, run_s)
                return date_s, run_s
            else:
                log.debug('ICON-EU %s/%sz: HTTP %d', date_s, run_s, r.status_code)
        except Exception as e:
            log.debug('ICON-EU probe %s/%sz: %s', date_s, run_s, e)
    return None


def check_gefs_available():
    """
    Probe NOAA S3 for the most recent available GEFS run.
    GEFS runs at 00,06,12,18 UTC. Returns (date_str, run_str) or None.
    """
    now_utc = datetime.now(timezone.utc)
    for delta_h in range(0, 25, 6):
        dt   = now_utc - timedelta(hours=delta_h)
        gh   = (dt.hour // 6) * 6
        cand = dt.replace(hour=gh, minute=0, second=0, microsecond=0)
        gdate = cand.strftime('%Y%m%d')
        grun  = '%02d' % cand.hour
        probe = (
            f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{gdate}/{grun}/'
            f'atmos/pgrb2ap5/gec00.t{grun}z.pgrb2a.0p50.f000'
        )
        try:
            rp = requests.head(probe, timeout=10)
            if rp.status_code == 200:
                log.info('GEFS available: %s/%sz', gdate, grun)
                return gdate, grun
            else:
                log.debug('GEFS %s/%sz: HTTP %d', gdate, grun, rp.status_code)
        except Exception as e:
            log.debug('GEFS probe %s/%sz: %s', gdate, grun, e)
    return None


def load_last_meta():
    """Load the last written meta.json to compare run IDs."""
    if META_FILE.exists():
        try:
            with open(META_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def write_outputs(data: dict, icon_run, gefs_run):
    """Write forecast_data.json and meta.json to docs/."""
    DOCS_DIR.mkdir(exist_ok=True)

    # Write the full data JSON
    log.info('Writing %s ...', DATA_FILE)
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    size_mb = DATA_FILE.stat().st_size / 1e6
    log.info('  → %.1f MB', size_mb)

    # Write meta
    meta = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'icon_run': list(icon_run) if icon_run else None,
        'gefs_run': list(gefs_run) if gefs_run else None,
        'run_id':   (icon_run[0] + icon_run[1] if icon_run else
                     gefs_run[0] + gefs_run[1] if gefs_run else 'unknown'),
        'n_steps':  len(data.get('times', [])),
        'n_pts':    len(data.get('lats', [])),
        'models':   list(data.get('models', {}).keys()),
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info('Wrote %s', META_FILE)
    return meta


def run_pipeline(icon_only=False, gefs_only=False):
    """
    Main pipeline:
    1. Check which model runs are available
    2. Compare with last processed run — skip if unchanged
    3. Patch server.py constants for our resolution
    4. Call server.load_thread() which sets server._data
    5. Write outputs
    """
    # ── Availability probes ───────────────────────────────────────────────────
    log.info('=== EU Convective Outlook — Static Generator ===')
    log.info('Grid step: %.4f°', DEFAULT_GRID_STEP)

    icon_run = None if gefs_only  else check_icon_eu_available()
    gefs_run = None if icon_only  else check_gefs_available()

    if not icon_run and not gefs_run:
        log.error('Neither ICON-EU nor GEFS data is available right now. Aborting.')
        sys.exit(1)

    if icon_only and not icon_run:
        log.error('--icon-only requested but ICON-EU is not available. Aborting.')
        sys.exit(1)

    # ── Skip if already up to date ────────────────────────────────────────────
    skip_check = os.environ.get('SKIP_CACHE_CHECK', '0') == '1'
    if not skip_check:
        last_meta = load_last_meta()
        new_run_id = ((icon_run[0] + icon_run[1]) if icon_run else
                      (gefs_run[0] + gefs_run[1]) if gefs_run else '')
        if last_meta.get('run_id') == new_run_id:
            log.info('Run %s already processed — nothing to do.', new_run_id)
            log.info('Set SKIP_CACHE_CHECK=1 to force regeneration.')
            sys.exit(0)

    # ── Import and patch server module ────────────────────────────────────────
    log.info('Importing server module...')
    # xarray must be imported before server.py so cfgrib registers open_datasets
    try:
        import xarray as _xr
        import cfgrib as _cfgrib_pre
        # Manually register open_datasets if cfgrib doesn't expose it yet
        if not hasattr(_cfgrib_pre, 'open_datasets'):
            from cfgrib.xarray_store import open_datasets as _ods
            _cfgrib_pre.open_datasets = _ods
            log.info('cfgrib.open_datasets manually registered from xarray_store')
        else:
            log.info('cfgrib.open_datasets already available')
    except Exception as _e:
        log.warning('xarray/cfgrib pre-import: %s', _e)
    try:
        import server
    except ImportError as e:
        log.error('Cannot import server.py: %s', e)
        log.error('Make sure generate_static.py is in the same directory as server.py.')
        sys.exit(1)

    # Patch resolution — must happen before the grid is built
    server.GRID_STEP = DEFAULT_GRID_STEP

    # Suppress Flask startup, silence progress to just logging
    def _prog(pct, msg):
        log.info('[%3d%%] %s', pct, msg)
    server.prog = _prog

    # Patch cfgrib.open_datasets to log errors at WARNING so CI logs show
    # exactly what goes wrong, rather than silently returning empty datasets.
    import cfgrib as _cfgrib_mod
    if not hasattr(_cfgrib_mod, 'open_datasets'):
        from cfgrib.xarray_store import open_datasets as _ods
        _cfgrib_mod.open_datasets = _ods
        log.info('cfgrib.open_datasets injected from cfgrib.xarray_store')

    _orig_open_datasets = _cfgrib_mod.open_datasets

    def _loud_open_datasets(path, **kw):
        try:
            result = _orig_open_datasets(path, **kw)
            if not result:
                log.warning("cfgrib.open_datasets returned 0 datasets for %s", path)
            return result
        except Exception as e:
            log.warning("cfgrib.open_datasets FAILED for %s: %s: %s",
                        path, type(e).__name__, e)
            raise

    _cfgrib_mod.open_datasets = _loud_open_datasets
    log.info("cfgrib patched for verbose error reporting")

    # Quick cfgrib self-test
    try:
        import eccodes
        log.info("eccodes Python package version: %s", eccodes.__version__)
    except Exception as e:
        log.warning("eccodes import issue: %s", e)

    # Optionally disable one model by monkey-patching availability
    if icon_only:
        # We'll still let load_thread detect ICON; just limit GEFS members
        log.info('icon-only mode: GEFS will be skipped')
        _orig_gefs_probe = requests.head

        def _head_no_gefs(url, **kw):
            if 'noaa-gefs-pds' in url:
                class _Fake:
                    status_code = 404
                return _Fake()
            return _orig_gefs_probe(url, **kw)

        requests.head = _head_no_gefs

    # ── Run the pipeline ──────────────────────────────────────────────────────
    log.info('Starting data pipeline...')
    t0 = time.time()

    server._data = None
    server._run  = None
    server._loading = False

    try:
        server.load_thread()
    except Exception as e:
        log.exception('Pipeline failed: %s', e)
        sys.exit(1)

    elapsed = time.time() - t0
    log.info('Pipeline completed in %.1f s (%.1f min)', elapsed, elapsed / 60)

    if server._data is None:
        log.error('No data produced by pipeline — check logs above.')
        sys.exit(1)

    # ── Write outputs ─────────────────────────────────────────────────────────
    # Determine which run was actually used (from server's detection)
    actual_icon_run = None
    actual_gefs_run = None

    if 'icon' in server._data.get('models', {}):
        actual_icon_run = icon_run
    if 'gefs' in server._data.get('models', {}):
        actual_gefs_run = gefs_run

    meta = write_outputs(server._data, actual_icon_run, actual_gefs_run)
    log.info('=== Done ===')
    log.info('  Run:    %s', meta['run_id'])
    log.info('  Steps:  %d', meta['n_steps'])
    log.info('  Points: %d', meta['n_pts'])
    log.info('  Models: %s', meta['models'])
    log.info('  Output: %s', DATA_FILE)


def main():
    global DEFAULT_GRID_STEP  # must be declared before any reference in this scope

    res_help = 'Grid resolution in degrees (default: %.4f)' % DEFAULT_GRID_STEP
    parser = argparse.ArgumentParser(description='EU Convective Outlook - Static Generator')
    parser.add_argument('--icon-only', action='store_true',
                        help='Fetch ICON-EU only (skip GEFS; faster)')
    parser.add_argument('--gefs-only', action='store_true',
                        help='Fetch GEFS only (skip ICON-EU)')
    parser.add_argument('--resolution', type=float, default=None, help=res_help)
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if run ID unchanged')
    args = parser.parse_args()

    if args.resolution:
        DEFAULT_GRID_STEP = args.resolution
        os.environ['GRID_STEP'] = str(args.resolution)

    if args.force:
        os.environ['SKIP_CACHE_CHECK'] = '1'

    run_pipeline(icon_only=args.icon_only, gefs_only=args.gefs_only)


if __name__ == '__main__':
    main()
