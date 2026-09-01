# Detailed Implementation Plan — Phases 1–3

**Commit discipline (applies throughout):** every relocation is a *move-only* commit (`git mv`, imports fixed, zero logic changes), followed by separate *change* commits. No compatibility shims — you're the only consumer; stale import paths die immediately. Each work package ends with the smoke/regression suite green.

---

## Phase 1 — Restore the spine

**Objective:** one importable generation; `pip install -e .` works in a fresh venv; every module imports side-effect-free; `pytest` collects and passes a small honest suite.

### WP 1.1 — Quarantine dead generations *(p1-quarantine)*

| Item | Action | Detail |
|---|---|---|
| `MetTool_Historic.py` | move → `references/` | Provenance for legacy KNB/cascade methods; drops phantom deps (`wx`, `osgeo`, `wdmtoolbox`, `ambhas`) from the package. |
| `pipeline.py` | delete | Imports five nonexistent modules. Ledger/orchestration ideas resurface in P3 (`store/manifest.py`) and P7 (runner). |
| `_zarr.py` | split → `scripts/` | Salvage `get_filepaths`, `netcdf_to_zarr` (note: currently ignores its `netcdf_dir` arg — fixed at P3 rewrite), `audit_netcdf_cache`, and the working chunk-aligned batch-append loop into `scripts/netcdf_to_zarr.py` as inert functions under a `main()` guard. Delete all module-level execution, duplicate blocks, and personal `C:\Users\...` paths. This file is the raw material for `store/ingest.py` in P3 — nothing is lost, it just stops being importable package code. |
| `derivations/precipitation.py` | delete | 0 bytes. |
| `tests/` | delete stale suite | All 8 test modules target the previous generation (`met_timeseries.aggregation`, `.ledger`, `.config`, `prism._fetch_single_day`, `disaggregate_temperature_pattern` — none import). **Before deleting:** harvest `conftest.py`'s `catchment_geojson`/`dissolved_polygons`/`simple_dataset` fixtures (reusable as-is) and skim `test_derivations.py`/`test_disaggregation.py` for golden expected *values* worth carrying into new tests. |

### WP 1.2 — Create foundation homes, dissolve the grab-bags *(p1-foundations)*

**New files:**

- **`geometry.py`** — receives: `BoundingBox` + `contains()` from `sources/base.py`; the **single** `clip_dataset` (keep the `utils.py` version — it's the most general, with `lat_dim`/`lon_dim` params and half-cell padding); `_bounds_to_polygon` from `nldas.py` (promoted to public `bounds_to_polygon`). `CACHE_BOUNDS` lands here *temporarily*, docstring-flagged "becomes a `config.Settings` default in P3."
- **`viz.py`** — receives `plot_bounds` from `sources/base.py`. Only module allowed to import matplotlib.
- **`export/hspf_hdf5.py`** — receives `hdf5WDM` from `utils.py` verbatim (rename/polish deferred to P7; this is a placement move so the HSPF seed stops being buried).
- **`spatial/`** package — `git mv` of `polygons.py` and `weights.py` (move-only; weights split into `weights.py`+`aggregate.py` waits for P4).

**Modifications:**

- `sources/base.py` — shrinks to the `FetchFunction` Protocol only (contract rewrite is P2). Remove matplotlib imports.
- All `BoundingBox`/`CACHE_BOUNDS`/`clip_dataset` importers rewired to `geometry`: `nldas.py`, `prism.py`, `mrms.py`, `narr.py`, `stations/ndawn.py`, `stations/synoptic.py`, `spatial/weights.py`. Delete the private `_clip_dataset` copies in `nldas.py` and `prism.py` (plus prism's `_clip_dataarray`) — behavior of the shared one is identical, so this is safe pre-P2.
- `utils.py` — delete (`mem_gb` and the `psutil`/`h5py` module-level imports vanish with it).
- Trivial hygiene while touching: remove `weights.py`'s duplicated back-to-back docstring and its unused `from met_timeseries import polygons` import; remove the `print(missing_vars)` in `fetch_nldas`; update the stale module docstring in `nldas.py` (describes a removed Giovanni "datarods" workflow) and the stale `sources/__init__.py` docstring (describes the old `(bounds, start, end)` contract).

**Verification tasks (each a grep, must return empty):** references to `met_timeseries.utils`, `met_timeseries.pipeline`, `met_timeseries.polygons` (old path), `sources.base import CACHE_BOUNDS`, `matplotlib` outside `viz.py`.

### WP 1.3 — Packaging, docs, smoke tests *(p1-packaging)*

- **`pyproject.toml`:** add missing real deps (`pvlib`, `pyet`, `scipy`, `zarr`, `dask`, `h5py`, `netCDF4`, `pyproj`, `fsspec`); remove the phantom `met-timeseries = "met_timeseries.cli:main"` entry point (returns honestly in P3); create extras `[viz]` (matplotlib), `[stations]` (SynopticPy), `[wdm]` (reserved), `[cloud]` (s3fs, reserved), `[dev]` (pytest, pytest-cov). **Audit before pruning:** `cfgrib` (kept — `mrms.py` reads GRIB2), `metpy`/`mettoolbox`/`click` (grep usage; `mettoolbox` is imported by `disaggregation.py` and `pet.py` → keep; `click` unused until P3 → defer adding).
- **`environment.yml`:** reduce to conda-binary necessities (GDAL/eccodes/esmpy stack) + `pip: -e .[dev]`, so pyproject is the single dependency truth. Decide whether `xesmf`/`esmpy` are actually used (I found no imports — likely prune).
- **New smoke suite:** `tests/conftest.py` (salvaged fixtures), `tests/unit/test_imports.py` (import every module — this alone would have caught most of the current rot), `tests/unit/test_geometry.py` (clip ascending/descending lat), `tests/unit/spatial/test_polygons.py` and `test_weights_kernel.py` (dissolve behavior, `compute_weights` sums/caching on the fixture grid — reconstructable from the old `test_aggregation.py` expectations).
- **README + `examples/quickstart.py`:** rewrite to only what works: load/dissolve polygons → fetch one day (NLDAS or PRISM, with cache dir) → clip → compute weights. No `PipelineConfig`, ledger, or Parquet claims.

**Phase 1 exit:** fresh venv → editable install → all imports clean → smoke suite green → zero references to nonexistent modules.

---

## Phase 2 — Correctness fixes

**Objective:** the compute path produces *right* data; every fix lands with a regression test. This phase creates one new module (`validation.py`) and substantially modifies four existing ones.

### WP 2.1 — `validation.py` (new) *(p2-validation-module)*

Two functions now (grows in P6):
- `validate_time_axis(ds, freq, start=None, end=None, on_gap="reindex"|"raise")` — asserts datetime64 dtype, monotonic, unique; compares against the expected index for the window; `"reindex"` fills gaps with NaN and logs a count, `"raise"` aborts. Pure, no I/O.
- `assert_grid_identical(ds, reference)` — exact lat/lon shape+value equality (extracted from the checks in `audit_netcdf_cache`; P3 ingest becomes its main consumer).

### WP 2.2 — NLDAS time axis + download consolidation *(p2-nldas-time)*

In `sources/nldas.py`:

- **Delete both fabricated-time blocks** (the `start_ts + timedelta(h)` reconstruction in `download` and `download_bulk`). New rule: time comes from CF decoding only. After concat, call `validate_time_axis(..., freq="1h", start, end, on_gap="reindex")` — a failed granule becomes an explicit NaN gap, never a shifted label. If CF decoding genuinely fails for some granules (the reason the hack exists), that's a per-granule decode error to surface, not silently paper over.
- **Consolidate `download` vs `download_bulk` → one function** (confirmed decision: keep the bulk materialize-to-staging → `open_mfdataset` → clip → load pattern; delete the streaming `_open_and_clip` path and the ThreadPoolExecutor variant). Signature: `download(start_date, end_date, variables, bounds, staging_dir=None, max_connections=...)`.
- **Auth extraction:** `earthaccess.login()` leaves the download body. New module-level `ensure_authenticated(strategy="environment")`; download raises a clear "not authenticated" error rather than prompting. Entry points (P3 CLI, notebooks) call it once.
- Delete the dead commented block in `_read_from_cache`.

### WP 2.3 — Shared cache adapter + source contract *(p2-cache-adapter)*

- **Extract to `sources/base.py`** the logic currently duplicated verbatim in `fetch_nldas` and `fetch_prism`: *"if cached: read; compute missing variables; if any: download-missing → load → close → merge → rewrite; else download-all → write; return"*. One adapter function taking a `downloader` callable, plus **atomic writes** (write temp file, `os.replace`) so a crash never leaves a corrupt day. Both fetchers become thin: URL/product specifics + adapter call.
- **PRISM cache extent (confirmed decision):** clip to regional bounds *before* caching in `prism.py`; docstring updated to match. Existing full-CONUS cache files are handled by the P3 migration script, not here. Also fix the `date` variable str→date rebinding and make the inter-request `time.sleep(2)` a named module constant.
- **Shift-once invariant documented and tested:** cache stores unshifted midnight-stamped time; `_shift_time_coord` applies exactly once on read. Test: read the same cached day twice → identical timestamps.
- **Rewrite `FetchFunction`** to the real converged contract — `fetch(date, cache_dir, bounds=None, variables=None, overwrite_cache=False) -> xr.Dataset` — and annotate `fetch_nldas`, `fetch_prism`, `fetch_mrms` against it. (Full `mrms`/`narr` adapter adoption stays in P6; the contract annotation is free now.)
- **Audit candidates for deletion** while here: `get_nldas_gridcells`, `_generate_nldas_grid`, prism's `generate_grid` — grep for consumers; if only the dead test suite used them, delete (git preserves).

### WP 2.4 — Weights semantics *(p2-weights-semantics)*

In `spatial/weights.py`:

- Fix the never-executed pair: `get_weights` passes `dataset=`/`min_coverage` into `build_weightmap(datarray=...)` (TypeError) and `weighted_mean_timeseries` does `["weights"]` then `['weights']` again (KeyError). Standardize the parameter name (`dataset`) across the module.
- `build_weightmap`: **no silent `continue`** on low coverage — always compute, record per-zone `coverage` as a coordinate on the output, emit a warning below threshold; raise only on zero total weight. Fix the `dtype=datarray[lat_dim].dtype` bug (weights are float64, full stop).
- **Semantics change (the important one):** remove the trailing `weights_ds * valid_mask` (which silently assumed a `time` dim and broke normalization) and stop pre-normalizing per zone. Stored weights = raw fractional cell coverage + recorded per-zone totals. Application-time formula — `Σ(w·x) / Σ(w·validity(x))` per timestep — is implemented now in `weighted_mean_timeseries`/`_weight_dataset` (the full `xr.dot` contraction API replaces these in P4; the *semantics* must be right first).

### WP 2.5 — Disaggregation hygiene + seeds *(p2-disagg-hygiene)*

In `disaggregation.py` (still at top level; moves to `temporal/` in P3):

- **Seed plumbing:** `rng: Generator | int | None` threaded through `disaggregate_precipitation_stochastic` → `_molnar_burlando_disagg` → `_molnar_burlando_split`, replacing the per-call unseeded `default_rng()`. Same for the hybrid wrapper.
- **Delete duplicates:** `_disaggregate_precipitation_hybrid` (underscore copy) and `_weights_normalise_respec` (keeper: `_weights_normalise` with offset).
- **Fix `_weights_diurnal` dimension-agnostically** (per our revised contract): index the 24-element profile through the time coordinate xarray-natively so it broadcasts for 1-D, `(time, zone)`, and gridded inputs alike; `"1H"` → `"1h"` (also in `_weights_static`).
- Replace the unreadable throw-inside-lambda `_WEIGHT_DISPATCH` entries with small named functions that raise proper `ValueError`s.
- **Stopgap dimension validation** with clear messages on entry to `disaggregate()` (full per-method capability declarations arrive in P5).

**Phase 2 tests:** mocked-granule gap → NaN hours + unshifted labels; monotonicity validator unit tests; cache atomicity (temp-file naming, interrupted write leaves no final file); merge-missing-vars round trip; PRISM double-read no-double-shift; weight application with injected NaN cells renormalizes exactly; same-seed cascade reproducibility + daily mass conservation; **conservation property tests** for `sum`/`mean_additive`/`mean_multiplicative` on synthetic series (the permanent backbone); diurnal weights sum to 1/day in all three dimensionalities.

**Phase 2 exit:** a real one-month NLDAS + PRISM fetch over a small bbox (network-marked test) passes time-axis and weight-sum validation end to end.

---

## Phase 3 — Store-centric core

**Objective:** canonical consolidated Zarr per source is the only thing science code reads; all persistence is fsspec-URI-addressed; backfills are resumable; the package skeleton reaches its final shape.

### WP 3.1 — `config.py` and `schema.py` (new) *(p3-config-schema)*

- **`config.py` — `Settings` dataclass:** per-source store URIs, per-source raw-cache URIs, local staging dir, region bounds (**`CACHE_BOUNDS` migrates here from `geometry.py` as the default value**, ending its life as a baked constant), `storage_options`, auth strategy. Construction: explicit kwargs > env vars (`MET_TS_*`) > optional TOML file > defaults. Constructed only at entry points, passed down — no module reads the environment itself.
- **`schema.py` — canonical registry:** table of canonical names (`precip`, `temp_air`, `wind_u`, `swdown`, ...) with units and descriptions; per-source native→canonical mappings (NLDAS `Tair/Rainf/SWdown/...`; PRISM `ppt/tmax/tmin/...`; NDAWN's `COLUMN_UNITS`/`HOURLY_VARIABLE_MAP` dicts migrate here in P6). Confirmed decision applied: **stores keep native names**; mapping is applied at read. ⚠️ *Open decision for you at P3 kickoff:* canonical temperature unit — K (SI-pure, NLDAS-native) vs °C (PRISM/HSPF-ergonomic). Everything downstream keys off this; pick once.

### WP 3.2 — `store/` package (new) *(p3-store)*

- **`store/ingest.py`** — the productionized rewrite of the P1-salvaged script (fixing the ignored-`netcdf_dir` bug by construction): per-source **chunk policy table** (NLDAS `time=2160, lat=-1, lon=-1`; PRISM `time=90`; ~5–100 MB compressed target); batch size = strict multiple of time chunk (your existing pattern, kept); **pre-append validation** using P2's `assert_grid_identical` + expected-steps-per-day + monotonic non-overlapping continuation vs the store's current max time; `mode="w"` initialization vs append; consolidate metadata once at the end.
- **`store/manifest.py`** — the ledger reborn: records keyed `(source, date, variables, ingest_version, status)`, stored *beside the store* via fsspec (format decision: parquet vs JSONL — recommend **JSONL** for append-only simplicity and human-debuggability at your scale). API: `mark_complete`, `is_complete`, `get_incomplete(start, end, variables)`. Both fetch-to-cache and append-to-store update it. Single-writer assumption documented explicitly.
- **`store/reader.py`** — `open_source(name, config, canonical=True)`: opens consolidated Zarr lazily by URI + `storage_options`, applies schema-on-read renames/unit attrs, runs minimal invariant checks. Identical code path for `file://`, `memory://`, `s3://`.
- **Migration (one-off, in `scripts/`):** run `audit_netcdf_cache` against your existing `.cache/nldas` and `.cache/prism/800m`; **re-clip the mixed-extent CONUS PRISM files** to regional bounds (P2 changed write behavior; old files must match or `join="exact"` ingest breaks); then backfill both stores and write the manifest retroactively.

### WP 3.3 — fsspec URI currency *(p3-uri-currency)*

- Signature sweep: every persistence parameter (`cache_dir` in `nldas`/`prism`/`mrms` fetchers and the base adapter, all `store/` paths, weightmap save/load stubs) becomes a URI string + `storage_options`; internals use `fsspec.filesystem`/`fsspec.open` instead of `pathlib`. Download **staging stays genuinely local** (tempdir) by design.
- Atomicity strategy per backend, documented in the adapter: local = temp file + `os.replace`; object stores = single-put to the final key (per-object puts are atomic) — no temp-rename emulation.
- `Path.glob` cache discovery is replaced by manifest queries; no directory-listing-driven logic survives above the staging layer.
- **The interoperability proof:** integration suite runs the full chain — mocked fetch → cache → ingest → manifest → `open_source` → clip → weighted aggregation — twice, once on `file://`, once on `memory://`, asserting identical results. This is the test that makes "cloud-ready" a fact rather than a claim.

### WP 3.4 — CLI + final structural moves *(p3-cli)*

- **`cli.py`** (click; dependency added now, entry point restored in pyproject): `met-timeseries fetch <source> --start --end`, `ingest <source> --start --end`, `audit <source>`, `status` (manifest report). Auth (`ensure_authenticated`) and `Settings` construction happen here and only here. This is what makes cron/containers/AWS Batch natural later.
- **Last skeleton move:** `disaggregation.py` → `temporal/` (move-only; the `core/patterns/cascade/resample` split is P5's job). After this commit the package tree matches the target structure; P4–P7 only fill in files, never relocate them.
- README updated with the new mental model: sources → store → science, one diagram, CLI examples.

**Phase 3 tests:** manifest resume (`get_incomplete` after partial backfill); ingest rejects gapped/overlapping/grid-drifted batches; schema-on-read mapping round-trip; CLI smoke via click's runner; the dual-backend (`file://`+`memory://`) end-to-end above.

**Phase 3 exit:** two months of NLDAS + PRISM ingested locally via CLI; `open_source` returns validated canonical data; killing the process mid-backfill and re-running resumes without duplication; the whole integration suite green on `memory://`.

---

## Decisions to confirm before/at each phase kickoff

| # | Decision | When | Recommendation |
|---|---|---|---|
| 1 | Delete `xesmf`/`esmpy` from environment.yml if unimported | P1 | Delete; restore if regridding ever lands |
| 2 | Delete `get_nldas_gridcells`/`_generate_nldas_grid`/prism `generate_grid` if consumerless | P2 | Delete |
| 3 | Canonical temperature/units convention | P3 start | Flag — needs your call (K vs °C) |
| 4 | Manifest format | P3 | JSONL |
| 5 | NLDAS staging retention (delete raw granules after ingest vs keep) | P3 | Delete after successful append + manifest record |

Rough sizing: P1 is 1–2 focused sessions, mostly mechanical; P2 is the careful one (each fix needs its test, and the NLDAS time-axis work needs a real mocked-granule harness); P3 is the largest new-code phase but builds on salvaged, already-working ingest logic. Want me to start executing Phase 1?