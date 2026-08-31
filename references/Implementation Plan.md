# Phased Implementation Plan

**Sequencing logic.** Phases are ordered so each one makes the next cheaper and safer: you can't fix bugs confidently without an importable package (P1), you shouldn't build the store on top of wrong data (P2), the aggregation and temporal engines should read from the store rather than the NetCDF pile (P3 before P4/P5), and recipes/export only make sense once there's a canonical `(time, zone)` product to label and export (P7 last). **Testing is not a separate phase** — the stale suite is deleted in P1, and every phase's exit criteria include tests for what it touched.

**Critical path for your short-term goal** (every met input, arbitrary timestep): **P1 → P2 → P3 → P4 → P5**. P6 and P7 serve the longer-term ensemble/sensitivity goal; P6 can run in parallel with P4/P5.

| Phase | Theme | Size | Critical path |
|---|---|---|---|
| 1 | Restore the spine | S–M | ✔ |
| 2 | Correctness fixes | M | ✔ |
| 3 | Store-centric core (cloud interop) | L | ✔ |
| 4 | Aggregation engine | M | ✔ |
| 5 | Temporal engine, arbitrary Δt | M–L | ✔ |
| 6 | Stations & schema hardening | M | — |
| 7 | Recipes, provenance, HSPF export | L | — |

---

## Phase 1 — Restore the spine

**Goal:** one importable, honestly-packaged generation of the code. Fresh `pip install -e .` works; importing any module has zero side effects; `pytest` collects and passes a minimal suite; nothing in `src/` lies about what exists.

**Alterations:**

- `MetTool_Historic.py` — move out of `src/` into `references/`. It's provenance for the legacy methods (KNB pan evap, cascade), not shippable code.
- `pipeline.py` — delete. It imports five modules that don't exist and calls functions with stale signatures. Its two good ideas (completion ledger, month-batched orchestration) are deliberately resurrected in Phase 3 in store-centric form. Git history preserves it.
- `_zarr.py` — strip all module-level execution and personal `C:\Users\mfratki\...` paths. Salvage the three real assets — `get_filepaths`, `netcdf_to_zarr`, `audit_netcdf_cache` — as inert functions (they become the seed of Phase 3's ingest module). The scratch experiments move to a scripts/notebooks area or are deleted.
- `sources/base.py` — remove the module-level `matplotlib` imports and `plot_bounds` (relocate to an example or future `viz` module). Keep `BoundingBox`. Mark `CACHE_BOUNDS` as "moves to config in Phase 3." Leave `FetchFunction` for Phase 2, where the source contract is decided.
- `utils.py` — dissolve the grab-bag: `clip_dataset` becomes the *single* shared implementation (new `spatial.py` or similar), and the duplicate `_clip_dataset` copies in `nldas.py` and `prism.py` are mechanically replaced with imports of it (behavior is identical, so this is safe now). `hdf5WDM` moves to a new `export.py` placeholder — it's the embryo of the Phase 7 HSPF layer and shouldn't be buried. `mem_gb` is deleted (drops the undeclared `psutil` dependency).
- `derivations/precipitation.py` — 0 bytes; delete.
- `pyproject.toml` — truth-up: add `pvlib`, `pyet`, `scipy`, `zarr`, `dask`, `h5py`, `netCDF4`, `pyproj`, `fsspec`; structure extras: `[viz]` (matplotlib), `[stations]` (SynopticPy), `[cloud]` (s3fs), `[dev]` (pytest). Remove the phantom `met-timeseries = "met_timeseries.cli:main"` entry point (restored for real in Phase 3). Audit whether `cfgrib`/`metpy`/`click` are actually used and prune.
- `environment.yml` — reduce to conda-necessary binaries (GDAL/eccodes stack) + `pip: -e .[dev]` so pyproject is the single source of dependency truth.
- `tests/` — delete the stale suite wholesale (it tests `met_timeseries.aggregation`, `.ledger`, `.config` and old function names; none of it imports). Before deleting, skim it once for salvageable *fixtures and expected values* — `conftest.py`'s polygon/grid fixtures are reusable. Replace with a smoke suite: import every module, run `load_polygons` and `compute_weights` on the small fixtures.
- `README.md` and `examples/quickstart.py` — rewrite to describe only what works today (polygons → fetch one day → clip). No references to `PipelineConfig`, ledger, or Parquet layout until those exist again.

**Exit criteria:** fresh-venv editable install; `python -c "import met_timeseries"` plus every submodule imports cleanly with no side effects; smoke tests green; zero references to nonexistent modules anywhere in `src/`.

---

## Phase 2 — Correctness fixes on the existing compute path

**Goal:** the data that everything downstream will be built on is right. Each fix lands with a regression test.

**Alterations:**

- `sources/nldas.py`:
  - **Kill the fabricated time axis** (both `download` and `download_bulk`). New rule: time comes from CF decoding only. After concat, *validate* — monotonic, no duplicates, and compared against the expected hourly index for the requested window. Failed granules produce explicit NaN gaps (reindex to the expected index) or raise; they never shift labels. This is the highest-stakes fix in the plan.
  - **Collapse `download` vs `download_bulk` to one implementation.** Recommendation: keep the bulk pattern (materialize granules to a staging dir, `open_mfdataset`, clip, load) — it's more predictable in batch/container contexts and maps directly onto Phase 3 ingest. Delete the other.
  - **Move `earthaccess.login()` out of download functions.** One `ensure_authenticated()` at entry-point level, strictly non-interactive when running unattended (env-var strategy), injectable for tests.
  - **Atomic cache writes**: `_write_to_cache` writes to a temp file and renames, so a crash never leaves a corrupt day in the cache. Same in prism.
  - The merge-on-missing-variables logic (read cache → download missing vars → merge → rewrite) is duplicated verbatim in prism; extract one shared cache-adapter helper.
- `sources/prism.py`:
  - **Resolve the cache-extent drift** (decision, flagged below): the docstring promises clipping to `CACHE_BOUNDS`; the code caches full CONUS. Recommendation: clip to the regional bounds before caching — at 800 m, CONUS-per-day-per-variable is enormous, and your reuse domain is fixed anyway. Consequence: existing cache files have mixed extents; Phase 3's audit/repair step re-clips them (mixed extents would break `join="exact"` ingest).
  - Document the `_shift_time_coord` invariant explicitly: cache stores *unshifted* time; the shift is applied exactly once on read. Add a test that reading a cached file twice doesn't double-shift.
- `weights.py`:
  - Fix the broken pair: `get_weights` passes `dataset=` to a function whose parameter is `datarray` (TypeError), and `weighted_mean_timeseries` subscripts `["weights"]['weights']` (KeyError). Standardize the parameter name across the module.
  - `build_weightmap`: coverage below `min_coverage` must never silently `continue`. Warn + record per-polygon coverage as a variable/coordinate on the returned weights Dataset so callers can filter or fail deliberately.
  - **Change the masking/normalization model** — this is the important design fix: stop baking `valid_mask` (from `isel(time=0)`) into stored weights and stop pre-normalizing against it. Store *raw fractional-coverage weights* only. Renormalization against NaNs happens at application time, per variable per timestep: `mean = Σ(w·x) / Σ(w·validity(x))`. NaN patterns differ between variables and over time; normalizing at build time is wrong by construction. (Full application API lands in Phase 4; the semantics change lands here.)
- `disaggregation.py`:
  - **Seed plumbing**: thread `rng: np.random.Generator | int | None` through `disaggregate_precipitation_stochastic → _molnar_burlando_disagg → _molnar_burlando_split`. Unseeded stays allowed interactively; recipes (Phase 7) will always pass seeds.
  - Delete the dead duplicates: `_disaggregate_precipitation_hybrid` (underscore version) and whichever of `_weights_normalise` / `_weights_normalise_respec` loses.
  - Fix `_weights_diurnal`'s numpy-broadcast bug by indexing the 24-element profile through the time coordinate xarray-natively (works for any dimensionality), and change deprecated `"1H"` to `"1h"`.
  - Add dimension validation (clear error messages) as a stopgap; the real contract is declared in Phase 5.
- `sources/base.py` — rewrite `FetchFunction` to the *actual* contract you've converged on: `fetch(date, cache_dir, bounds=None, variables=None) -> xr.Dataset`, and annotate all fetchers against it.

**Tests added:** mocked-granule gap → NaN hours, never shifted labels; monotonic-time validator; cache write atomicity; no-double-shift for PRISM; weights sum to 1 over valid cells at application time with injected NaNs; same-seed cascade reproducibility + daily-mass conservation; conservation property tests (`sum`, `mean_additive`, `mean_multiplicative`) — these become the permanent backbone of the suite.

**Exit criteria:** a real one-month NLDAS + PRISM fetch over a small bbox passes time-axis and weight-sum validation end-to-end.

---

## Phase 3 — Store-centric core (the cloud-interoperability phase)

**Goal:** one canonical, chunk-aligned, consolidated Zarr store per source is the *only* thing downstream code reads; every storage location is an fsspec URI so local vs S3 is a config value, not an architecture. This inverts today's design where the per-day NetCDF pile is the de-facto datastore.

**New modules** (names indicative): `config.py`, `ingest.py`, `store.py`, `manifest.py`.

**Alterations:**

- **`config.py` (reborn):** small dataclass — store URIs per source, staging/cache root, region bounds (this is where `CACHE_BOUNDS` finally lives, as a *default*, not a constant baked into `sources/base.py`), `storage_options` passthrough, auth strategy. Every field overridable by environment variables so containers configure it without code.
- **Path currency rule:** all persistence APIs take URI strings + `storage_options`, never bare `Path`. Applies to: cache dirs in `nldas.py` / `prism.py` / `mrms.py`, all of the ingest module, and weight persistence. `Path.glob` discovery (`get_filepaths`) is replaced by manifest queries (below). Local staging of downloads may remain genuinely local — staging is transient by definition.
- **`ingest.py`** (productionized `_zarr.py`): fix the `netcdf_dir`-ignored bug; parameterize per-source chunk policy (NLDAS `time=2160, lat=-1, lon=-1`; PRISM `time=90`; target ~5–100 MB compressed chunks); batches strictly multiples of the time chunk (your existing pattern, kept); **validate before every append** — coordinate identity against a reference, expected steps/day, monotonic non-overlapping time vs the store's current max (wire in `audit_netcdf_cache`); consolidate metadata once at the end. Include the one-off **repair/migration** path: audit the existing NetCDF caches, re-clip mixed-extent PRISM files, then backfill.
- **`manifest.py`** (the ledger, resurrected properly): per-store record keyed `(source, date, variables, ingest_version)` stored beside the store (JSON/parquet via fsspec). Supports `get_incomplete(start, end)` → resumable, idempotent backfills — the fundamental cloud batch pattern. Both fetch-to-staging and append-to-store update it.
- **`store.py` (the read API):** `open_source("nldas", config) -> xr.Dataset` — opens consolidated Zarr lazily, identical code for `file://` and `s3://`. This is also where **schema-on-read** lands: a per-source mapping table (native name → canonical name, units attr, and any fixed conversions) so downstream code sees one vocabulary (`precip`, `temp_air`, … in SI, UTC). Recommendation (decision, below): store keeps *native* variable names — a faithful archive you never have to rewrite when the schema evolves; canonicalization is a cheap read-time rename.
- **Minimal CLI** (restores the pyproject entry point honestly): `met-timeseries ingest nldas --start ... --end ... --config ...`. This is what makes cron/AWS Batch/containers natural.
- **Tests:** manifest resume semantics; append validation rejects gapped/overlapping batches; full URI round-trip using fsspec's `memory://` filesystem — this lets you prove cloud-path behavior in unit tests with no S3 account.

**Exit criteria:** two months ingested locally end-to-end via CLI; `open_source` returns validated data; the identical test suite passes against `memory://` URIs — that's your interoperability proof.

---

## Phase 4 — Aggregation engine rework

**Goal:** weights are a persisted, validated tensor; aggregation is a single memory-bounded contraction producing the canonical `(time, zone)` product; the per-metzone fetch/aggregate loop from the old pipeline disappears.

**Alterations, all centered on `weights.py`:**

- **Kernel speedup** in `_compute_weights_cached`: replace the full-grid scan-and-`continue` double loop with `np.searchsorted` windowing to the polygon's bbox rows/cols, so only candidate cells are intersected. (Matters at PRISM 800 m: ~10⁵–10⁶ cells per grid vs a few hundred candidates.) Keep shapely-exact intersection — no need for `exactextract`/rasterio backends yet.
- **API consolidation to a compute / persist / apply trio**, replacing today's five overlapping functions:
  - `build_weightmap(grid, polygons, ids)` → weights Dataset with dims `(zone, lat, lon)`, per-zone coverage, and **signature attrs** (grid hash from the lat/lon arrays, polygon-set hash, source name) for cache validity. Zone centroid lat/lon stored as coordinates (Phase 5 needs them for solar).
  - `save/load_weightmap(uri)` → **Zarr, not NetCDF** (NetCDF can't write to object storage cleanly; drops the `netcdf4` engine dependency from this path). `load` validates signatures against the target dataset's coords exactly and recomputes on mismatch.
  - `aggregate_to_zones(dataset, weights)` → `(time, zone)` Dataset. Implementation is a **tensor contraction, not broadcast-multiply**: `xr.dot(x.fillna(0), w, dims=[lat, lon]) / xr.dot(x.notnull(), w, dims=[lat, lon])`. This removes the `(zone, time, lat, lon)` intermediate (your main "why isn't vectorized faster" culprit) and implements the Phase 2 NaN-renormalization semantics in one expression.
  - Delete `get_weights`, `weighted_mean_timeseries`, `_weight_dataset`, `aggregate_over_polygon` remnants.
- **New thin extraction path** (the pipeline's replacement, a few lines now): `open_source` → clip once to the polygon-set's union bbox → grid-stage derivations (Phase 5 defines which) → `aggregate_to_zones` → one `(time, zone)` dataset for *all* zones at once.
- **Benchmark task (do, record, keep):** one month NLDAS × your metzone set — old loop vs contraction, wall time and peak RSS. This is where you get the concrete answer to your vectorization question, and the numbers go in the README.

**Tests:** equivalence with a brute-force per-polygon loop on a small fixture; NaN-cell renormalization property; signature-mismatch invalidation; memory bound (no 4-D intermediate).

**Exit criteria:** all-zones monthly extraction in one contraction, benchmarked, bit-for-bit equal to the reference loop on fixtures.

---

## Phase 5 — Temporal engine on areal-mean series, at arbitrary timestep

**Goal:** your short-term deliverable — every met input extendable at arbitrary Δt — plus resolution of the disaggregation dimensional confusion and the pvlib per-cell cost, by moving temporal work *after* spatial aggregation.

**Alterations:**

- `disaggregation.py`:
  - **Declare the contract**: disaggregation operates on `(time,)` or `(time, zone)` arrays — validated with clear errors. The gridded stochastic path is removed, with the rationale documented (independent per-cell cascades destroy spatial correlation and over-smooth the areal hyetograph after averaging; per-zone cascades on the areal mean are both correct for HSPF and orders of magnitude cheaper). The `apply_ufunc(vectorize=True)` machinery collapses to a simple per-zone loop over 1-D series — zones number in the dozens, so this is honest and fast.
  - **`target_freq` becomes a first-class argument** of `disaggregate()` and every wrapper; `_infer_freq` is only ever used for the *source* frequency.
  - **Weight generators parameterized by `target_freq`**: `_weights_trapezoidal`/`_weights_solar` are already continuous in time — evaluate at target bin centers (or integrate across bins for coarse Δt). The diurnal profile becomes a continuous function of fractional day (interpolated from the 24-point profile, or a user-supplied callable) normalized per coarse period at any Δt — killing the hardcoded 24-element/`"1h"` assumption in `_weights_diurnal`/`_weights_static`/`disaggregate_pevt`.
  - **Cascades stay dyadic, honestly**: document the constraint; reach arbitrary Δt by cascading to the finest dyadic step ≤ target, then a new **sum-conserving `conservative_resample(da, target_freq)`** helper (which also serves the *upward* direction, e.g. hourly → 3 h model inputs). `_cascade_to_hourly` generalizes to `_cascade_to_freq`.
- `derivations/radiation.py`: **zone-level solar geometry**. `clearsky_radiation_ineichen`, `extra_radiation_pvlib`, `daytime_mask_solar_elevation` get `(time, zone)` counterparts that call pvlib once per zone centroid (coordinates supplied by the Phase 4 weights) — ~30 calls instead of ~3,800 per-cell calls. Grid versions are kept only if a grid-stage consumer remains; otherwise deprecated. `lazy_clearsky_ineichen` (broken as written — `apply_ufunc` on a coord-dependent function) is deleted.
- **Derive-order policy** (new, small): each derivation registers a stage — `"grid"` (must happen pre-aggregation, e.g. wind speed from U/V, since √(u²+v²) of means ≠ mean of speeds) or `"zone"` (post-aggregation, e.g. solar, PET, humidity conversions where you choose areal-mean inputs). The extraction path from Phase 4 consults this. Where the right stage is scientifically debatable, that's explicitly a recipe knob later — you've turned a code accident into a sensitivity axis.
- **Timezone discipline** (documented in one place, enforced at boundaries): UTC internally everywhere; PRISM's 12Z shift already conforms; conversion to local standard time happens only in the Phase 7 export layer.

**Tests:** conservation properties parameterized over `target_freq` ∈ {15 min, 1 h, 3 h, 6 h} for every method; solar/trapezoid weights sum to 1 per coarse period at every Δt; cascade + conservative resample conserves daily totals; zone-centroid vs grid-mean clearsky agreement within tolerance on a small zone.

**Exit criteria:** any canonical variable disaggregated from daily to a user-chosen Δt with conservation verified — your short-term goal, demonstrable.

---

## Phase 6 — Station sources & schema hardening (parallelizable after P3)

**Goal:** every source — gridded and station — emits the canonical schema, so stations can serve as alternative forcings, disaggregation patterns, or validation references in the ensemble.

**Alterations:**

- `stations/ndawn.py`: convert °F/inch/mph/Langley and CST at the adapter boundary; return `(time, station)` xarray Datasets with lat/lon/elevation coordinates and canonical names, via the same Phase 3 mapping machinery. Its `COLUMN_UNITS`/`HOURLY_VARIABLE_MAP` dicts become entries in the shared schema registry.
- `stations/synoptic.py`: same conformance; keep the lazy-import pattern (it's already right) and formalize it via the `[stations]` extra.
- `sources/mrms.py` / `sources/narr.py`: conform to the Phase 2 fetch contract, Phase 2 cache adapter (atomic writes), and Phase 3 ingest/manifest registration, each with its own chunk policy.
- **New `validation.py`:** schema validator (names, units attrs, dims, tz-awareness) + optional physical range checks (warn), invoked at store-read and pre-export.

**Exit criteria:** one validator passes on every source's output; a station series can be dropped into the Phase 5 engine as a `fine_pattern` without special-casing.

---

## Phase 7 — Recipes, provenance, and HSPF export

**Goal:** the end vision — N labeled, reproducible forcing variants delivered into HSPF inputs, including *extending existing model records*.

**Alterations:**

- **New `recipes.py`:** declarative `ForcingRecipe` — sources, per-variable derivation method + stage, disaggregation method + parameters, `target_freq`, seed, zone set, period. Serializable (TOML/JSON) and content-hashed → `recipe_id`. Your existing dispatch registries in `disaggregation.py` are already the lookup mechanism recipes need; this layer just makes the choices data instead of code.
- **Provenance:** every output Dataset carries attrs — package version, recipe hash, source store URIs (and their ingest versions from the manifest), seed, creation timestamp. The recipe file itself is persisted next to outputs. Without this, a 10-variant sensitivity study becomes untraceable within days.
- **`export/` (grown from `hdf5WDM`):** three targets, in priority order to confirm with you: (a) **hsp2 HDF5** — your `hdf5WDM` class already reads the `/TIMESERIES/<name>/table` layout, so the writer mirrors it; (b) **classic WDM** via `wdmtoolbox` as an optional extra; (c) Parquet for analysis. Export is where UTC → local standard time and SI → model units conversions happen — nowhere else.
  - **Extend-existing-record semantics** (your stated short-term use case): read an existing DSN's units/timestep/end-date, resample the recipe output conservatively to that timestep (Phase 5's resampler), validate continuity at the splice point, append.
- **Ensemble runner:** iterate recipes → outputs organized as `outputs/{recipe_id}/...` (URI-based, so local or S3) + a catalog table (`recipe_id` → parameters → output URI) that your sensitivity analysis reads directly.

**Exit criteria:** two recipes differing in one knob (e.g., PET method, or cascade seed) run end-to-end on fixture data, produce provenance-stamped outputs, export into an HDF5/WDM readable by the existing reader, and appear in the catalog.

---

## Decisions to confirm (recommendations embedded)

1. **Delete** `pipeline.py` and the stale tests rather than archive in-tree (git history suffices) — P1.
2. **PRISM cache extent:** clip to regional bounds before caching; migrate existing CONUS-extent files during P3 repair — P2/P3.
3. **NLDAS download path:** keep the materialize-then-open bulk pattern; delete the streaming variant — P2.
4. **Store variable names:** native names in the Zarr archive, canonicalization on read — P3.
5. **Export target priority:** hsp2 HDF5 first vs classic WDM first — depends on which HSPF runtime your models use; defer to P7 kickoff.

## Deliberately out of scope (until a phase proves the need)

Dask distributed / cluster parallelism (your clipped domain fits in RAM; the wins are algorithmic), kerchunk/VirtualiZarr virtual stores, non-regular-grid support in the weights kernel, a plugin/entry-point system for sources, and any GUI. Each is a known escape hatch, not a foundation.

The one habit change that underwrites the whole plan: **experiments live on git branches and either merge or die** — no more `_variant` copies of functions. Every phase above ends with the losers deleted.