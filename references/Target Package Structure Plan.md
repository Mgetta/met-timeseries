# Target Package Structure Plan

Before the tree itself, the three rules that make the structure worth having — the current codebase's problems (dead generations, duplicated clip/cache logic, physics tangled with I/O) are all violations of one of these:

1. **Strict layering.** Imports flow one direction only. A module may import from layers *below* it, never above, never sideways across siblings at the same layer. This is what makes "swap NLDAS for a cloud-hosted source" or "add a second aggregation backend" a local change.
2. **One home per concern.** Every recurring duplicate today (3× `clip_dataset`, 2× cache-merge logic, 2× time-fixing) exists because there was no obvious module that owned it. The structure below names an owner for each.
3. **The store is the waist of the hourglass.** Everything above acquisition reads *only* from the store layer. Sources push in; science reads out. That's the local/cloud interoperability boundary.

---

## Proposed repository layout

```
met-timeseries/
├── pyproject.toml              # single source of dependency truth + extras
├── environment.yml             # conda binaries only (GDAL/eccodes) + pip -e .[dev]
├── README.md
├── references/                 # NOT installed: MetTool_Historic.py, papers (2005-OUDINETP2.pdf)
├── examples/                   # runnable quickstarts, example_catchments.geojson
├── scripts/                    # NOT installed: one-off ops — cache migration/repair,
│   │                           #   benchmarks, backfill drivers (the honest home for
│   │                           #   what _zarr.py's scratch body is today)
├── tests/                      # mirrors src layout (detailed below)
└── src/met_timeseries/
    │
    │  ── Layer 0: foundations (import nothing internal) ──
    ├── __init__.py             # public API façade + __version__; nothing else
    ├── config.py               # Settings: store URIs, region bounds default, storage_options,
    │                           #   auth strategy; env-var overrides  [P3]
    ├── geometry.py             # BoundingBox, bbox↔polygon helpers, THE clip_dataset  [P1]
    ├── schema.py               # canonical variable registry: names, SI units, per-source
    │                           #   native→canonical mappings (absorbs NDAWN's dicts)  [P3/P6]
    ├── validation.py           # time-axis, schema, and physical-range validators  [P2→P6]
    │
    │  ── Layer 1: acquisition (imports: L0) ──
    ├── sources/
    │   ├── __init__.py
    │   ├── base.py             # Fetch contract (Protocol), shared cache adapter:
    │   │                       #   atomic write, read-merge-missing-vars, URI-aware  [P2]
    │   ├── nldas.py            # earthaccess; auth resolved by caller, never inside  [P2]
    │   ├── prism.py
    │   ├── mrms.py
    │   ├── narr.py
    │   └── stations/
    │       ├── __init__.py
    │       ├── ndawn.py        # unit/tz conversion at this boundary → (time, station)  [P6]
    │       └── synoptic.py     # lazy import kept; [stations] extra  [P6]
    │
    │  ── Layer 2: the store — the hourglass waist (imports: L0, L1) ──
    ├── store/
    │   ├── __init__.py         # exports open_source()
    │   ├── ingest.py           # netcdf→zarr chunk-aligned append + pre-append audit
    │   │                       #   (salvaged netcdf_to_zarr / audit_netcdf_cache)  [P3]
    │   ├── manifest.py         # ledger reborn: (source, date, vars, version) records,
    │   │                       #   get_incomplete() for resumable backfills  [P3]
    │   └── reader.py           # open_source(name, config) → lazy canonical Dataset;
    │                           #   schema-on-read rename/units live here  [P3]
    │
    │  ── Layer 3: science engines (imports: L0, store.reader only) ──
    ├── spatial/
    │   ├── __init__.py
    │   ├── polygons.py         # load_polygons (moved as-is)  [P1]
    │   ├── weights.py          # weight kernel + build/save/load weightmap (Zarr)  [P4]
    │   └── aggregate.py        # aggregate_to_zones xr.dot contraction  [P4]
    ├── temporal/
    │   ├── __init__.py         # disaggregate() + dispatch registries
    │   ├── core.py             # conservation framework, freq inference, offsets  [P5]
    │   ├── patterns.py         # diurnal/trapezoidal/solar weight generators,
    │   │                       #   continuous-in-Δt  [P5]
    │   ├── cascade.py          # Molnar-Burlando / Olsson, seeded RNG, dyadic  [P2/P5]
    │   └── resample.py         # conservative_resample (up & down)  [P5]
    ├── derivations/            # pure physics: xarray in → xarray out, zero I/O
    │   ├── __init__.py         # public re-exports (as today)
    │   ├── registry.py         # stage policy: "grid" vs "zone" per derivation  [P5]
    │   ├── constants.py
    │   ├── temperature.py
    │   ├── humidity.py
    │   ├── wind.py
    │   ├── radiation.py        # + zone-centroid pvlib variants  [P5]
    │   └── pet.py
    │
    │  ── Layer 4: products (imports: everything below) ──
    ├── recipes/
    │   ├── __init__.py
    │   ├── recipe.py           # ForcingRecipe dataclass, TOML/JSON I/O, content hash  [P7]
    │   ├── provenance.py       # attrs stamping, catalog table  [P7]
    │   └── runner.py           # recipe → (time, zone) forcing set; ensemble loop  [P7]
    ├── export/
    │   ├── __init__.py
    │   ├── hspf_hdf5.py        # hsp2 layout; grows from utils.hdf5WDM (reader+writer)  [P1 stub → P7]
    │   ├── wdm.py              # classic WDM via wdmtoolbox; [wdm] extra  [P7]
    │   └── parquet.py          # analysis output  [P7]
    │
    │  ── Layer 5: interfaces ──
    ├── cli.py                  # ingest/backfill/extract/run-recipe subcommands  [P3+]
    └── viz.py                  # plot_bounds etc.; matplotlib only here; [viz] extra  [P1]
```

---

## The import DAG (the actual architecture)

```
cli ──► recipes ──► export
              │
              ▼
   spatial  temporal  derivations     (peers — must not import each other,
        \      |      /                EXCEPT: temporal.patterns → derivations.radiation)
         ▼     ▼     ▼
           store.reader
               ▲
   store.ingest ──► sources
               \       │
                ▼      ▼
        config  schema  geometry  validation      (foundations)
```

Enforceable rules, worth writing into the README and eventually a lint check (`importlinter` has a declarative contract format for exactly this):

- **`sources/` never imports `spatial/`, `temporal/`, `derivations/`, `store/`.** Today's `nldas.py` docstring describes weight computation living in the source module — that coupling is explicitly severed. A source's whole job: bytes from a provider → *native-schema* `xr.Dataset` for one date.
- **`derivations/` imports nothing internal except `constants` and siblings.** Pure functions; this is what keeps physics testable with golden values and reusable at either grid or zone stage. The one sanctioned cross-layer edge: `temporal/patterns.py` may call `derivations/radiation.py` for solar weights (it already does today).
- **Science layers read data only through `store.reader`.** No `Path.glob` over NetCDF piles anywhere above Layer 2.
- **Unit/timezone conversions happen at exactly two places:** source adapters (native → SI/UTC canonical) and `export/` (canonical → model-local units/time). If a conversion appears anywhere else, it's a bug by policy.
- **`config.Settings` is constructed at entry points** (`cli.py`, notebooks, `recipes.runner`) and *passed down* — lower layers never reach out to environment variables or global state themselves. This is what makes every layer testable with `memory://` URIs.

---

## Where every current file goes

| Current | Destination | Note |
|---|---|---|
| `MetTool_Historic.py` | `references/` | out of the installable package |
| `pipeline.py` | deleted | ideas resurface as `store.manifest` + `recipes.runner` |
| `_zarr.py` | split: functions → `store/ingest.py`; scratch body → `scripts/` or deleted | |
| `utils.py` | dissolved: `clip_dataset` → `geometry.py`; `hdf5WDM` → `export/hspf_hdf5.py`; `mem_gb` deleted | no `utils.py` in the new tree — deliberately |
| `polygons.py` | `spatial/polygons.py` | unchanged content |
| `weights.py` | `spatial/weights.py` + `spatial/aggregate.py` | split kernel from application |
| `disaggregation.py` (582 lines, 4 concerns) | `temporal/{core, patterns, cascade, resample}.py` | the registries stay in `temporal/__init__.py` |
| `sources/base.py` | `BoundingBox`+`CACHE_BOUNDS` → `geometry.py`/`config.py`; contract stays in `sources/base.py`; `plot_bounds` → `viz.py` | |
| `sources/{nldas,prism,mrms,narr}.py` | stay, slimmed | dedup'd cache/clip logic moves to `sources/base.py` + `geometry.py` |
| `stations/` | `sources/stations/` | stations *are* sources; one acquisition layer, one contract |
| `derivations/` | stays, + `registry.py`, − empty `precipitation.py` | best-organized part of the codebase already |

---

## Design decisions embedded in this structure (and the alternatives I rejected)

**Stations move under `sources/`.** A station network and a gridded product are both "acquisition adapters that emit canonical datasets" — the dims differ (`(time, station)` vs `(time, lat, lon)`), the role doesn't. Keeping a separate top-level `stations/` would force `schema.py` and future consumers to special-case two acquisition trees. The alternative (top-level split) only wins if station handling grows its own large machinery — it hasn't.

**`store/` is a package, not a module.** Ingest, manifest, and reader have different dependency needs (ingest imports `sources`; reader must not) and different failure modes. Separating them keeps the read path — the thing every notebook and engine touches — free of `earthaccess`/download machinery. This is also your cloud seam: `reader.open_source` + fsspec URIs is the *entire* surface that changes between laptop and S3.

**`temporal/` and `spatial/` as peers, forbidden from importing each other.** Your Phase 5 decision (disaggregate areal-mean series, not grids) makes these truly independent stages composed by `recipes/runner.py`. If they can't import each other, the "does disaggregation happen on grids or zones?" confusion that produced the current 1-D/3-D contract bug becomes structurally impossible to reintroduce.

**No `utils.py`, ever.** It's where the current architecture went to dissolve — a WDM reader, a memory profiler, and a clip function shared a file. Every function must claim a layer; if it can't, that's a design smell to resolve, not park.

**`schema.py` at Layer 0, not inside `store/`.** Both source adapters (write side) and `store.reader` (read side) and `validation.py` consult the same registry; putting it in the foundations layer avoids a sources→store import.

**Flat-ish, not deep.** Two levels max (`sources/stations/` is the deepest). Rejected: a `core/` subpackage (foundations sit fine as top-level modules at this scale) and per-source subpackages like `sources/nldas/{fetch,grid}.py` (nothing is big enough yet; split when a file passes ~500 meaningful lines, the threshold `disaggregation.py` already crossed).

**`__init__.py` as a curated façade.** The package root exports the ~10 things a user needs (`open_source`, `load_polygons`, `build_weightmap`, `aggregate_to_zones`, `disaggregate`, `ForcingRecipe`, `Settings`…) so notebooks read cleanly, while module paths stay the contract for internal imports. Underscore-prefix genuinely private helpers; everything else in a module's namespace is fair game for your own notebooks.

---

## Tests and packaging mirror the same shape

```
tests/
├── conftest.py                 # small synthetic grids/polygons (salvage current fixtures)
├── fixtures/                   # tiny static files: mini-NetCDF granules, geojson
├── unit/                       # mirrors src/: test_geometry.py, spatial/test_weights.py,
│   │                           #   temporal/test_conservation.py (the property-test backbone:
│   │                           #   every method × every target_freq must conserve)
├── integration/                # store round-trips on memory:// URIs; ingest→read→aggregate
│                               #   chains; no network
└── network/                    # @pytest.mark.network, skipped by default: one real
                                #   NLDAS/PRISM day, ledger resume, earthaccess auth
```

`pyproject.toml` extras map one-to-one onto optional structure: `[cloud]` → s3fs (only `store/` benefits), `[stations]` → SynopticPy (`sources/stations/synoptic.py`), `[wdm]` → wdmtoolbox (`export/wdm.py`), `[viz]` → matplotlib (`viz.py`), `[dev]` → pytest + importlinter. Core install must import and run the full grid→zone→disaggregate path with none of them.

---

## Migration path — the structure arrives phase by phase, not big-bang

You don't build this tree in one commit; each phase from the implementation plan creates or fills specific slots, so the structure is always fully-wired for what exists:

- **P1:** create `geometry.py`, `viz.py`, `export/` stub, `scripts/`; delete/relocate the dead generation. Tree is shallow but honest.
- **P2:** `validation.py` (time-axis checks), fattened `sources/base.py` (cache adapter, real contract).
- **P3:** `config.py`, `schema.py`, the whole `store/` package, `cli.py`. **This is the last structural move of the "old" modules — after P3 the skeleton is complete** and P4–P7 only fill in `spatial/aggregate.py`, the `temporal/` split, `derivations/registry.py`, `recipes/`, and `export/`.
- **No compatibility shims:** you're the only consumer, so renames are done with `git mv` in dedicated *move-only* commits (no logic changes in the same commit — keeps diffs reviewable and `git blame` useful), and stale import paths die immediately rather than lingering as aliases.

One optional follow-up worth doing at P3: encode the import DAG as an `importlinter` contract in `pyproject.toml` so the layering is CI-enforced rather than documentation-enforced — that's the cheapest possible insurance against this codebase re-growing its current tangles.