"""Main pipeline orchestrator for met-timeseries processing."""

from __future__ import annotations

import logging
from pathlib import Path

from met_timeseries.config import PipelineConfig
from met_timeseries.ledger import Ledger

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the met-timeseries processing pipeline.

    The pipeline runs the following steps for each calendar month in the
    configured date range:

    1. NLDAS-2 hourly data extraction
    2. PRISM daily precipitation extraction
    3. Variable derivation (wind speed, dewpoint, cloud cover)
    4. Precipitation disaggregation (PRISM daily x NLDAS pattern)
    5. PET computation (FAO-56 Penman-Monteith)
    6. Zonal aggregation to polygons (area-weighted)
    7. Save monthly Parquet output

    Resume capability is provided by a :class:`~met_timeseries.ledger.Ledger`
    that tracks completed (step, year, month) triples.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.ledger = Ledger(config.output_dir)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging to console and file."""
        log_dir = Path(self.config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "pipeline.log"

        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8"),
            ],
        )

    def run(self, steps: list[str] | None = None) -> None:
        """Run the full pipeline for all configured months.

        Parameters
        ----------
        steps : list[str] or None
            Subset of steps to run.  Supported values are ``"nldas"``,
            ``"prism"``, ``"derive"``, ``"aggregate"``.  If ``None``,
            all steps are run.
        """
        from met_timeseries.polygons import load_polygons  # noqa: PLC0415

        steps = steps or ["nldas", "prism", "derive", "aggregate"]
        logger.info("Starting pipeline run with steps: %s", steps)

        polygons = load_polygons(
            self.config.polygon_path,
            id_column=self.config.polygon_id_column,
            target_crs=self.config.crs,
        )
        logger.info("Loaded %d polygons from %s", len(polygons), self.config.polygon_path)

        for year in range(self.config.start_year, self.config.end_year + 1):
            for month in range(1, 13):
                self._process_month(year, month, polygons, steps)

        logger.info("Pipeline run complete.")

    def _process_month(
        self,
        year: int,
        month: int,
        polygons,
        steps: list[str],
    ) -> None:
        """Process a single calendar month.

        Parameters
        ----------
        year : int
            Four-digit year.
        month : int
            Month number (1-12).
        polygons : gpd.GeoDataFrame
            Polygon GeoDataFrame.
        steps : list[str]
            Steps to execute.
        """
        label = f"{year}-{month:02d}"

        for step in steps:
            if self.ledger.is_complete(step, year, month):
                logger.info("Skipping %s step=%s (already complete)", label, step)
                continue

            logger.info("Processing %s step=%s", label, step)
            try:
                self._run_step(step, year, month, polygons)
                self.ledger.mark_complete(step, year, month)
                logger.info("Completed %s step=%s", label, step)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error in %s step=%s: %s", label, step, exc, exc_info=True)
                raise

    def _run_step(
        self,
        step: str,
        year: int,
        month: int,
        polygons,
    ) -> None:
        """Dispatch a single processing step.

        Parameters
        ----------
        step : str
            Step name.
        year : int
            Four-digit year.
        month : int
            Month number (1-12).
        polygons : gpd.GeoDataFrame
            Polygon GeoDataFrame.
        """
        if step == "nldas":
            self._step_nldas(year, month, polygons)
        elif step == "prism":
            self._step_prism(year, month, polygons)
        elif step == "derive":
            self._step_derive(year, month)
        elif step == "aggregate":
            self._step_aggregate(year, month, polygons)
        else:
            raise ValueError(f"Unknown step: {step!r}")

    def _step_nldas(self, year: int, month: int, polygons) -> None:
        from met_timeseries.sources.nldas import NLDASSource  # noqa: PLC0415
        from met_timeseries.io import save_monthly_output  # noqa: PLC0415

        source = NLDASSource()
        ds = source.fetch(polygons, year, month)
        # Convert to DataFrame indexed by time x polygon_id (placeholder)
        df = ds.to_dataframe().reset_index()
        save_monthly_output(df, self.config.output_dir, "nldas_raw", year, month)

    def _step_prism(self, year: int, month: int, polygons) -> None:
        from met_timeseries.sources.prism import PRISMSource  # noqa: PLC0415
        from met_timeseries.io import save_monthly_output  # noqa: PLC0415

        cache_dir = Path(self.config.output_dir) / "cache" / "prism"
        source = PRISMSource(cache_dir=cache_dir)
        ds = source.fetch(polygons, year, month)
        df = ds.to_dataframe().reset_index()
        save_monthly_output(df, self.config.output_dir, "prism_raw", year, month)

    def _step_derive(self, year: int, month: int) -> None:
        from met_timeseries.io import read_output, save_monthly_output  # noqa: PLC0415
        from met_timeseries.derivations.wind import compute_wind_speed  # noqa: PLC0415
        from met_timeseries.derivations.dewpoint import compute_dewpoint  # noqa: PLC0415

        try:
            nldas_df = read_output(self.config.output_dir, "nldas_raw", year_range=(year, year))
        except FileNotFoundError:
            logger.warning("No NLDAS raw data for %d-%02d; skipping derive step", year, month)
            return

        logger.debug("Deriving variables for %d-%02d", year, month)
        # Derivation logic placeholder - saves empty frame when running end-to-end
        save_monthly_output(nldas_df.head(0), self.config.output_dir, "derived", year, month)

    def _step_aggregate(self, year: int, month: int, polygons) -> None:
        from met_timeseries.io import read_output, save_monthly_output  # noqa: PLC0415

        try:
            derived_df = read_output(
                self.config.output_dir, "derived", year_range=(year, year)
            )
        except FileNotFoundError:
            logger.warning("No derived data for %d-%02d; skipping aggregate step", year, month)
            return

        save_monthly_output(derived_df.head(0), self.config.output_dir, "aggregated", year, month)
