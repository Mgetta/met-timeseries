"""Configuration dataclass for the met-timeseries pipeline."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path


def _default_variables() -> list[str]:
    return ["prcp", "rsds", "temp", "wind_speed", "dewpoint", "cloud_cover", "pet"]


def _default_sources() -> list[str]:
    return ["nldas", "prism"]


@dataclass
class PipelineConfig:
    """Configuration for the met-timeseries processing pipeline.

    Parameters
    ----------
    polygon_path : Path
        Path to the polygon GeoJSON or GeoPackage file.
    output_dir : Path
        Root output directory for processed data. Default is ``"output"``.
    start_date : str
        Start date in ``YYYY-MM-DD`` format. Default is ``"1996-01-01"``.
    end_date : str
        End date in ``YYYY-MM-DD`` format, or ``"present"`` to use today's date.
        Default is ``"present"``.
    sources : list[str]
        Data sources to use. Default is ``["nldas", "prism"]``.
    variables : list[str]
        Variables to extract and derive. Default includes all 7 supported variables.
    polygon_id_column : str
        Column name for polygon IDs in the input file. Default is ``"polygon_id"``.
    crs : int
        Target coordinate reference system as an EPSG code. Default is ``4326``.
    """

    polygon_path: Path
    output_dir: Path = field(default_factory=lambda: Path("output"))
    start_date: str = "1996-01-01"
    end_date: str = "present"
    sources: list[str] = field(default_factory=_default_sources)
    variables: list[str] = field(default_factory=_default_variables)
    polygon_id_column: str = "polygon_id"
    crs: int = 4326

    def __post_init__(self) -> None:
        """Validate and normalise configuration values."""
        self.polygon_path = Path(self.polygon_path)
        self.output_dir = Path(self.output_dir)

        # Validate start_date
        try:
            datetime.date.fromisoformat(self.start_date)
        except ValueError as exc:
            raise ValueError(
                f"start_date must be in YYYY-MM-DD format, got {self.start_date!r}"
            ) from exc

        # Validate / resolve end_date
        if self.end_date == "present":
            self.end_date = datetime.date.today().isoformat()
        else:
            try:
                datetime.date.fromisoformat(self.end_date)
            except ValueError as exc:
                raise ValueError(
                    f"end_date must be in YYYY-MM-DD format or 'present', "
                    f"got {self.end_date!r}"
                ) from exc

        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
            )

        valid_sources = {"nldas", "prism"}
        for src in self.sources:
            if src not in valid_sources:
                raise ValueError(
                    f"Unknown source {src!r}. Valid sources are: {valid_sources}"
                )

    @property
    def start_year(self) -> int:
        """First year of the processing period."""
        return int(self.start_date[:4])

    @property
    def end_year(self) -> int:
        """Last year of the processing period."""
        return int(self.end_date[:4])
