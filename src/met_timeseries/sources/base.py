"""
Minimal protocol used to describe data source fetch functions.

No abstract base class is used; source modules simply expose a plain function
matching the ``FetchFunction`` signature.
"""


from typing import Protocol
import xarray as xr
from met_timeseries.geometry import BoundingBox

class FetchFunction(Protocol):
    """Protocol satisfied by any ``fetch_*`` function in the sources package."""

    def __call__(
        self,
        bounds: BoundingBox,
        start: str,
        end: str | None = None,
        **kwargs: object,
    ) -> xr.Dataset: ...
