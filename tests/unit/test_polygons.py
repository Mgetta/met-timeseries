
from met_timeseries.spatial.polygons import _dissolve_catchments


def test_dissolve_catchments(catchment_gdf):
    dissolved = _dissolve_catchments(catchment_gdf, "metzone")
    assert len(dissolved) == 2
    assert "metzone" in dissolved.columns
    assert "geometry" in dissolved.columns
    # Check that the geometries are valid
    assert dissolved.geometry.is_valid.all()
