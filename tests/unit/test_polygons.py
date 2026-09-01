
from met_timeseries.spatial.polygons import _dissolve_catchments
import geopandas as gpd


def test_dissolve_catchments():
    # Create a simple GeoDataFrame with two catchments
    gdf = gpd.GeoDataFrame(
        {
            "metzone": ["A", "A", "B"],
            "geometry": [
                gpd.points_from_xy([0], [0]).buffer(1)[0],
                gpd.points_from_xy([1], [1]).buffer(1)[0],
                gpd.points_from_xy([2], [2]).buffer(1)[0],
            ],
        },
        crs="EPSG:4326",
    )

    dissolved = _dissolve_catchments(gdf, "metzone")
    assert len(dissolved) == 2
    assert "metzone" in dissolved.columns
    assert "geometry" in dissolved.columns
    # Check that the geometries are valid
    assert dissolved.geometry.is_valid.all()
