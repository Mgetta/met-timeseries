


pd.concat([ms.stations.nearby(ms.Point(point.y,point.x), limit=4) for point in list(catchments_subset.centroid)])