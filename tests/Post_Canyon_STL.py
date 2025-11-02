
from terrain_trails.TerrainTrailSTL import generate_stls

generate_stls(
    boundary="tests/resources/post_canyon.gpx",
    path_width=0.70,
    support_width=0.45,
    path_clearance=0.18,
    height_factor=1,
    base_height=0,
    edge_width=1.5,
    max_print_size=[150,150],
    tiles=1,
    resolution=10,
    downsample_factor=1,
)