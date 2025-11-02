
from terrain_trails.TerrainTrailSTL import generate_stls

generate_stls(
    boundary="tests/resources/post_canyon.gpx",
    #boundary="tests/resources/kingsley.gpx",
    path_width=0.70,
    support_width=0.45,
    path_clearance=0.18,
    base_height=0,
    max_print_size=[150,150],
    tiles=1,
    resolution=10,
    downsample_factor=1,
    waterway_include=["Flume Creek", "Phelps Creek", "Ditch Creek", 767682091, 767682089],
    waterbody=["Lower Green Point Reservoir","Kingsley Reservoir"]
)