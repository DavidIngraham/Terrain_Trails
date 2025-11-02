"""
Created on Fri Feb 18 16:43:16 2022

@author: jkoet
"""

from copy import deepcopy
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import shapely as shp
import trimesh as tm
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split as shp_split
from importlib import resources


from .coordinate_utils import cord2dist


def rotate_mesh(msh: tm.Trimesh, angle: float) -> tm.Trimesh:
    """Rotate mesh around Z by `angle` (radians)."""
    rotmat = np.zeros((4, 4), dtype=float)
    rotmat[0, 0] = np.cos(angle)
    rotmat[0, 1] = -np.sin(angle)
    rotmat[1, 0] = np.sin(angle)
    rotmat[1, 1] = np.cos(angle)
    rotmat[2, 2] = 1.0
    rotmat[3, 3] = 1.0
    msh.apply_transform(rotmat)
    return msh


def y_dovetails(p1: Polygon, p2: Polygon, rp: Polygon) -> List[List[float]]:
    """
    Locate dovetail positions between two adjacent polygons sharing a horizontal edge
    with p2 above p1. Returns list of [x, y, z] positions (z = 0).
    """
    loc: List[List[float]] = []
    if p1.bounds[3] > p2.bounds[1] + 1e-6:
        print("tiles overlap (Y)")
        return []

    # lower poly
    p1i = p1.intersection(rp)
    if p1i.geom_type == "GeometryCollection":
        return []
    xy = np.array(p1i.exterior.xy).T
    xy = xy[xy[:, 1] > p1i.bounds[3] - 0.1]
    edge_pts = [xy[:, 0].min(), xy[:, 0].max()]

    # upper poly
    p2i = p2.intersection(rp)
    if p2i.geom_type == "GeometryCollection":
        return []
    xy = np.array(p2i.exterior.xy).T
    xy = xy[xy[:, 1] < p1i.bounds[3] + 0.1]
    edge_pts = np.vstack((edge_pts, [xy[:, 0].min(), xy[:, 0].max()]))

    edge_pts = [np.max(edge_pts[:, 0]), np.min(edge_pts[:, 1])]
    span = float(np.diff(edge_pts))
    if abs(span) > 150:
        loc.append([edge_pts[0] + span / 4.0, p1.bounds[3], 0.0])
        loc.append([edge_pts[0] + 3.0 * span / 4.0, p1.bounds[3], 0.0])
    elif abs(span) > 80:
        loc.append([edge_pts[0] + span / 2.0, p1.bounds[3], 0.0])
    return loc


def x_dovetails(p1: Polygon, p2: Polygon, rp: Polygon) -> List[List[float]]:
    """
    Locate dovetail positions between two adjacent polygons sharing a vertical edge
    with p2 to the right of p1. Returns list of [x, y, z] positions (z = 0).
    """
    loc: List[List[float]] = []
    if p1.bounds[2] > p2.bounds[0] + 1e-6:
        print("tiles overlap (X)")
        return []

    # left poly
    p1i = p1.intersection(rp)
    if p1i.geom_type == "GeometryCollection":
        return []
    xy = np.array(p1i.exterior.xy).T
    xy = xy[xy[:, 0] > p1i.bounds[2] - 0.1]
    edge_pts = [xy[:, 1].min(), xy[:, 1].max()]

    # right poly
    p2i = p2.intersection(rp)
    if p2i.geom_type == "GeometryCollection":
        return []
    xy = np.array(p2i.exterior.xy).T
    xy = xy[xy[:, 0] < p1i.bounds[2] + 0.1]
    edge_pts = np.vstack((edge_pts, [xy[:, 1].min(), xy[:, 1].max()]))

    edge_pts = [np.max(edge_pts[:, 0]), np.min(edge_pts[:, 1])]
    span = float(np.diff(edge_pts))
    if abs(span) > 150:
        loc.append([p1.bounds[2], edge_pts[0] + span / 4.0, 0.0])
        loc.append([p1.bounds[2], edge_pts[0] + 3.0 * span / 4.0, 0.0])
    elif abs(span) > 80:
        loc.append([p1.bounds[2], edge_pts[0] + span / 2.0, 0.0])
    return loc


def split_polygon(poly: Polygon, y: float) -> Tuple[Union[Polygon, MultiPolygon, int], Union[Polygon, MultiPolygon]]:
    """
    Split `poly` with a horizontal line at (poly.bounds[1] + y).
    Returns (lower, upper). `lower` may be -1 if everything is above the line.
    """
    split_y = poly.bounds[1] + y
    split_line = LineString([[poly.bounds[0] - 1.0, split_y], [poly.bounds[2] + 1.0, split_y]])
    parts = np.array(shp_split(poly, split_line).geoms, dtype=object)
    # choose upper vs lower by their max Y
    is_upper = np.array([p.bounds[3] for p in parts]) > split_y
    upper = parts[is_upper]
    lower = parts[~is_upper]

    if np.all(is_upper):
        lower_out: Union[Polygon, MultiPolygon, int] = -1
    elif lower.shape[0] > 1:
        lower_out = MultiPolygon(lower.tolist())
    else:
        lower_out = lower[0]

    if not np.any(is_upper):
        upper_out: Union[Polygon, MultiPolygon] = Polygon()
    elif upper.shape[0] > 1:
        upper_out = MultiPolygon(upper.tolist())
    else:
        upper_out = upper[0]

    return lower_out, upper_out


def dovetail_inserts(edge_poly: MultiPolygon, rp: Polygon, dovetail_height: float) -> tm.Trimesh:
    """
    Create dovetail cutout meshes along seams defined by `edge_poly` relative to `rp`.
    """
    resource_path = resources.files(__package__).joinpath("resources")
    
    insert = tm.load(resource_path.joinpath("dovetail_insert.stl"))
    t = np.eye(4)
    t[2, 2] *= (dovetail_height - 0.6) / 10.0
    insert.apply_transform(t)
    insert.export("print_files/dovetail_insert.stl")

    cutout = tm.load(resource_path.joinpath("dovetail_cutout.stl"))
    t = np.eye(4)
    t[2, 2] *= (dovetail_height + 2.0) / 10.0
    cutout.apply_transform(t)
    cutout.apply_translation([0.0, 0.0, -2.0])  # shift down 2mm to prevent co-linear edges

    cutouts: List[tm.Trimesh] = []

    # Horizontal seams (Y)
    cut_loc: List[List[float]] = []
    parts = list(edge_poly.geoms)
    for i in range(len(parts) - 1):
        # Y+
        line = LineString([[parts[i].bounds[0], parts[i].bounds[3] + 0.1],
                           [parts[i].bounds[2], parts[i].bounds[3] + 0.1]])
        for j in range(i + 1, len(parts)):
            if parts[j].intersects(line):
                cut_loc += y_dovetails(parts[i], parts[j], rp)
        # Y-
        line = LineString([[parts[i].bounds[0], parts[i].bounds[1] - 0.1],
                           [parts[i].bounds[2], parts[i].bounds[1] - 0.1]])
        for j in range(i + 1, len(parts)):
            if parts[j].intersects(line):
                cut_loc += y_dovetails(parts[j], parts[i], rp)

    for c in cut_loc:
        new_cutout = deepcopy(cutout)
        new_cutout.apply_translation(c)
        cutouts.append(new_cutout)

    # Vertical seams (X): rotate base cutout sideways
    cutout_rot = rotate_mesh(cutout.copy(), np.pi / 2.0)
    cut_loc = []
    for i in range(len(parts) - 1):
        # X+
        line = LineString([[parts[i].bounds[2] + 0.1, parts[i].bounds[1] + 10.0],
                           [parts[i].bounds[2] + 0.1, parts[i].bounds[3] - 10.0]])
        for j in range(i + 1, len(parts)):
            if parts[j].intersects(line):
                cut_loc += x_dovetails(parts[i], parts[j], rp)
        # X-
        line = LineString([[parts[i].bounds[0] - 0.1, parts[i].bounds[1] + 10.0],
                           [parts[i].bounds[0] - 0.1, parts[i].bounds[3] - 10.0]])
        for j in range(i + 1, len(parts)):
            if parts[j].intersects(line):
                cut_loc += x_dovetails(parts[j], parts[i], rp)

    for c in cut_loc:
        new_cutout = deepcopy(cutout_rot)
        new_cutout.apply_translation(c)
        cutouts.append(new_cutout)

    # Union all cutouts
    return tm.boolean.union(cutouts)


def print_scaling_tiled(
    dem: Any,
    Boundary: Polygon,
    print_size: Sequence[float],
    tiles: int,
    dovetail_height: float,
) -> Tuple[float, np.ndarray, MultiPolygon, List[Any]]:
    """
    Determine scale and tile layout, generate dovetail cutouts if requested.

    Returns
    -------
    scale : float
        Print scale factor.
    corner : np.ndarray shape (2,)
        [min_lon, min_lat] used for cord2dist origin.
    edge_poly : MultiPolygon
        Tile polygons in model coordinates (post-rotation back to original).
    cutouts_wrapped : list
        [cutouts] where cutouts is a Trimesh union (or [] if none).
    """
    print("Optimizing print size and tile layout.")

    x = dem.lon
    y = dem.lat
    corner = np.array([np.min(x), np.min(y)], dtype=float)

    # Border in meters (array of coords), then polygon
    bourder_poly = cord2dist(Boundary, corner=corner)  # noqa: F841  (kept original name)
    print_angle = np.linspace(0.0, np.pi - np.pi / 180.0, 180)

    # Layout rows/cols possibilities
    factors = np.arange(int(tiles)) + 1
    rows = factors[(tiles % factors) == 0]
    rows = np.tile(rows, (print_angle.shape[0], 1))
    print_angle = np.tile(print_angle, (rows.shape[1], 1)).ravel(order="C")
    rows = rows.ravel(order="F")

    scale = np.zeros_like(print_angle, dtype=float)
    corners = np.ones((tiles, 2, print_angle.shape[0]), dtype=float) * np.nan

    for i in range(print_angle.shape[0]):
        col = int(tiles / rows[i])

        # rotated boundary polygon in meters
        rp = shp_rotate(Polygon(bourder_poly), print_angle[i], use_radians=True, origin=[0.0, 0.0])
        scale[i] = print_size[1] * rows[i] / (rp.bounds[3] - rp.bounds[1])

        # refine scale if X is limiting in any row
        scale_ok = False
        while not scale_ok:
            scale_ok = True
            row_height = print_size[1] / scale[i]
            rp_work = rp
            for j in range(rows[i]):
                if j == rows[i] - 1:
                    poly_row = rp_work
                else:
                    lower, rp_work = split_polygon(rp_work, row_height)
                    poly_row = rp_work if lower == -1 else lower  # lower is the bottom chunk

                width = (poly_row.bounds[2] - poly_row.bounds[0])
                if scale[i] > print_size[0] * col / width:
                    scale[i] = print_size[0] * col / width
                    rp_work = shp_rotate(Polygon(bourder_poly), print_angle[i], use_radians=True, origin=[0.0, 0.0])
                    scale_ok = False
                    break

                # tile column offsets for this row
                offset = (width - (np.ceil(width / (print_size[0] / scale[i])) * print_size[0] / scale[i])) / 2.0
                corners[j * col:(j + 1) * col, 0, i] = offset + poly_row.bounds[0] + np.arange(col) * print_size[0] / scale[i]
                corners[j * col:(j + 1) * col, 1, i] = np.full(col, poly_row.bounds[1])
                if getattr(rp_work, "length", 0.0) == 0.0:
                    break

    idx = int(np.argmax(scale))
    print_angle = float(print_angle[idx])

    print("print angle: {0:.2f} deg".format(print_angle * 180.0 / np.pi))

    scale_val = float(scale[idx])
    corners_use = (corners[:, :, idx] * scale_val)
    corners_use = corners_use[~np.isnan(corners_use).any(axis=1), :]

    rows_use = int(rows[idx])

    # Rotate tiles to align with axes for dovetail placement
    rp = shp_rotate(Polygon(np.asarray(bourder_poly) * scale_val), print_angle, use_radians=True, origin=[0.0, 0.0])

    edge_parts: List[Polygon] = []
    rect = np.array([[0.0, 0.0], [0.0, print_size[1]], [print_size[0], print_size[1]], [print_size[0], 0.0]])
    for i in range(corners_use.shape[0]):
        poly = Polygon(corners_use[i, :] + rect)
        if poly.intersection(rp).length != 0.0:
            edge_parts.append(poly)
    edge_poly = MultiPolygon(edge_parts)

    if dovetail_height > 0:
        cutouts = dovetail_inserts(edge_poly, rp, dovetail_height)
        cutouts = rotate_mesh(cutouts, -print_angle)
    else:
        cutouts = []

    # rotate edge polygons back
    edge_poly = shp_rotate(edge_poly, -print_angle, use_radians=True, origin=[0.0, 0.0])

    # DEM resolution (mm) at this scale
    xm, ym = cord2dist(x=x, y=y, corner=corner)
    print(
        "DEM resolution: {0:.2f}x{1:.2f} mm".format(
            ((xm.max() - xm.min()) / (xm.shape[0] - 1) * scale_val),
            ((ym.max() - ym.min()) / (ym.shape[0] - 1) * scale_val),
        )
    )
    dem.scale_factor = scale_val
    return scale_val, corner, edge_poly, [cutouts]
