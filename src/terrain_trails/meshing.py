# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 06:51:38 2022

@author: jkoet
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 06:51:38 2022

@author: jkoet
"""

import numpy as np
import shapely as shp
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, LineString
import trimesh as tm
import matplotlib as mpl
from terrain_trails.coordinate_utils import *
from terrain_trails.dem import Dem
from typing import Union, Sequence, List, Tuple

Geom = Union[Polygon, MultiPolygon]
Number = Union[int, float]
ArrayLike = Union[np.ndarray, Sequence[Number]]

def delete_small_holes(mp: Polygon | MultiPolygon, offset: float, min_area: float) -> MultiPolygon:
    """
    Delete small holes in a polygon or multipolygon.
    """
    if offset > 0:
        offset = -offset

    if isinstance(mp, MultiPolygon):
        polygons = []
        for poly in mp.geoms:
            holes = [
                hole for hole in poly.interiors
                if Polygon(hole).buffer(offset).area > 0.1
            ]
            if poly.area >= min_area:
                polygons.append(Polygon(poly.exterior, holes))
        return MultiPolygon(polygons)
    else:
        holes = [
            hole for hole in mp.interiors
            if Polygon(hole).buffer(offset).area > 0.1
        ]
        if mp.area >= min_area:
            return Polygon(mp.exterior, holes)
        return MultiPolygon()


def del_small_holes_buffer(
    path: Polygon, offset: float, clearance: float, cutout: Polygon | MultiPolygon
) -> Polygon:
    """
    Delete small holes in a polygon buffer.
    """
    if offset > 0:
        offset = -offset

    cutout = cutout.union(path.buffer(clearance)) if not isinstance(cutout, list) else path.buffer(clearance)

    if isinstance(cutout, MultiPolygon):
        for poly in cutout.geoms:
            for hole in poly.interiors:
                hole_poly = Polygon(hole)
                if hole_poly.buffer(offset).area < 0.1:
                    path = path.union(hole_poly.buffer(-offset))
    else:
        for hole in cutout.interiors:
            hole_poly = Polygon(hole)
            if hole_poly.buffer(offset).area < 0.1:
                path = path.union(hole_poly.buffer(-offset))
    return path


def border_points(dem: Dem, poly: np.ndarray) -> np.ndarray:
    """
    Generate points along the border of a polygon along DEM grid lines.
    """
    pts = [poly]
    for i in range(poly.shape[0] - 1):
        # x crossings
        x = dem.lon[(dem.lon >= min(poly[i:i + 2, 0])) & (dem.lon <= max(poly[i:i + 2, 0]))]
        y = poly[i, 1] + (x - poly[i, 0]) * np.diff(poly[i:i + 2, 1]) / np.diff(poly[i:i + 2, 0])
        pts.append(np.column_stack((x, y)))

        # y crossings
        y = dem.lat[(dem.lat >= min(poly[i:i + 2, 1])) & (dem.lat <= max(poly[i:i + 2, 1]))]
        x = poly[i, 0] + (y - poly[i, 1]) * np.diff(poly[i:i + 2, 0]) / np.diff(poly[i:i + 2, 1])
        pts.append(np.column_stack((x, y)))

    return np.vstack(pts)


def border_points_2(dem: Dem, poly: np.ndarray) -> np.ndarray:
    """
    Generate points along the border of a polygon along DEM grid lines.
    """
    pts = []
    for i in range(poly.shape[0] - 1):
        # x crossings
        n = max(
            sum((dem.lon >= min(poly[i:i + 2, 0])) & (dem.lon <= max(poly[i:i + 2, 0]))),
            sum((dem.lat >= min(poly[i:i + 2, 1])) & (dem.lat <= max(poly[i:i + 2, 1]))),
        )

        x = np.linspace(poly[i, 0], poly[i + 1, 0], n + 1)[1:]
        y = np.linspace(poly[i, 1], poly[i + 1, 1], n + 1)[1:]

        pts.append(np.column_stack((x, y)))

    return np.vstack(pts)


def terrain_mesh(
    dem: Dem,
    Boundary: Polygon,
    scale: float,
    corner: np.ndarray,
    height_factor: float,
    base_height: float,
    water_drop: float,
) -> tm.Trimesh:
    """
    Create a terrain mesh that bounds a lat/lon polygon.
    """
    x, y = cord2dist(x=dem.lon, y=dem.lat, corner=corner, f=scale)
    gridsize = np.mean([np.diff(x[:2]), np.diff(y[:2])])

    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    edge = border_points_2(dem, dist2cord(xy=np.array(Boundary.exterior.xy).T, corner=dem.corner, f=scale))

    pts = MultiPoint(np.column_stack((x, y)))
    idx = np.array([Boundary.buffer(-gridsize * 0.5).contains(pt) for pt in pts.geoms])
    x, y = x[idx], y[idx]
    idx = idx.reshape(dem.lat.shape[0], dem.lon.shape[0])

    z = np.hstack((dem.z[idx], dem.get_elev(edge)))
    edge = cord2dist(edge, corner=corner, f=scale)

    z = (z - z.min()) * scale * height_factor + base_height + 1

    verts = np.vstack((x, y)).T
    verts = np.vstack((verts, edge))

    surf = mpl.tri.Triangulation(verts[:, 0], verts[:, 1])

    vert_cutoff = verts.shape[0] - edge.shape[0]
    t = surf.triangles
    mask = np.any(t > vert_cutoff, axis=1)
    surf.set_mask(mask)

    msh = tm.creation.extrude_triangulation(verts, surf.get_masked_triangles(), 5)

    verts = verts[:vert_cutoff, :]
    verts = dist2cord(verts, corner=corner, f=scale)

    idx = msh.vertices[:, 2] > 0
    z = dem.get_elev(dist2cord(msh.vertices[idx, :2], corner=corner, f=scale))
    z = (z - dem.z.min()) * scale * height_factor + base_height + 1
    msh.vertices[idx, 2] = z

    tm.repair.fix_normals(msh)
    return msh

def included_points(
    poly: Polygon,
    dem: Dem,
    corner: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """
    Vectorized selection of DEM grid points inside polygon (in distance space).
    Returns a boolean mask shaped like dem.z / dem.lat×dem.lon.
    """
    grid_lon, grid_lat = np.meshgrid(dem.lon, dem.lat)
    pts_ll = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))
    pts_xy = cord2dist(xy=pts_ll, corner=corner, f=scale_factor)

    # Vectorized point-in-polygon
    mask_flat = shp.contains(poly, shp.points(pts_xy[:, 0], pts_xy[:, 1]))
    return mask_flat.reshape(dem.lat.shape[0], dem.lon.shape[0])


def buffer_smooth(ply: Geom, offset: Union[float, List[float]], s: float = 0.0) -> Geom:
    """
    Apply (optionally compound) buffer operations, then simplify.
    """
    if not isinstance(offset, list):
        offset = [offset]

    out = shp.buffer(ply, offset[0])
    if len(offset) == 1:
        out = shp.buffer(out, -offset[0])
    else:
        out = shp.buffer(out, -(offset[0] + offset[1]))
        out = shp.buffer(out, +offset[1])

    if s > 0:
        out = out.simplify(s)  # preserve_topology default True in 2.0
    return out


def meshgen_wb(ply: Union[Geom, List[Geom]], h: List[float], fname: str, elev: List[float]) -> List[tm.Trimesh]:
    """
    Extrude one or more (multi)polygons by heights `h`, translate by `elev`, optionally export STL.
    """
    if not isinstance(ply, list):
        ply = [ply]

    meshes: List[tm.Trimesh] = []
    for i in range(len(ply)):
        geom = ply[i]
        if geom.geom_type == "MultiPolygon":
            parts = []
            for g in shp.get_parts(geom):
                if g.area > 1.0:
                    m = tm.creation.extrude_polygon(g, h[i])
                    m.apply_translation([0, 0, elev[i]])
                    parts.append(m)
            mesh = tm.boolean.union(parts) if parts else tm.Trimesh()
        else:
            mesh = tm.creation.extrude_polygon(geom, h[i])
            mesh.apply_translation([0, 0, elev[0]])
        meshes.append(mesh)
        if fname:
            mesh.export(f"{fname}_{i+1}.stl")
    return meshes


def meshgen2(
    ply: Union[Geom, List[Geom]],
    dem: Dem,
    corner: np.ndarray,
    sf: float,
    hf: float,
    base: float,
    fname: str | List[str] | None = None,
) -> List[tm.Trimesh]:
    """
    Extrude polygons to a height based on DEM range under each polygon.
    """
    if not isinstance(ply, list):
        ply = [ply]

    out: List[tm.Trimesh] = []
    for i, geom in enumerate(ply):
        if geom.geom_type == "MultiPolygon":
            parts = []
            for g in shp.get_parts(geom):
                if g.area > 0.5:
                    c = np.vstack(g.exterior.xy).T
                    c = dist2cord(c, corner=corner, f=sf)
                    z = (dem.get_elev(c) - float(np.min(dem.z))) * sf * hf + base + 1.0 - 3.0
                    m = tm.creation.extrude_polygon(g, float(np.max(z) - np.min(z) + 4.0))
                    m.apply_translation([0, 0, float(np.min(z))])
                    parts.append(m)
            mesh = tm.boolean.union(parts) if parts else tm.Trimesh()
            out.append(mesh)
            if fname:
                name = fname if isinstance(fname, str) else fname[i]
                mesh.export(f"{name}_{i+1}.stl")
        else:
            c = np.vstack(geom.exterior.xy).T
            c = dist2cord(c, corner=corner, f=sf)
            z = (dem.get_elev(c) - float(np.min(dem.z))) * sf * hf + base + 1.0 - 3.0
            mesh = tm.creation.extrude_polygon(geom, float(np.max(z) - np.min(z) + 4.0))
            mesh.apply_translation([0, 0, float(np.min(z))])
            out.append(mesh)
            if fname:
                name = fname if isinstance(fname, str) else fname[i]
                mesh.export(f"{name}_{i+1}.stl")
    return out


def offset_lines(lines: LineString, offsets: Union[float, List[float]]) -> List[Geom]:
    """
    Buffer a LineString by one or more offsets.
    """
    if not isinstance(offsets, list):
        offsets = [offsets]
    return [shp.buffer(lines, o) for o in offsets]


def offset_polygon(ply: Geom, offsets: Union[float, List[float]]) -> List[Geom]:
    """
    Offset a polygon/multipolygon by one or more distances.
    """
    if not isinstance(offsets, list):
        offsets = [offsets]

    out: List[Geom] = []
    for o in offsets:
        if o == 0:
            out.append(ply)
            continue
        if ply.geom_type == "MultiPolygon":
            # Buffer each part; flatten MultiPolygons produced by buffer
            buffed_parts: List[Polygon] = []
            for g in shp.get_parts(ply):
                b = shp.buffer(g, o)
                if isinstance(b, MultiPolygon):
                    buffed_parts.extend(list(shp.get_parts(b)))
                else:
                    buffed_parts.append(b)
            out.append(MultiPolygon(buffed_parts))
        else:
            out.append(shp.buffer(ply, o))
    return out


def merge_paths_2d(
    paths: List[LineString],
    b: List[Geom],
    pWidth: float,
    sWidth: float,
    clearance: float,
) -> Tuple[List[Geom], Geom]:
    """
    Inflate line strings into cutout/top/support polygons with priorities.

    Returns
    -------
    tuple(polys + [cutout])
      polys: list of [cutout, top, support] for each path (may be empty list if path is empty)
      cutout: union of all cutouts
    """
    polys: List[List[Geom]] = []
    cutout: Geom = MultiPolygon()  # start empty

    for i, path in enumerate(list(paths)):
        if path.is_empty:
            polys.append([])
            continue

        # Primary path polygon (priority order)
        p = offset_lines(path, pWidth / 2.0)[0]
        p = p.intersection(b[0])

        # delete tiny interior holes in buffered path region
        p = del_small_holes_buffer(p, pWidth / 2.0, clearance, cutout)

        # subtract earlier cutouts (priority rule)
        if polys:
            for prev in polys:
                if prev:
                    p = p.difference(prev[0])

        # Build three sizes: [cutout, top, support]
        trip = offset_polygon(p, [clearance / 2.0, 0.0, -(pWidth - sWidth) / 2.0])

        # smoothing
        trip[0] = buffer_smooth(trip[0], [-0.22], clearance / 2.0)
        trip[1] = buffer_smooth(trip[1], [+0.12, -0.12], clearance / 4.0)
        trip[2] = buffer_smooth(trip[2], [-0.02], clearance / 4.0)

        polys.append(trip)

        # update global cutout union (vectorized-friendly union_all)
        cutout = trip[0] if shp.is_empty(cutout) else shp.union_all([cutout, trip[0]])

    return tuple(polys + [cutout])  # type: ignore[return-value]