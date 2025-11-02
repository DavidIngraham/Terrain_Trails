from __future__ import annotations

import os
import math
from typing import Iterable, List, Tuple

import numpy as np
import requests
import rasterio
from tqdm import tqdm
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.transform import xy, Affine
from rasterio.warp import reproject, Resampling
from shapely.geometry import Polygon, box, mapping
from platformdirs import user_cache_dir
import matplotlib.pyplot as plt


TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"


def _dataset_for_res(res: int) -> tuple[str, str]:
    """
    Return (dataset_name, preferred_format) for TNM Access.
    Supports 10 m (1/3") and 30 m (1").
    """
    if res == 10:
        return "National Elevation Dataset (NED) 1/3 arc-second", "GeoTIFF"
    if res == 30:
        return "National Elevation Dataset (NED) 1 arc-second", "GeoTIFF"
    raise ValueError("res must be 10 or 30 (meters)")


def _deg_buffer_for_res(res: int) -> float:
    """
    Small geographic buffer in degrees (~10 pixels) to ensure clean edges.
    """
    # 1/3″ ≈ 1/10800 deg per pixel; 1″ ≈ 1/3600 deg per pixel
    return (10 / 10800.0) if res == 10 else (10 / 3600.0)


def _meters_to_degrees(east_m: float, north_m: float, ref_lat_deg: float) -> tuple[float, float]:
    """
    Convert meter offsets (east, north) to degrees (lon, lat) near ref_lat.
    """
    lat_deg = north_m / 111_111.0
    lon_deg = east_m / (math.cos(math.radians(ref_lat_deg)) * 111_111.0)
    return lon_deg, lat_deg


def _query_tnm(dataset: str, bbox: tuple[float, float, float, float], prod_format: str | None) -> list[dict]:
    """
    Query TNM Access for items in dataset intersecting bbox.
    """
    params = {
        "datasets": dataset,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "max": 50,
        "outputFormat": "json",
    }
    if prod_format:
        params["prodFormats"] = prod_format
    r = requests.get(TNM_API, params=params, timeout=45)
    r.raise_for_status()
    return r.json().get("items", [])


def _download(url: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(out_path),
            disable=(total == 0),
        ) as pbar:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return out_path

def downsample_raster(z, transform, dsf, method=Resampling.average, nodata=None):
    if dsf <= 1:
        return z, transform

    print("Downsampling DEM Data")

    h, w = z.shape
    H, W = int(h / dsf), int(w / dsf)

    # Pre-fill with dst nodata
    fill = nodata if nodata is not None else 0
    z_ds = np.full((H, W), fill, dtype="float32")

    transform_ds = transform * Affine.scale(dsf)

    reproject(
        source=z.astype("float32"),
        destination=z_ds,
        src_transform=transform,
        src_crs="EPSG:4326",
        dst_transform=transform_ds,
        dst_crs="EPSG:4326",
        resampling=method,
        src_nodata=fill,
        dst_nodata=fill,
        init_dest_nodata=True,   # <- important: keep nodata where no valid samples
    )
    if np.min(z_ds) < -100:
        raise Warning('DEM contains large negative numbers after resampling')

    return z_ds, transform_ds

class Dem:
    """
    Auto-downloading DEM wrapper with the same public API:

        Dem(poly, res, offset, dsf)
        get_elev(l)
        plot_elev()

    Parameters
    ----------
    poly : array-like (N×2) of [lon, lat] vertices in degrees
        The area of interest polygon. Only its bounds are used for clipping.
    res : int
        10 (≈1/3") or 30 (≈1") meters.
    offset : array-like (2,) of [east_m, north_m]
        Offset in meters to add to the output coordinates (useful for meshing frames).
    dsf : int
        Downsample factor. Use 1 to disable.
    """

    def __init__(self, poly, res, offset, dsf):
        print("Loading Elevation Data...")

        # ---- Validate & normalize inputs ----
        poly = np.asarray(poly, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2:
            raise ValueError("poly must be an Nx2 array of [lon, lat]")

        res = int(res)
        if res not in (10, 30):
            raise ValueError("res must be 10 or 30 (meters)")

        dsf = int(dsf)
        east_m = float(offset[0])
        north_m = float(offset[1])

        dataset, fmt = _dataset_for_res(res)
        buf_deg = _deg_buffer_for_res(res)

        # Convert offset meters → degrees (near the minimum polygon latitude)
        ref_lat = float(poly[:, 1].min())
        lon_off_deg, lat_off_deg = _meters_to_degrees(east_m, north_m, ref_lat)

        # AOI bounds with a small buffer; we download in geographic coords (EPSG:4326)
        lon_min, lat_min = float(poly[:, 0].min()) - buf_deg, float(poly[:, 1].min()) - buf_deg
        lon_max, lat_max = float(poly[:, 0].max()) + buf_deg, float(poly[:, 1].max()) + buf_deg
        aoi_bbox = (lon_min, lat_min, lon_max, lat_max)

        # ---- Cache directory ----
        cache_dir = user_cache_dir("terrain_trails", "dinglabs")
        os.makedirs(cache_dir, exist_ok=True)
        print(f'Using DEM cache at: {cache_dir}')

        # ---- Query TNM & download all intersecting GeoTIFFs ----
        items = _query_tnm(dataset, aoi_bbox, fmt)
        if not items:
            items = _query_tnm(dataset, aoi_bbox, None)
        if not items:
            raise RuntimeError("No DEM tiles returned by TNM for this bbox/resolution.")

        local_paths: list[str] = []
        for it in items:
            url = it.get("downloadURL")
            if not url:
                continue
            name = os.path.basename(url.split("?")[0])
            if not name.lower().endswith(".tif"):
                # Skip non-TIFF items (some entries are containers/archives)
                continue
            out_path = os.path.join(cache_dir, name)
            if not os.path.exists(out_path):
                try:
                    _download(url, out_path)
                except Exception as e:
                    print(f"Warning: failed to download {url}: {e}")
                    continue
            local_paths.append(out_path)

        if not local_paths:
            raise RuntimeError("No usable GeoTIFFs found/downloaded for the requested area.")

        # ---- Mosaic all tiles ----
        srcs= [rasterio.open(p) for p in local_paths]
        profile = srcs[0].profile.copy()
        mosaic, transform = merge(srcs)

        # Squeeze (handles shapes like (1, 1, H, W) -> (H, W), or (1, H, W) -> (H, W))
        mosaic = np.squeeze(mosaic)
        if mosaic.ndim == 3:          # still (bands, H, W) for some reason
            mosaic = mosaic[0, :, :]  # take first band
        elif mosaic.ndim != 2:
            raise ValueError(f"Unexpected mosaic ndim={mosaic.ndim}")

        # Update profile to match the 2-D array
        mem_profile = profile.copy()
        mem_profile.update(
            driver="GTiff",
            height=mosaic.shape[0],
            width=mosaic.shape[1],
            transform=transform,
            count=1,
            dtype=mosaic.dtype,
        )

        geom = mapping(box(*aoi_bbox))
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**mem_profile) as ds:
                ds.write(mosaic, 1)  # write a 2-D array to band 1
                clipped, clipped_transform = mask(ds, [geom], crop=True)

        # clipped comes back as (bands, H, W); take band 1
        z = np.squeeze(clipped)
        if z.ndim == 3:
            z = z[0, :, :]
        elif z.ndim != 2:
            raise ValueError(f"Unexpected clipped ndim={z.ndim}")

        # ---- Optional block-average downsampling ----
        if dsf > 1:
            z, clipped_transform = downsample_raster(z, clipped_transform, dsf, method=Resampling.average)


        # ---- Build lon/lat coordinate axes (pixel centers) ----
        # Use rasterio.transform.xy for accuracy (handles pixel center properly)
        height, width = z.shape
        cols = np.arange(width)
        rows = np.arange(height)

        # Vectorized via list-comprehension (fast enough for typical DEM tiles)
        xs = np.array([xy(clipped_transform, 0, c, offset="center")[0] for c in cols], dtype=float)
        ys = np.array([xy(clipped_transform, r, 0, offset="center")[1] for r in rows], dtype=float)


        # ---- Apply requested output offset to coordinates (not to the raster) ----
        self.lon = xs + lon_off_deg
        self.lat = ys + lat_off_deg
        self.z = z.astype("float32")

        # ---- Final tight crop to the original polygon bounds (optional but keeps parity with your behavior) ----
        lon_sel = (self.lon >= poly[:, 0].min() - buf_deg) & (self.lon <= poly[:, 0].max() + buf_deg)
        lat_sel = (self.lat >= poly[:, 1].min() - buf_deg) & (self.lat <= poly[:, 1].max() + buf_deg)
        self.z = self.z[np.where(lat_sel)[0]][:, np.where(lon_sel)[0]]
        self.lat = self.lat[lat_sel]
        self.lon = self.lon[lon_sel]
        self.corner = [float(self.lon[0]), float(self.lat[0])]

        print('DEM Loaded')

    # ---- Public methods (same names) ----

    def get_elev(self, l: np.ndarray) -> np.ndarray:
        """
        Nearest-neighbor elevation lookup.

        Parameters
        ----------
        l : array-like (M×2) of [lon, lat]

        Returns
        -------
        np.ndarray shape (M,)
        """
        pts = np.asarray(l, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("l must be an Mx2 array of [lon, lat]")

        # Grid spacings (guard against 1-element axes)
        dy = (self.lat[-1] - self.lat[0]) / max(1, (len(self.lat) - 1))
        dx = (self.lon[-1] - self.lon[0]) / max(1, (len(self.lon) - 1))

        r = np.clip(np.round((pts[:, 1] - self.lat[0]) / dy).astype(int), 0, len(self.lat) - 1)
        c = np.clip(np.round((pts[:, 0] - self.lon[0]) / dx).astype(int), 0, len(self.lon) - 1)
        return self.z[r, c]

    def plot_elev(self) -> None:
        """
        Quicklook plot of the DEM (with geographic extents).
        """
        plt.imshow(
            np.flip(self.z, axis=0),
            cmap="viridis",
            extent=[self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()],
            aspect="auto",
        )
        plt.colorbar(label="Elevation (m)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("DEM")
        plt.show()
