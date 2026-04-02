from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import sobel, gaussian_filter
from tqdm import tqdm

def collect_scenes(data_root: Path):
    scenes = {"Amga": [], "Yunkor": []}
    for tif in data_root.rglob("*.tif"):
        path_str = str(tif).upper()
        if "АМГА" in path_str:
            territory = "Amga"
        elif "ЮНКОР" in path_str:
            territory = "Yunkor"
        else:
            continue

        m = re.search(r"KANOPUS_(\d{8})_", tif.name)
        if not m:
            continue
        try:
            date = datetime.strptime(m.group(1), "%Y%m%d")
        except:
            continue

        name = tif.name.upper()
        if ".L2.MS." in name:
            sensor = "MS"
        elif ".L2.PMS." in name:
            sensor = "PMS"
        else:
            continue

        scenes[territory].append({"path": tif, "year": date.year, "date": date, "sensor": sensor})

    for key in scenes:
        scenes[key].sort(key=lambda x: x["year"])
    return scenes

def reproject_to_ref(src_path, ref_transform, ref_crs, ref_height, ref_width):
    with rasterio.open(src_path) as src:
        dst = np.zeros((src.count, ref_height, ref_width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, range(1, src.count + 1)),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        return dst

def align_and_downsample(scenes: dict, output_dir: Path = None):
    amga_ms = next(item for item in scenes["Amga"] if item["sensor"] == "MS")
    amga_pms = next(item for item in scenes["Amga"] if item["sensor"] == "PMS")
    yunkor_ms = next(item for item in scenes["Yunkor"] if item["sensor"] == "MS")

    with rasterio.open(amga_ms["path"]) as src:
        ref_meta = src.meta.copy()
        ref_transform = src.transform
        ref_crs = src.crs
        ref_height, ref_width = src.height, src.width

    amga_pms_downsampled = reproject_to_ref(amga_pms["path"], ref_transform, ref_crs, ref_height, ref_width)
    yunkor_ms_aligned = reproject_to_ref(yunkor_ms["path"], ref_transform, ref_crs, ref_height, ref_width)
    result = {'reference_meta': ref_meta, 'amga_pms_downsampled': amga_pms_downsampled, 'yunkor_ms_aligned': yunkor_ms_aligned}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in [('amga_pms_downsampled', amga_pms_downsampled),('yunkor_ms_aligned', yunkor_ms_aligned)]:
            out_path = output_dir / f"{name}.tif"
            meta_out = ref_meta.copy()
            meta_out.update({'driver': 'GTiff', 'height': ref_height, 'width': ref_width,
                             'count': arr.shape[0], 'dtype': arr.dtype})
            with rasterio.open(out_path, 'w', **meta_out) as dst:
                for i in range(arr.shape[0]):
                    dst.write(arr[i], i+1)
            result[f'{name}_path'] = out_path

    return result

def prepare_ms_timeseries(scenes, output_dir=None):
    aligned = {}

    for territory in ["Amga", "Yunkor"]:
        all_items = scenes[territory]
        if not all_items:
            continue

        ms_items = [item for item in all_items if item["sensor"] == "MS"]
        if not ms_items:
            print(f"{territory}: нет MS-снимков, пропускаем")
            continue

        ref_item = ms_items[0]
        with rasterio.open(ref_item["path"]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_height = src.height
            ref_width = src.width
            ref_meta = src.meta.copy()

        arrays = []
        for item in tqdm(all_items, desc=f"Aligning {territory} (MS+PMS)"):
            if item["sensor"] == "MS" and item["path"] == ref_item["path"]:
                with rasterio.open(item["path"]) as src:
                    ms_array = src.read()
            else:
                ms_array = reproject_to_ref(
                    item["path"], ref_transform, ref_crs, ref_height, ref_width
                )
            arrays.append({
                "year": item["year"],
                "date": item["date"],
                "ms_array": ms_array
            })

        if output_dir:
            out_dir = Path(output_dir) / territory
            out_dir.mkdir(parents=True, exist_ok=True)
            for sc in arrays:
                out_path = out_dir / f"{territory}_{sc['year']}_{sc['date'].strftime('%Y%m%d')}_aligned.tif"
                meta_out = ref_meta.copy()
                meta_out.update({
                    "driver": "GTiff",
                    "height": ref_height,
                    "width": ref_width,
                    "count": sc["ms_array"].shape[0],
                    "dtype": sc["ms_array"].dtype
                })
                with rasterio.open(out_path, "w", **meta_out) as dst:
                    dst.write(sc["ms_array"])

        aligned[territory] = {
            "arrays": arrays,
            "ref_meta": ref_meta,
            "ref_transform": ref_transform,
            "ref_crs": ref_crs,
            "ref_height": ref_height,
            "ref_width": ref_width
        }

    return aligned

get_blue = lambda array: array[0, :, :].astype(np.float32)
get_green = lambda array: array[1, :, :].astype(np.float32)
get_red = lambda array: array[2, :, :].astype(np.float32)
get_nir = lambda array: array[3, :, :].astype(np.float32)

def compute_ndwi(ms_array):
    green = get_green(ms_array)
    nir = get_nir(ms_array)
    ndwi = (green - nir) / (green + nir + 1e-10)
    return ndwi

def compute_ndvi(ms_array):
    red = get_red(ms_array)
    nir = get_nir(ms_array)
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

def compute_slope(dem, resolution):
    dzdx = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * resolution)
    dzdy = (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0)) / (2 * resolution)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * (180 / np.pi)
    return slope.astype(np.float32)

def compute_aspect(dem, resolution):
    dzdx = (np.roll(dem, -1, axis=1) - np.roll(dem, 1, axis=1)) / (2 * resolution)
    dzdy = (np.roll(dem, -1, axis=0) - np.roll(dem, 1, axis=0)) / (2 * resolution)
    aspect = np.arctan2(dzdy, dzdx) * (180 / np.pi)
    aspect = 90 - aspect
    aspect[aspect < 0] += 360
    return aspect.astype(np.float32)

def process_dem(territory, dem_path, ref_meta, output_dir):
    with rasterio.open(dem_path) as src_dem:
        dem_data = src_dem.read(1).astype(np.float32)
        src_transform = src_dem.transform
        src_crs = src_dem.crs

    dst_shape = (ref_meta['height'], ref_meta['width'])
    dem_aligned = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=dem_data,
        destination=dem_aligned,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_meta['transform'],
        dst_crs=ref_meta['crs'],
        resampling=Resampling.bilinear
    )
    dem_smoothed = gaussian_filter(dem_aligned, sigma=1)

    res_x = abs(ref_meta['transform'].a)
    res_y = abs(ref_meta['transform'].e)
    resolution = (res_x + res_y) / 2
    slope = compute_slope(dem_smoothed, resolution)
    aspect = compute_aspect(dem_smoothed, resolution)

    out_dir = Path(output_dir) / "morpho"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = ref_meta.copy()
    meta_out.update({'count': 1, 'dtype': np.float32, 'compress': 'lzw'})

    slope_path = out_dir / f"{territory}_slope.tif"
    with rasterio.open(slope_path, 'w', **meta_out) as dst:
        dst.write(slope, 1)

    aspect_path = out_dir / f"{territory}_aspect.tif"
    with rasterio.open(aspect_path, 'w', **meta_out) as dst:
        dst.write(aspect, 1)

    return {
        'slope': slope_path,
        'aspect': aspect_path,
    }

def reproject_dem_to_match(dem_path, target_meta):
    with rasterio.open(dem_path) as src_dem:
        dem_data = src_dem.read(1).astype(np.float32)
        src_transform = src_dem.transform
        src_crs = src_dem.crs

    dst_shape = (target_meta['height'], target_meta['width'])
    dem_aligned = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=dem_data,
        destination=dem_aligned,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_meta['transform'],
        dst_crs=target_meta['crs'],
        resampling=Resampling.bilinear
    )
    dem_aligned = gaussian_filter(dem_aligned, sigma=1)
    return dem_aligned

def load_morpho_layers(morpho_paths):
    """Загружает морфометрические слои из файлов."""
    layers = []
    with rasterio.open(morpho_paths['slope']) as src:
        layers.append(src.read(1).astype(np.float32)[np.newaxis, :, :])
    with rasterio.open(morpho_paths['aspect']) as src:
        layers.append(src.read(1).astype(np.float32)[np.newaxis, :, :])
    return np.concatenate(layers, axis=0)

def get_feature_functions():
    features = OrderedDict()
    features["B"] = get_blue
    features["G"] = get_green
    features["R"] = get_red
    features["NIR"] = get_nir
    features["NDWI"] = compute_ndwi
    features["NDVI"] = compute_ndvi
    return features

def build_composite(ms_array, feature_dict=None):
    if feature_dict is None:
        feature_dict = get_feature_functions()
    layers = []
    for name, func in feature_dict.items():
        layer = func(ms_array)
        layers.append(layer[np.newaxis, :, :])
    return np.concatenate(layers, axis=0)

def main():
    data_root = Path(__file__).parent / "GISIT_Якутск_Данные"
    scenes = collect_scenes(data_root)

    results_dir = Path(__file__).parent / "result"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем DEM для территорий (один раз)
    dem_paths = {}
    for territory in ["Amga", "Yunkor"]:
        dem_path = Path(__file__).parent / f"{territory}_dem.tif"
        if dem_path.exists():
            dem_paths[territory] = dem_path
            print(f"DEM для {territory} найден, будет добавлен уклон и экспозиция")
        else:
            print(f"DEM для {territory} не найден, морфометрические признаки не добавлены")

    for territory in ["Amga", "Yunkor"]:
        items = scenes[territory]
        if not items:
            continue

        for item in tqdm(items, desc=f"Processing {territory}"):
            if item["sensor"] != "MS":
                continue  

            with rasterio.open(item["path"]) as src:
                ms_array = src.read().astype(np.float32)
                meta = src.meta.copy()
            composite = build_composite(ms_array)

            if territory in dem_paths:
                dem_aligned = reproject_dem_to_match(dem_paths[territory], meta)
                # Размер пикселя 
                res_x = abs(meta['transform'].a)
                res_y = abs(meta['transform'].e)
                resolution = (res_x + res_y) / 2
                slope = compute_slope(dem_aligned, resolution)
                aspect = compute_aspect(dem_aligned, resolution)
                composite = np.concatenate([
                    composite,
                    slope[np.newaxis, :, :],
                    aspect[np.newaxis, :, :]
                ], axis=0)

            date_str = item["date"].strftime("%d%m%Y")
            out_path = results_dir / "composites" / territory / f"{date_str}.tif"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            meta_out = meta.copy()
            meta_out.update({
                "driver": "GTiff",
                "count": composite.shape[0],
                "dtype": composite.dtype,
                "compress": "lzw"
            })
            with rasterio.open(out_path, "w", **meta_out) as dst:
                dst.write(composite)
            print(f"{territory} {date_str}: композит ({composite.shape[0]} каналов) сохранён в {out_path}")

if __name__ == "__main__":
    main()