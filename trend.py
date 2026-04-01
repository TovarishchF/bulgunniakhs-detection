from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject
from rasterio.transform import Affine
from affine import Affine as AffineClass
import geopandas as gpd
from shapely.geometry import mapping, box
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def collect_scenes(data_root: Path):
    """
    Читает папку и возвращает сцены по территориям:
    {
        "Amga": [ {path, year, sensor}, ... ],
        "Yunkor": [ {path, year, sensor}, ... ]
    }
    """

    scenes = {
        "Amga": [],
        "Yunkor": []
    }

    for tif in data_root.rglob("*.tif"):
        path_str = str(tif).upper()

        # определяем территорию
        if "АМГА" in path_str:
            territory = "Amga"
        elif "ЮНКОР" in path_str:
            territory = "Yunkor"
        else:
            continue

        # извлекаем дату
        m = re.search(r"KANOPUS_(\d{8})_", tif.name)
        if not m:
            continue

        try:
            date = datetime.strptime(m.group(1), "%Y%m%d")
        except:
            continue

        # определяем формат
        name = tif.name.upper()
        if ".L2.MS." in name:
            sensor = "MS"
        elif ".L2.PMS." in name:
            sensor = "PMS"
        else:
            continue

        scenes[territory].append({
            "path": tif,
            "year": date.year,
            "sensor": sensor
        })

    # сортировка по времени
    for key in scenes:
        scenes[key].sort(key=lambda x: x["year"])

    return scenes

def reproject_to_ref(src_path, ref_transform, ref_crs, ref_height, ref_width):
    """Перепроецирует изображение на опорную сетку."""
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

    result = {
        'reference_meta': ref_meta,
        'amga_pms_downsampled': amga_pms_downsampled,
        'yunkor_ms_aligned': yunkor_ms_aligned
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in [('amga_pms_downsampled', amga_pms_downsampled),
                          ('yunkor_ms_aligned', yunkor_ms_aligned)]:
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
    """
    Для каждой территории (Amga, Yunkor) приводит все MS-снимки к единой сетке,
    используя первый MS-снимок территории как опорный.
    Возвращает словарь: {territory: {'arrays': [{'year': year, 'ms_array': arr}],
                                      'ref_meta': метаданные опорного снимка}}
    """
    aligned = {}

    for territory in ["Amga", "Yunkor"]:
        ms_items = [item for item in scenes[territory] if item["sensor"] == "MS"]
        if not ms_items:
            continue

        # Опорный снимок — первый (уже отсортировано по годам)
        ref_item = ms_items[0]
        with rasterio.open(ref_item["path"]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_height = src.height
            ref_width = src.width
            ref_meta = src.meta.copy()

        arrays = []
        for item in tqdm(ms_items, desc=f"Aligning {territory} MS"):
            if item["path"] == ref_item["path"]:
                with rasterio.open(item["path"]) as src:
                    ms_array = src.read()
            else:
                ms_array = reproject_to_ref(
                    item["path"], ref_transform, ref_crs, ref_height, ref_width
                )
            arrays.append({"year": item["year"], "ms_array": ms_array})

        # Сохраняем выровненные файлы (опционально)
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for sc in arrays:
                out_path = out_dir / f"{territory}_{sc['year']}_aligned.tif"
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

def compute_ndwi(ms_array):
    """
    NDWI = (Green - NIR) / (Green + NIR)
    Предполагается порядок каналов: 0 – Blue, 1 – Green, 2 – Red, 3 – NIR.
    """
    green = ms_array[1, :, :].astype(np.float32)
    nir = ms_array[3, :, :].astype(np.float32)
    ndwi = (green - nir) / (green + nir + 1e-10)
    return ndwi

def compute_trend(index_arrays, years):
    """
    Для каждого пикселя строит линейную регрессию.
    index_arrays: list of 2D numpy arrays (height, width) – значения индекса
    years: list of int (соответствует порядку)
    Возвращает: slope, intercept
    """
    height, width = index_arrays[0].shape
    slope = np.full((height, width), np.nan, dtype=np.float32)
    intercept = np.full((height, width), np.nan, dtype=np.float32)

    years = np.array(years, dtype=np.float32)
    years_mean = years.mean()
    years_centered = years - years_mean

    for i in tqdm(range(height), desc="Pixel trend"):
        for j in range(width):
            y = [arr[i, j] for arr in index_arrays]
            valid = ~np.isnan(y)
            if np.sum(valid) < 2:
                continue
            x = years_centered[valid]
            y_valid = np.array(y)[valid]
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y_valid)
            slope[i, j] = model.coef_[0]
            intercept[i, j] = model.intercept_

    return slope, intercept

def main():
    data_root = Path(__file__).parent / "GISIT_Якутск_Данные"
    scenes = collect_scenes(data_root)

    # Папка для результатов
    results_dir = Path(__file__).parent / "result"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Выравниваем MS-снимки для каждой территории
    aligned = prepare_ms_timeseries(scenes, output_dir=data_root / "aligned_ms")

    for territory in ["Amga", "Yunkor"]:
        if territory not in aligned:
            print(f"{territory}: нет MS-снимков, пропускаем")
            continue

        # Получаем данные территории
        data = aligned[territory]   # это словарь с ключами 'arrays', 'ref_meta', ...
        arrays = data["arrays"]      # список {'year': ..., 'ms_array': ...}
        arrays.sort(key=lambda x: x["year"])  # сортируем по году

        years = [sc["year"] for sc in arrays]
        ndwi_arrays = [compute_ndwi(sc["ms_array"]) for sc in arrays]

        slope, intercept = compute_trend(ndwi_arrays, years)

        out_slope = results_dir / f"{territory}_NDWI_trend_slope.tif"
        out_intercept = results_dir / f"{territory}_NDWI_trend_intercept.tif"

        # Метаданные из опорного снимка
        meta_out = data["ref_meta"].copy()
        meta_out.update({"count": 1, "dtype": slope.dtype, "compress": "lzw"})

        with rasterio.open(out_slope, "w", **meta_out) as dst:
            dst.write(slope, 1)
        with rasterio.open(out_intercept, "w", **meta_out) as dst:
            dst.write(intercept, 1)

        print(f"{territory}: тренд сохранён в {out_slope} и {out_intercept}")

if __name__ == "__main__":
    main()