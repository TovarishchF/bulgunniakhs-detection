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


def main():
    data_root = Path(__file__).parent / "GISIT_Якутск_Данные"

    scenes = collect_scenes(data_root)

    # проверка адекватности и по времени тоже
    for territory in ("Amga", "Yunkor"):
        scenes_list = scenes[territory]
        years = [item["year"] for item in scenes_list]  # генератор списка
        print(f"{territory}: {len(scenes_list)} снимков, годы: {', '.join(map(str, years))}")



if __name__ == "__main__":
    main()