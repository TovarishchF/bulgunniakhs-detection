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