import logging
from pathlib import Path
import rasterio
from rasterio import transform
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _process_tiff(input_tiff: str, output_geojson: str, use_pixel_center: bool = True):
    with rasterio.open(input_tiff) as src:
        logger.info(f"Размер: {src.width} x {src.height}, CRS: {src.crs}")
        band = src.read(1)
        rows, cols = np.where(band == 1)             
        points = []
        affine_transform = src.transform
        for row, col in zip(rows, cols):
            if use_pixel_center: 
                x, y = transform.xy(affine_transform, row, col)
            else: 
                x, y = transform.xy(affine_transform, row, col, offset='ul')
            points.append(Point(x, y))
        gdf = gpd.GeoDataFrame(geometry=points, crs=src.crs)
        gdf.to_file(output_geojson, driver='GeoJSON')

def process_folder(folder_path: str, output_folder: str = None, use_pixel_center: bool = True):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена")
    tiff_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    for tiff_file in tiff_files:
        if output_folder:
            output_path = Path(output_folder) / (tiff_file.stem + ".geojson")
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        else:
            output_path = tiff_file.parent / (tiff_file.stem + ".geojson")   
        _process_tiff(str(tiff_file), str(output_path), use_pixel_center)

if __name__ == "__main__":
    territories = ["Amga", "Yunkor"]
    for territory in tqdm(territories, desc="Обработка территорий"):
        input_folder = f"result/masks_rf/{territory}"
        output_folder = f"result/t_vector/{territory}"
        process_folder(folder_path=input_folder, output_folder=output_folder, use_pixel_center=True)