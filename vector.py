import logging, rasterio
import geopandas as gpd, numpy as np
from pathlib import Path
from rasterio import transform
from shapely.geometry import Point
#pip8 в данном месте не учтён - правило одна строка = один import, но так в этом случае удобнее

def _process_tiff(input_tiff: str, output_geojson: str, use_pixel_center: bool = True):
    with rasterio.open(input_tiff) as src:
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
        logger.info(f"Размер: {src.width} x {src.height}, CRS: {src.crs}")
        gdf = gpd.GeoDataFrame(geometry=points, crs=src.crs)
        gdf.to_file(output_geojson, driver='GeoJSON')

def process_folder(folder_path, output_folder: str = None, use_pixel_center: bool = True):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError("Папка не найдена")
    tiff_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    if len(tiff_files)==0: 
        raise FileNotFoundError("Файлов нет")
    for tiff_file in tiff_files:
        if output_folder:
            output_path = Path(output_folder) / (tiff_file.stem + ".geojson")
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        else:
            output_path = tiff_file.parent / (tiff_file.stem + ".geojson")   
        _process_tiff(str(tiff_file), str(output_path), use_pixel_center)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
#заметка для ГР - логгер - фиксатор событий внутри кода, для него ставят уровень, колличество не контролируемо в выводе, т.к уровень - INFO() срабатывает и в pyogrio(внутренности geopandas) нужно вручную менять уровень, но тут это некритично.

if __name__ == "__main__":
    territories = ["Amga", "Yunkor"]
    for territory in territories:
        process_folder(folder_path=f"result/masks_rf/{territory}", output_folder=f"result/t_vector/{territory}", use_pixel_center=True)