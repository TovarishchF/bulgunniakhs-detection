import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import geopandas as gpd
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import subprocess
import sys
#файл не Р, поэтому тут всё по pip8

def ensure_points_generated(points_base_dir):
    if points_base_dir.exists() and any(points_base_dir.glob("**/*.geojson")):
        print("Точки уже существуют, пропускаем vector.py")
        return True
    print("Точки не найдены")
    vector_script = Path(__file__).parent / "vector.py"
    if not vector_script.exists():
        print("Ошибка: vector.py не найден")
        return False
    try:
        subprocess.run([sys.executable, str(vector_script)], check=True, cwd=Path(__file__).parent)
        print("vector.py выполнен успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске vector.py: {e}")
        return False


def get_crs_from_composite(territory, composites_base):
    comp_dir = Path(composites_base) / territory
    tif_files = list(comp_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"Нет композитов в {comp_dir}")
    with rasterio.open(tif_files[0]) as src:
        return src.crs


def generate_heatmap_fast(territory, points_base_dir, composites_base, output_dir,
                          resolution=10, sigma=100, point_weight=0.05):
    points_dir = Path(points_base_dir) / territory
    geojson_files = list(points_dir.glob("*.geojson"))
    if not geojson_files:
        print(f"Нет GeoJSON в {points_dir}")
        return

    target_crs = get_crs_from_composite(territory, composites_base)

    all_points = []
    for f in geojson_files:
        gdf = gpd.read_file(f)
        gdf = gdf.to_crs(target_crs)
        all_points.append(gdf)

    if not all_points:
        print("Нет данных")
        return

    gdf_all = pd.concat(all_points, ignore_index=True)
    gdf_all = gpd.GeoDataFrame(gdf_all, geometry='geometry', crs=target_crs)
    if len(gdf_all) == 0:
        print("Нет точек")
        return

    coords = np.array([(geom.x, geom.y) for geom in gdf_all.geometry])
    crs = target_crs

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    margin = 0.05 * max(xmax - xmin, ymax - ymin)
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin

    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))
    transform = Affine(resolution, 0, xmin, 0, -resolution, ymax)

    mask = np.zeros((height, width), dtype=np.float32)
    for x, y in coords:
        col = int(round((x - xmin) / resolution))
        row = int(round((ymax - y) / resolution))
        if 0 <= row < height and 0 <= col < width:
            mask[row, col] += point_weight

    sigma_pixels = sigma / resolution
    density = gaussian_filter(mask, sigma=sigma_pixels, mode='reflect')

    # Логарифмическое масштабирование
    density_log = np.log1p(density)
    if density_log.max() > 0:
        density_norm = density_log / density_log.max()
    else:
        density_norm = density_log

    out_dir = Path(output_dir) / territory
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": np.float32, "crs": crs, "transform": transform, "compress": "lzw", "nodata": np.nan}
    raster_path = out_dir / "heatmap_density.tif"
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(density.astype(np.float32), 1)

    # Сохраняем PNG с логарифмической нормировкой
    cmap = LinearSegmentedColormap.from_list('hot', ['black', 'purple', 'red', 'yellow', 'white'])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(density_norm, extent=[xmin, xmax, ymin, ymax], origin='upper', cmap=cmap, alpha=0.7)
    plt.colorbar(im, ax=ax, label='Плотность точек (логарифм)')
    ax.set_title(f'Тепловая карта – {territory}')
    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    plt.tight_layout()
    png_path = out_dir / "heatmap.png"
    plt.savefig(png_path, dpi=150) #снова есть момент с базовым dpi
    plt.close()
    print(f"Тепловая карта для {territory} сохранена в {png_path}")

if __name__ == "__main__":
    output_dir = Path(__file__).parent / "result" / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)
    points_base_dir = Path(__file__).parent / "result" / "t_vector"
    composites_base = Path(__file__).parent / "result" / "composites"

    if not ensure_points_generated(points_base_dir):
        print("Не удалось получить точки.")
        sys.exit(1) #Р доволен 

    for territory in ["Amga", "Yunkor"]:
        try:
            generate_heatmap_fast(territory, points_base_dir, composites_base, output_dir, resolution=10, sigma=100, point_weight=0.05)
        except Exception as e:
            print(f"Ошибка для {territory}: {e}")