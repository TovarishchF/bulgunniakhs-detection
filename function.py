#!/usr/bin/env python3
import pandas as pd
import glob
import os
import sys
import argparse
import zipfile
from shapely import wkt
import geopandas as gpd
from pathlib import Path
import csv


def find_csv_files(input_folder):
    csv_sources = []
    zip_files = glob.glob(os.path.join(input_folder, "*.zip"))
    print(f"ZIP архивов: {len(zip_files)}")
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                csv_in_zip = [f for f in zf.namelist() if f.endswith('.csv')]
                if not csv_in_zip:
                    print(f"В {os.path.basename(zip_file)} нет CSV файлов")
                    continue
                for csv_name in csv_in_zip:
                    csv_sources.append({'source': zip_file,'type': 'zip', 'csv_name': csv_name, 'name': f"{os.path.basename(zip_file)}/{csv_name}"})
        except Exception as e:
            print(f"НЕ ПРОЧИТАЛОСЬ - {os.path.basename(zip_file)}: {e}")
    return csv_sources


def read_csv_from_zip(source_info):
    with zipfile.ZipFile(source_info['source'], 'r') as zf:
        with zf.open(source_info['csv_name']) as csv_file:
            content = csv_file.read().decode('utf-8', errors='ignore')
            rows = []
            for line in content.split('\n'):
                if not line.strip():
                    continue
                reader = csv.reader([line], quotechar='"', delimiter=',')
                try:
                    row = next(reader)
                    rows.append(row)
                except:
                    rows.append(line.split(','))
            if not rows:
                return None
            
            headers = rows[0]
            headers = [h.strip('"') for h in headers]
            
            data_rows = []
            for row in rows[1:]:
                if len(row) < len(headers):
                    row.extend([''] * (len(headers) - len(row)))
                elif len(row) > len(headers):
                    if 'POLYGON' in row[-1] or 'MULTIPOLYGON' in row[-1]:
                        pass
                    else:
                        extra_cols = row[len(headers)-1:]
                        row = row[:len(headers)-1] + [','.join(extra_cols)]
                data_rows.append(row)
            df = pd.DataFrame(data_rows, columns=headers)
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip('"')
            return df


def clean_wkt(wkt_str):
    if pd.isna(wkt_str):
        return None
    wkt_str = str(wkt_str).strip()
    wkt_str = wkt_str.strip('"')
    wkt_str = wkt_str.replace('\\"', '"')
    wkt_str = wkt_str.replace('\x00', '')
    
    if not wkt_str or wkt_str == '':
        return None
    
    wkt_upper = wkt_str.upper()
    if not (wkt_upper.startswith('POLYGON') or wkt_upper.startswith('MULTIPOLYGON')):
        if wkt_str.startswith('((('):
            wkt_str = f'MULTIPOLYGON {wkt_str}'
        elif wkt_str.startswith('(('):
            wkt_str = f'POLYGON {wkt_str}'
    return wkt_str


def safe_wkt_loads(wkt_str):
    try:
        if wkt_str is None:
            return None
        geom = wkt.loads(wkt_str)
        if geom.is_valid:
            return geom
        else:
            from shapely.validation import make_valid
            geom_fixed = make_valid(geom)
            if geom_fixed.is_valid:
                return geom_fixed
            return None
    except Exception as e:
        # Для отладки раскомментировать: print(f"Ошибка WKT: {str(e)[:50]}, строка: {str(wkt_str)[:100]}")
        return None


def merge_csv_to_geojson(input_folder, output_file, wkt_column='OBJ_WKT', dropna=True):
    csv_sources = find_csv_files(input_folder)
    print(f"\nCSV файлов: {len(csv_sources)}")
    dataframes = []
    stats = {'success': 0, 'errors': 0, 'total_rows': 0, 'failed_files': []}
    
    for i, source in enumerate(csv_sources, 1):
        try:
            df = read_csv_from_zip(source)
            if df is None or len(df) == 0:
                stats['errors'] += 1
                stats['failed_files'].append(source['name'])
                print(f"{i}/{len(csv_sources)} {source['name']}: пустой файл")
                continue
            
            if wkt_column not in df.columns:
                stats['errors'] += 1
                stats['failed_files'].append(source['name'])
                print(f"{i}/{len(csv_sources)} {source['name']}: нет столбца '{wkt_column}'")
                continue
            
            df[wkt_column] = df[wkt_column].apply(clean_wkt)
            df = df.dropna(subset=[wkt_column])
            
            if len(df) == 0:
                stats['errors'] += 1
                stats['failed_files'].append(source['name'])
                print(f"ХА, не получилось - {i}/{len(csv_sources)} {source['name']}: нет валидного WKT")
                continue
            
            dataframes.append(df)
            stats['success'] += 1
            stats['total_rows'] += len(df)
            print(f"+{i}/{len(csv_sources)} {source['name']}: {len(df)} объектов")
                
        except Exception as e:
            stats['errors'] += 1
            stats['failed_files'].append(source['name'])
            print(f"{i}/{len(csv_sources)} {source['name']}: {str(e)[:100]}")
            continue
    
    if not dataframes:
        print("\nЧТО-ТО ПОЛЕТЕЛО", f"ПРИМЕРНО ВОТ ТУТ: {len(stats['failed_files'])}")
        return False
    
    print(f"Прочитано: {stats['success']} из {len(csv_sources)}")
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    print(f"Количество объектов: {len(combined_df)}")
    
    try:
        combined_df['geometry'] = combined_df[wkt_column].apply(safe_wkt_loads)
        valid_geoms = combined_df['geometry'].notna().sum()
        print(f"Валидных полигонов: {valid_geoms}, невалидных: {len(combined_df) - valid_geoms}")
        if dropna:
            combined_df = combined_df.dropna(subset=['geometry'])
        
    except Exception as e:
        print(f"Ошибка преобразования WKT: {e}")
        return False
    
    gdf = gpd.GeoDataFrame(combined_df, geometry='geometry', crs="EPSG:4326")
    if wkt_column in gdf.columns: gdf = gdf.drop(columns=[wkt_column])
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    
    try:
        '''
        gdf.to_file(output_file, driver="GeoJSON")
        print(f"\nФайл сохранён: {output_file}")
        print(f"Объектов: {len(gdf)}")
        print(f"Размер: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
        '''
        print()
        # Обрезка по Amga_big_AOI
        amga_file = 'scene/Amga_big_AOI.geojson'
        if os.path.exists(amga_file):
            amga_aoi = gpd.read_file(amga_file)
            print(f"CRS исходных данных: {gdf.crs}")
            print(f"CRS AOI: {amga_aoi.crs}")
            if gdf.crs != amga_aoi.crs:
                print(f"Разные CRS! Приводим к {gdf.crs}")
                amga_aoi = amga_aoi.to_crs(gdf.crs)
            amga_result = gpd.overlay(gdf, amga_aoi, how='intersection')
            amga_output = 'result/amga_clipped.geojson'
            Path('result').mkdir(parents=True, exist_ok=True)
            amga_result.to_file(amga_output, driver='GeoJSON')
            print(f"Объектов до: {len(gdf)}, объектов после: {len(amga_result)}")
            print(f"Сохранён: {amga_output}")
        else:
            print(f"\nФайл не найден: {amga_file}")
        
        # Обрезка по Yunkor_big_AOI
        print()
        yunkor_file = 'scene/Yunkor_big_AOI.geojson'
        if os.path.exists(yunkor_file):
            yunkor_aoi = gpd.read_file(yunkor_file)
            print(f"CRS исходных данных: {gdf.crs}")
            print(f"CRS AOI: {yunkor_aoi.crs}")
            if gdf.crs != yunkor_aoi.crs:
                print(f"Разные CRS! Приводим к {gdf.crs}")
                yunkor_aoi = yunkor_aoi.to_crs(gdf.crs)
            yunkor_result = gpd.overlay(gdf, yunkor_aoi, how='intersection')
            yunkor_output = 'result/yunkor_clipped.geojson'
            yunkor_result.to_file(yunkor_output, driver='GeoJSON')
            print(f"Объектов до: {len(gdf)}, объектов после: {len(yunkor_result)}")
            print(f"Сохранён: {yunkor_output}")
        else:
            print(f"\nФайл не найден: {yunkor_file}")
        return True
    except Exception as e:
        print(f"Ошибка сохранения: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Объединение CSV из ZIP в GeoJSON', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', '-i', default='Земельный Кадастр_Якутия', help='Папка с ZIP архивами (по умолчанию: Земельный Кадастр_Якутия)')
    parser.add_argument('--output', '-o', default='result/merged.geojson', help='Выходной GeoJSON (по умолчанию: result/merged.geojson)')
    parser.add_argument('--wkt-column', '-w', default='OBJ_WKT', help='Название столбца с WKT (по умолчанию: OBJ_WKT)')
    parser.add_argument('--keep-na', action='store_true', help='Не удалять строки с пустой геометрией')
    
    args = parser.parse_args()
    success = merge_csv_to_geojson(input_folder=args.input, output_file=args.output, wkt_column=args.wkt_column, dropna=not args.keep_na)
    
    sys.exit(0 if success else 1)