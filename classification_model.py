import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
from scipy.ndimage import generic_filter
import joblib
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage import morphology
import warnings
import sys
warnings.filterwarnings("ignore")

RF_CONFIG = {
    "n_estimators": 100, #200
    "max_depth": 16, #20
    "random_state": 42, #42
    "n_jobs": -1, #-1
    "class_weight": "balanced", #'balanced'
    "oob_score": True
}

if '--train' in sys.argv:
    TRAIN_NEW_MODEL = True  # обучить новую модель
else:
    TRAIN_NEW_MODEL = False  # не обучать

# удалить
TRAIN_NEW_MODEL = True

MODEL_PATH = Path(__file__).parent / "Amga_rf_multisource.pkl"

# Текстурные признаки (std в окне 3×3)
def compute_texture_features(image, window_size=3):
    C, H, W = image.shape
    texture = np.zeros_like(image)
    for c in range(C):
        texture[c] = generic_filter(image[c], np.std, size=window_size, mode='reflect')
    return texture

# Репроекция GeoJSON
def reproject_geojson(geojson_path, dst_crs):
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        raise ValueError("GeoJSON не имеет проекции. Укажите CRS в файле.")
    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
    return gdf

# Загрузка одного композита и маски
def load_training_data(composite_path, geojson_path):
    with rasterio.open(composite_path) as src:
        image = src.read().astype(np.float32)
        meta = src.meta.copy()
        dst_crs = src.crs

    gdf = reproject_geojson(geojson_path, dst_crs)

    if 'class' in gdf.columns:
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf['class'])]
    else:
        shapes = [(geom, 1) for geom in gdf.geometry]

    mask = rasterize(
        shapes,
        out_shape=(meta['height'], meta['width']),
        transform=meta['transform'],
        fill=0,
        dtype=np.uint8
    )
    obj_pixels = np.sum(mask)
    print(f"  Объектов: {obj_pixels} пикселей (всего {mask.size})")
    if obj_pixels == 0:
        raise ValueError("Маска пуста")
    return image, mask, meta

# Сбор пикселей из одного снимка (исходные каналы + текстуры)
def collect_samples(image, mask, texture, background_ratio=5):
    combined = np.concatenate([image, texture], axis=0)
    C = combined.shape[0]

    h, w = mask.shape
    obj_idx = np.where(mask == 1)
    n_obj = len(obj_idx[0])
    if n_obj == 0:
        return np.empty((0, C)), np.empty(0)

    bg_idx = np.where(mask == 0)
    n_bg = min(len(bg_idx[0]), n_obj * background_ratio)
    chosen_bg = np.random.choice(len(bg_idx[0]), n_bg, replace=False)
    bg_y = bg_idx[0][chosen_bg]
    bg_x = bg_idx[1][chosen_bg]

    X = []
    y = []
    for i in range(n_obj):
        X.append(combined[:, obj_idx[0][i], obj_idx[1][i]])
        y.append(1)
    for i in range(n_bg):
        X.append(combined[:, bg_y[i], bg_x[i]])
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"    Собрано {len(X)} пикселей (объектов: {n_obj}, фон: {n_bg}), признаков: {X.shape[1]}")
    return X, y

# Нормализация
def normalize_by_stats(image, mean, std, eps=1e-6):
    C = image.shape[0]
    for c in range(C):
        image[c] = (image[c] - mean[c]) / (std[c] + eps)
    return image

def compute_stats_from_samples(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, std

# Обучение модели на объединённых данных
def train_and_evaluate(X, y, config):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_thr = 0.5
    for thr in thresholds:
        y_pred_bin = (y_pred_prob >= thr).astype(int)
        f1 = f1_score(y_val, y_pred_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    y_pred_bin = (y_pred_prob >= best_thr).astype(int)
    print("\n=== Оценка на отложенной выборке (20%) ===")
    print(f"Оптимальный порог: {best_thr:.3f}")
    print(f"Accuracy: {accuracy_score(y_val, y_pred_bin):.4f}")
    print(f"Recall (объекты): {recall_score(y_val, y_pred_bin):.4f}")
    print(f"F1-score: {f1_score(y_val, y_pred_bin):.4f}")
    print("\nОтчёт классификации:\n", classification_report(y_val, y_pred_bin, target_names=['фон', 'объект']))
    print("Матрица ошибок:\n", confusion_matrix(y_val, y_pred_bin))

    print("\n=== Важность признаков ===")
    n_features = X.shape[1] // 2
    for i in range(n_features):
        orig_imp = model.feature_importances_[i]
        tex_imp = model.feature_importances_[i + n_features]
        print(f"Признак {i} (исх): {orig_imp:.4f} | текстура {i}: {tex_imp:.4f}")

    # Сохраняем валидационные данные для визуализации
    val_dir = Path("validation_data")
    val_dir.mkdir(parents=True, exist_ok=True)
    np.save(val_dir / "y_val.npy", y_val)
    np.save(val_dir / "y_pred.npy", y_pred_bin)
    np.save(val_dir / "y_prob.npy", y_pred_prob)
    print(f"Валидационные данные сохранены в {val_dir}")

    return model, best_thr

# Предсказание для полного изображения
def predict_image(model, image, mean_stats, std_stats, batch_size=50000, threshold=0.5):
    texture = compute_texture_features(image)
    combined = np.concatenate([image, texture], axis=0)
    img_norm = combined.copy()
    normalize_by_stats(img_norm, mean_stats, std_stats)

    C, H, W = img_norm.shape
    image_flat = img_norm.reshape(C, -1).T
    n_samples = image_flat.shape[0]
    prob = np.zeros(n_samples, dtype=np.float32)
    for i in range(0, n_samples, batch_size):
        batch = image_flat[i:i+batch_size]
        prob[i:i+batch_size] = model.predict_proba(batch)[:, 1]
    prob_map = prob.reshape(H, W)
    print(f"  Вероятности: min={prob_map.min():.3f}, max={prob_map.max():.3f}, mean={prob_map.mean():.3f}")
    return prob_map

# Постобработка
def postprocess_mask(prob_map, threshold, min_area=100):
    binary = (prob_map > threshold).astype(np.uint8)
    if np.sum(binary) == 0:
        return binary
    label_img, num_labels = ndi.label(binary)
    sizes = np.bincount(label_img.ravel())
    for i in range(1, num_labels+1):
        if sizes[i] < min_area:
            binary[label_img == i] = 0
    binary = morphology.closing(binary, morphology.square(3))
    return binary

# Основная функция
def main():
    training_pairs = [
        ("Amga", "27082019", "27082019.geojson"),
        ("Amga", "06052025", "06052025.geojson"),
        ("Yunkor", "13072021", "13072021.geojson"),
    ]
    territories_to_predict = ["Amga", "Yunkor"]
    background_ratio = 5

    base_dir = Path(__file__).parent
    composite_base = base_dir / "result" / "composites"
    scene_dir = base_dir / "scene"
    output_dir = base_dir / "result" / "masks_rf"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка или обучение модели
    if TRAIN_NEW_MODEL:
        # Сбор обучающих пикселей
        all_X = []
        all_y = []
        print("Загрузка обучающих данных...")
        for territory, comp_name, geojson_name in training_pairs:
            comp_path = composite_base / territory / f"{comp_name}.tif"
            geojson_path = scene_dir / geojson_name
            if not comp_path.exists():
                print(f"  Пропуск: композит {comp_path} не найден")
                continue
            if not geojson_path.exists():
                print(f"  Пропуск: GeoJSON {geojson_path} не найден")
                continue
            print(f"Обработка {comp_name}...")
            image, mask, _ = load_training_data(comp_path, geojson_path)
            texture = compute_texture_features(image)
            X_pair, y_pair = collect_samples(image, mask, texture, background_ratio)
            if len(X_pair) > 0:
                all_X.append(X_pair)
                all_y.append(y_pair)

        if not all_X:
            raise RuntimeError("Нет обучающих данных! Проверьте пути.")
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        print(f"\nВсего собрано {len(X)} пикселей (объектов: {np.sum(y==1)}, фон: {np.sum(y==0)})")

        # 2. Нормализация и обучение
        mean_stats, std_stats = compute_stats_from_samples(X)
        print(f"Статистики каналов (первые 6 mean): {mean_stats[:6]}...")
        X_norm = X.copy()
        for i in range(X_norm.shape[1]):
            X_norm[:, i] = (X_norm[:, i] - mean_stats[i]) / (std_stats[i] + 1e-6)
        model, best_thr = train_and_evaluate(X_norm, y, RF_CONFIG)

        # 3. Сохранение модели
        joblib.dump({'model': model, 'mean': mean_stats, 'std': std_stats, 'threshold': best_thr}, MODEL_PATH)
        print(f"Модель сохранена в {MODEL_PATH}")
    else:
        # Загрузка существующей модели
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}. Установите TRAIN_NEW_MODEL = True для обучения.")
        checkpoint = joblib.load(MODEL_PATH)
        model = checkpoint['model']
        mean_stats = checkpoint['mean']
        std_stats = checkpoint['std']
        best_thr = checkpoint['threshold']
        print(f"Модель загружена из {MODEL_PATH}")
        print(f"Порог: {best_thr:.3f}")

    # Применение ко всем территориям
    for territory in territories_to_predict:
        comp_dir = composite_base / territory
        if not comp_dir.exists():
            continue
        out_terr_dir = output_dir / territory
        out_terr_dir.mkdir(parents=True, exist_ok=True)

        comp_files = sorted(comp_dir.glob("*.tif"))
        print(f"\nКлассификация {territory} ({len(comp_files)} файлов)...")
        for comp_path in tqdm(comp_files, desc=territory):
            with rasterio.open(comp_path) as src:
                img = src.read().astype(np.float32)
                meta_out = src.meta.copy()
            prob = predict_image(model, img, mean_stats, std_stats, batch_size=50000, threshold=best_thr)
            mask_bin = postprocess_mask(prob, best_thr, min_area=100)
            obj_pixels = np.sum(mask_bin)
            print(f"    Обнаружено {obj_pixels} пикселей объектов ({(obj_pixels/mask_bin.size)*100:.2f}%)")
            out_path = out_terr_dir / f"{comp_path.stem}_pred.tif"
            meta_out.update({"count": 1, "dtype": np.uint8, "compress": "lzw"})
            with rasterio.open(out_path, "w", **meta_out) as dst:
                dst.write(mask_bin, 1)

            # Сохраняем карту вероятностей для первого файла (отладка)
            if territory == "Amga" and comp_path == comp_files[0]:
                prob_out = out_terr_dir / f"{comp_path.stem}_prob.tif"
                meta_prob = meta_out.copy()
                meta_prob.update({"dtype": np.float32, "count": 1})
                with rasterio.open(prob_out, "w", **meta_prob) as dst:
                    dst.write(prob, 1)
                print(f"      Карта вероятностей сохранена в {prob_out}")

        print(f"Готово для {territory}")

if __name__ == "__main__":
    main()