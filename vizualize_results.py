import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib

PLOT_DIR = Path("result/plots")
STATS_DIR = Path("result/stats")
MASK_DIR = Path("result/masks_rf")
MODEL_PATH = Path("Amga_rf_multisource.pkl")
VALIDATION_DATA_DIR = Path("validation_data")

PLOT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Предполагается, что композит имеет 8 каналов: B, G, R, NIR, NDWI, NDVI, Slope, Aspect
FEATURE_NAMES = [
    "Blue (B)",
    "Green (G)",
    "Red (R)",
    "NIR",
    "NDWI",
    "NDVI",
    "Slope",
    "Aspect"
]

def load_masks_and_compute_stats(territory):
    """Загружает маски, считает площадь объектов, сохраняет CSV."""
    mask_dir = MASK_DIR / territory
    mask_files = sorted(mask_dir.glob("*_pred.tif"))
    if not mask_files:
        raise FileNotFoundError(f"Нет масок в {mask_dir}")

    stats = []
    for f in tqdm(mask_files, desc=f"Processing {territory}"):
        with rasterio.open(f) as src:
            mask = src.read(1).astype(np.uint8)
            transform = src.transform
            pixel_area = abs(transform.a * transform.e)
            obj_pixels = np.sum(mask == 1)
            obj_area_ha = obj_pixels * pixel_area / 10000
            date_str = f.stem.replace("_pred", "")
            stats.append({
                "file": f.name,
                "date": date_str,
                "obj_pixels": obj_pixels,
                "obj_area_ha": obj_area_ha,
                "total_pixels": mask.size,
                "fraction": obj_pixels / mask.size if mask.size > 0 else 0
            })
    df = pd.DataFrame(stats)
    df.to_csv(STATS_DIR / f"{territory}_stats.csv", index=False)
    return df

def plot_temporal_trend(df, territory):
    """График динамики площади объектов по датам."""
    df['date'] = pd.to_datetime(df['date'], format='%d%m%Y')
    df = df.sort_values('date')
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['obj_area_ha'], marker='o', linestyle='-')
    plt.xlabel('Дата')
    plt.ylabel('Площадь объектов (га)')
    plt.title(f'{territory} – Динамика деградации')
    plt.grid(True)
    out_path = PLOT_DIR / f"{territory}_temporal_trend.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"График сохранён: {out_path}")

def plot_feature_importance(model, territory):
    """График важности признаков с человеко-читаемыми названиями."""
    importances = model.feature_importances_
    n_original = len(FEATURE_NAMES)
    n_features = len(importances)
    # Если признаков больше, чем исходных (добавлены текстуры), то создаём имена для текстур
    if n_features > n_original:
        texture_names = [f"Texture_{name}" for name in FEATURE_NAMES]
        all_names = FEATURE_NAMES + texture_names
    else:
        all_names = FEATURE_NAMES[:n_features]
    idx = np.argsort(importances)[::-1]
    sorted_names = [all_names[i] for i in idx]
    sorted_importances = importances[idx]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importances)), sorted_importances)
    plt.yticks(range(len(importances)), sorted_names)
    plt.xlabel('Важность')
    plt.title(f'{territory} – Важность признаков')
    plt.tight_layout()
    out_path = PLOT_DIR / f"{territory}_feature_importance.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"График важности сохранён: {out_path}")

def plot_confusion_matrix_and_metrics(y_true, y_pred, territory):
    """Матрица ошибок и метрики PA, UA, F1."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    pa = tp / (tp + fn) if (tp+fn) > 0 else 0  # Producer's Accuracy (Recall)
    ua = tp / (tp + fp) if (tp+fp) > 0 else 0  # User's Accuracy (Precision)
    f1 = 2 * (pa * ua) / (pa + ua) if (pa+ua) > 0 else 0
    print(f"\n=== Метрики для {territory} ===")
    print(f"Producer's Accuracy (Recall): {pa:.4f}")
    print(f"User's Accuracy (Precision): {ua:.4f}")
    print(f"F1-score: {f1:.4f}")
    # Сохраняем метрики в CSV
    metrics_df = pd.DataFrame([{"PA": pa, "UA": ua, "F1": f1}])
    metrics_df.to_csv(STATS_DIR / f"{territory}_metrics.csv", index=False)
    # Тепловая карта
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['фон', 'объект'], yticklabels=['фон', 'объект'])
    plt.ylabel('Истина')
    plt.xlabel('Предсказание')
    plt.title(f'{territory} – Матрица ошибок')
    out_path = PLOT_DIR / f"{territory}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Матрица ошибок сохранена: {out_path}")
    return pa, ua, f1

def plot_roc_curve(y_true, y_prob, territory):
    """ROC-кривая с AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{territory} – ROC кривая')
    plt.legend()
    out_path = PLOT_DIR / f"{territory}_roc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"ROC кривая сохранена: {out_path}")

def plot_training_curve(model, oob_scores=None): # not usable, TL;DR
    """
    Отображает OOB score. Если передан список oob_scores по числу деревьев,
    строит кривую обучения.
    """
    if hasattr(model, 'oob_score_'):
        print(f"OOB Score (final): {model.oob_score_:.4f}")
        if oob_scores is not None and len(oob_scores) > 0:
            plt.figure(figsize=(10,5))
            plt.plot(range(1, len(oob_scores)+1), oob_scores, marker='o')
            plt.xlabel('Number of trees')
            plt.ylabel('OOB Score')
            plt.title('Random Forest OOB Score vs Number of Trees')
            plt.grid(True)
            out_path = PLOT_DIR / "oob_curve.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"График OOB сохранён: {out_path}")
        else:
            print("Для построения кривой OOB необходимо передать список oob_scores (например, из сохранённой истории обучения).")
    else:
        print("Модель не имеет OOB score. Убедитесь, что в конфиге RandomForestClassifier oob_score=True.")

def main():
    if MODEL_PATH.exists():
        checkpoint = joblib.load(MODEL_PATH)
        model = checkpoint['model']
        plot_feature_importance(model, "Amga")
        plot_training_curve(model)
    else:
        print(f"Модель не найдена: {MODEL_PATH}, пропускаем важность признаков и график обучения.")

    y_val_path = VALIDATION_DATA_DIR / "y_val.npy"
    y_pred_path = VALIDATION_DATA_DIR / "y_pred.npy"
    y_prob_path = VALIDATION_DATA_DIR / "y_prob.npy"
    if y_val_path.exists() and y_pred_path.exists():
        y_val = np.load(y_val_path)
        y_pred = np.load(y_pred_path)
        plot_confusion_matrix_and_metrics(y_val, y_pred, "Amga")
        if y_prob_path.exists():
            y_prob = np.load(y_prob_path)
            plot_roc_curve(y_val, y_prob, "Amga")
    else:
        print("Валидационные данные не найдены. Пропускаем матрицу ошибок и ROC.")

    for territory in ["Amga", "Yunkor"]:
        try:
            df = load_masks_and_compute_stats(territory)
            if not df.empty:
                plot_temporal_trend(df, territory)
            print(f"Статистика для {territory} сохранена в {STATS_DIR / f'{territory}_stats.csv'}")
        except Exception as e:
            print(f"Ошибка для {territory}: {e}")

if __name__ == "__main__":
    main()