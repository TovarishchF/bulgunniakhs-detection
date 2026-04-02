import subprocess
import sys
import argparse
from pathlib import Path

# Конфигурация путей
BASE_DIR = Path(__file__).parent
COMPOSITES_DIR = BASE_DIR / "result" / "composites"
MASKS_DIR = BASE_DIR / "result" / "masks_rf"
MODEL_PATH = BASE_DIR / "Amga_rf_multisource.pkl"

# Функции запуска скриптов
def run_script(script_name, args=None):
    """Запускает Python скрипт с переданными аргументами."""
    cmd = [sys.executable, str(BASE_DIR / script_name)]
    if args:
        cmd.extend(args)
    print(f"\n>>> Запуск {script_name} {' '.join(args) if args else ''}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"Ошибка при выполнении {script_name}, код {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Pipeline обработки данных")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Пропустить создание композитов")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Пропустить классификацию (маски уже существуют)")
    parser.add_argument("--skip-aggregation", action="store_true",
                        help="Пропустить агрегацию масок")
    parser.add_argument("--skip-visualization", action="store_true",
                        help="Пропустить визуализацию")
    parser.add_argument("--skip-cadastre", action="store_true",
                        help="Пропустить обработку кадастровых данных (CSV->GeoJSON)")
    parser.add_argument("--train-model", action="store_true",
                        help="Переобучить модель Random Forest (влияет на шаг классификации)")
    args = parser.parse_args()

    # 1. Кадастровые данные (необязательно)
    if not args.skip_cadastre:
        if (BASE_DIR / "function.py").exists():
            run_script("function.py")
        else:
            print("Файл function.py не найден, пропускаем обработку кадастровых данных")

    # 2. Создание композитов
    if not args.skip_preprocessing:
        if (BASE_DIR / "preprocessing.py").exists():
            run_script("preprocessing.py")
        else:
            print("Ошибка: preprocessing.py не найден")
            sys.exit(1)
    else:
        print("Пропускаем создание композитов (--skip-preprocessing)")

    # 3. Классификация
    if not args.skip_classification:
        # Проверяем, нужно ли переобучать модель
        classification_args = []
        if args.train_model:
            classification_args.append("--train")
        # Если модель уже есть и не требуется переобучение, запускаем без --train
        run_script("classification_model.py", classification_args)
    else:
        print("Пропускаем классификацию (--skip-classification)")

    # 4. Агрегация масок
    if not args.skip_aggregation:
        if (BASE_DIR / "mask_aggregation.py").exists():
            run_script("mask_aggregation.py")
        else:
            print("Ошибка: mask_aggregation.py не найден")
            sys.exit(1)
    else:
        print("Пропускаем агрегацию масок (--skip-aggregation)")

    # 5. Визуализация
    if not args.skip_visualization:
        if (BASE_DIR / "vizualize_results.py").exists():
            run_script("vizualize_results.py")
        else:
            print("Ошибка: vizualize_results.py не найден")
            sys.exit(1)
    else:
        print("Пропускаем визуализацию (--skip-visualization)")

    print("\nОбработка завершена!")

if __name__ == "__main__":
    main()