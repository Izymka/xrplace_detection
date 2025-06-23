from pathlib import Path
from collections import Counter
from PIL import Image
import glob
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os


def eda_df(path):
    base_path = Path(path)
    # Директории
    train_img_dir = base_path / "train" / "images"
    train_lbl_dir = base_path / "train" / "labels"
    yaml_path = base_path / "data.yaml" 

    # список изображений и аннотаций
    image_files = (
        glob.glob(str(train_img_dir / "*.jpg")) + 
        glob.glob(str(train_img_dir / "*.JPG")) +
        glob.glob(str(train_img_dir / "*.png")) +
        glob.glob(str(train_img_dir / "*.PNG"))
    )

    label_files = glob.glob(str(train_lbl_dir / "*.txt"))

    print(f"Кол-во изображений: {len(image_files)}")
    print(f"Кол-во файлов аннотаций: {len(label_files)}")
    print("---")

    # Размеры изображений
    image_sizes = []
    for img_path in image_files:
        with Image.open(img_path) as img:
            image_sizes.append(img.size)  # (width, height)

    df_sizes = pd.DataFrame(image_sizes, columns=["width", "height"])
    print(df_sizes.describe())

    # Распределение по классам
    class_counts = Counter()
    for lbl_file in label_files:
        with open(lbl_file, "r") as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1
    print("---\n\nРаспределение объектов по классам:\n")
    for class_id, count in sorted(class_counts.items()):
        print(f"Класс {class_id}: {count} объектов")

    # Загрузка классов из data.yaml
    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg.get("names", {})  
    print("---")
    print("Классы:", class_names)

    return [train_img_dir, train_lbl_dir, class_names],   # возвращаем Path-объект

def show_image_labels(train_img_dir, train_lbl_dir, class_names, img_index):
    # ==== Выбор изображения ====
    img_path = sorted(list(train_img_dir.glob("*.jpg")) 
                      + list(train_img_dir.glob("*.png")))[img_index]
    img_name = img_path.stem
    label_path = train_lbl_dir / f"{img_name}.txt"

    # ==== Загрузка изображения ====
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w_img, h_img = image.size

    # ==== Чтение и отрисовка аннотаций ====
    with open(label_path, "r") as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Преобразуем YOLO координаты в абсолютные
            x0 = (x_center - width / 2) * w_img
            y0 = (y_center - height / 2) * h_img
            x1 = (x_center + width / 2) * w_img
            y1 = (y_center + height / 2) * h_img

            class_name = class_names[int(class_id)] if int(class_id) in class_names else str(int(class_id))

            # Отрисовка прямоугольника и подписи
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            draw.text((x0, y0 - 10), class_name, fill="red")
        # ==== Отображение результата ====
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Аннотации для: {img_name}")
        plt.show()



def plot_box_distributions(labels_dir):
    """
    Строит гистограммы распределения ширины, высоты и площади боксов
    для каждого класса по YOLO-аннотациям из .txt файлов.

    :param labels_dir: Путь к папке с аннотациями в YOLO-формате.
    """
    box_stats = {}

    # Чтение всех файлов
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(labels_dir, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, _, _, w, h = parts
                    class_id = int(class_id)
                    w = float(w)
                    h = float(h)
                    area = w * h

                    if class_id not in box_stats:
                        box_stats[class_id] = {'widths': [], 
                                               'heights': [], 
                                               'areas': [],
                                               'filename':[]}

                    box_stats[class_id]['widths'].append(w)
                    box_stats[class_id]['heights'].append(h)
                    box_stats[class_id]['areas'].append(area)
                    box_stats[class_id]['filename'].append(filename)

    if not box_stats:
        print("Нет данных для построения гистограмм.")
        return

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins = 50

    for class_id, stats in box_stats.items():
        axes[0].hist(stats['widths'], bins=bins, alpha=0.5, label=f'Класс {class_id}')
        axes[1].hist(stats['heights'], bins=bins, alpha=0.5, label=f'Класс {class_id}')
        axes[2].hist(stats['areas'], bins=bins, alpha=0.5, label=f'Класс {class_id}')

    # Подписи и оформление
    axes[0].set_title("Распределение ширины")
    axes[0].set_xlabel("Ширина")
    axes[0].set_ylabel("Количество")

    axes[1].set_title("Распределение высоты")
    axes[1].set_xlabel("Высота")

    axes[2].set_title("Распределение площади")
    axes[2].set_xlabel("Площадь")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
