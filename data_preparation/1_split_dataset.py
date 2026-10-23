import os
import shutil
import random


def split_dataset(src_dir, dest_dir, train_ratio=0.70, val_ratio=0.15):
    print("Starting dataset split...")
    print("-" * 50)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    for class_name in sorted(os.listdir(src_dir)):
        class_path = os.path.join(src_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        total_files = len(files)

        print("Processing class: {} ({} files)".format(class_name, total_files))

        # Перемішуємо файли перед розбиттям
        random.shuffle(files)

        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)

        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        def copy_files(file_list, split_name):
            split_dir = os.path.join(dest_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for f in file_list:
                src_file = os.path.join(class_path, f)
                dest_file = os.path.join(split_dir, f)
                shutil.copy2(src_file, dest_file)

        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')

    print("-" * 50)
    print("Done! Dataset successfully split into train, val, and test.")
    print("Destination: {}".format(dest_dir))


if __name__ == "__main__":
    SOURCE_DATASET = r"D:\projects\Diploma\merged_dataset_filtered"
    DESTINATION_DATASET = r"D:\projects\Diploma\transfer_learning_datasets_100_train"

    # Фіксуємо random seed для стабільного розбиття
    random.seed(42)

    split_dataset(SOURCE_DATASET, DESTINATION_DATASET)