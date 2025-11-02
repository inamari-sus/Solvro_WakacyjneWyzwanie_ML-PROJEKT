import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tiffslide
from PIL import Image

# --- KONFIGURACJA GŁÓWNA ---
healthy_folder_path = './data/healthy'
sick_folder_path = './data/sick'
base_output_dir = './data/patches1'

PATCH_SIZE = 256
TARGET_MAGNIFICATION = 20
TISSUE_THRESHOLD = 25


def get_ndpi_files_from_local_folder(folder_path):
    ndpi_files = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.ndpi', '.svs')):
                ndpi_files.append(os.path.join(folder_path, filename))
    return ndpi_files


def extract_patches_from_local_file(file_path, output_dir, patch_size, target_magnification, tissue_threshold):
    file_name = os.path.basename(file_path)
    patient_folder = os.path.join(output_dir, os.path.splitext(file_name)[0])
    os.makedirs(patient_folder, exist_ok=True)

    try:
        slide = tiffslide.TiffSlide(file_path)
    except Exception as e:
        print(f"Błąd przy otwieraniu pliku WSI: {file_name}. Błąd: {e}")
        return

    try:
        native_mag_str = slide.properties.get(tiffslide.PROPERTY_NAME_OBJECTIVE_POWER, '40')
        native_magnification = float(native_mag_str)
        available_mags = [native_magnification / downsample for downsample in slide.level_downsamples]
        mag_diffs = [abs(mag - target_magnification) for mag in available_mags]
        level = mag_diffs.index(min(mag_diffs))
        print(f"Wybrano poziom {level} (powiększenie ~{available_mags[level]:.2f}x)")
    except Exception as e:
        print(f"Nie udało się ustalić powiększenia, używam poziomu 0. Błąd: {e}")
        level = 0

    try:
        thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
        thumbnail_hsv = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2HSV)
        saturation_channel = thumbnail_hsv[:, :, 1]
        _, tissue_mask = cv2.threshold(saturation_channel, tissue_threshold, 255, cv2.THRESH_BINARY)
    except Exception as e:
        print(f"Nie udało się stworzyć maski tkanki: {e}")
        return

    target_dims = slide.level_dimensions[level]
    mask_scaled = cv2.resize(tissue_mask, (target_dims[0], target_dims[1]), interpolation=cv2.INTER_NEAREST)

    patch_count = 0
    downsample_factor = slide.level_downsamples[level]

    with tqdm(total=(target_dims[1] // patch_size) * (target_dims[0] // patch_size)) as pbar:
        for y in range(0, target_dims[1], patch_size):
            for x in range(0, target_dims[0], patch_size):
                pbar.update(1)
                if x + patch_size <= target_dims[0] and y + patch_size <= target_dims[1]:
                    patch_center_x, patch_center_y = x + patch_size // 2, y + patch_size // 2
                    if mask_scaled[patch_center_y, patch_center_x] > 0:
                        level_0_x = int(x * downsample_factor)
                        level_0_y = int(y * downsample_factor)
                        patch = slide.read_region((level_0_x, level_0_y), level, (patch_size, patch_size))
                        patch_rgb = patch.convert("RGB")
                        patch_filename = os.path.join(patient_folder, f"patch_x{level_0_x}_y{level_0_y}.png")
                        patch_rgb.save(patch_filename)
                        patch_count += 1

    slide.close()
    print(f"Zakończono. Zapisano {patch_count} kafelków.")


def find_processed_files(base_output_dir):
    processed_files = set()
    for root, dirs, files in os.walk(base_output_dir):
        if os.path.basename(root) in ['healthy', 'sick']:
            processed_files.update(dirs)
    return processed_files


# --- GŁÓWNA LOGIKA SKRYPTU ---
def main():
    all_healthy_ndpi_files = get_ndpi_files_from_local_folder(healthy_folder_path)
    all_sick_ndpi_files = get_ndpi_files_from_local_folder(sick_folder_path)

    print(f"Znaleziono {len(all_healthy_ndpi_files)} plików .ndpi w folderze zdrowych.")
    print(f"Znaleziono {len(all_sick_ndpi_files)} plików .ndpi w folderze chorych.")

    processed_files_basenames = find_processed_files(base_output_dir)

    processed_healthy_files = [f for f in all_healthy_ndpi_files if
                               os.path.splitext(os.path.basename(f))[0] in processed_files_basenames]
    unprocessed_healthy_files = [f for f in all_healthy_ndpi_files if
                                 os.path.splitext(os.path.basename(f))[0] not in processed_files_basenames]

    processed_sick_files = [f for f in all_sick_ndpi_files if
                            os.path.splitext(os.path.basename(f))[0] in processed_files_basenames]
    unprocessed_sick_files = [f for f in all_sick_ndpi_files if
                              os.path.splitext(os.path.basename(f))[0] not in processed_files_basenames]

    print(
        f"Przetworzono już {len(processed_healthy_files)} plików healthy. Pozostało {len(unprocessed_healthy_files)} do przetworzenia.")
    print(
        f"Przetworzono już {len(processed_sick_files)} plików sick. Pozostało {len(unprocessed_sick_files)} do przetworzenia.")


    if len(unprocessed_healthy_files) > 0:
        train_healthy_new, test_healthy_new = train_test_split(unprocessed_healthy_files, test_size=0.2,
                                                               random_state=42)
        train_healthy_new, val_healthy_new = train_test_split(train_healthy_new, test_size=0.125, random_state=42)
    else:
        train_healthy_new, val_healthy_new, test_healthy_new = [], [], []

    if len(unprocessed_sick_files) > 0:
        train_sick_new, test_sick_new = train_test_split(unprocessed_sick_files, test_size=0.2, random_state=42)
        train_sick_new, val_sick_new = train_test_split(train_sick_new, test_size=0.125, random_state=42)
    else:
        train_sick_new, val_sick_new, test_sick_new = [], [], []


    train_healthy = train_healthy_new
    val_healthy = val_healthy_new
    test_healthy = test_healthy_new

    train_sick = train_sick_new
    val_sick = val_sick_new
    test_sick = test_sick_new


    os.makedirs(base_output_dir, exist_ok=True)
    datasets = {
        'train': {'healthy': train_healthy, 'sick': train_sick},
        'val': {'healthy': val_healthy, 'sick': val_sick},
        'test': {'healthy': test_healthy, 'sick': test_sick}
    }


    for split, categories in datasets.items():
        for category, files in categories.items():
            output_split_dir = os.path.join(base_output_dir, split, category)
            os.makedirs(output_split_dir, exist_ok=True)
            if files:
                print(f"\nPrzetwarzanie {split} - {category} ({len(files)} pacjentów)...")
                for file in files:
                    extract_patches_from_local_file(file, output_split_dir, PATCH_SIZE, TARGET_MAGNIFICATION,
                                                    TISSUE_THRESHOLD)

    print("\nWszystkie dane zostały przetworzone i zapisane w folderach.")


if __name__ == '__main__':
    main()