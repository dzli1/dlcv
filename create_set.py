import os
import random
import shutil
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = '/Users/louisyu/Downloads/wikiart'   # Where your big downloaded dataset is
DEST_DIR = './artset'          # The new folder you are creating
TARGET_COUNT = 900             # The number of images we want per class
MIN_CLASS_SIZE = 900          # Skip classes with fewer than this many images

def create_balanced_dataset():
    # 1. Create Destination Directory
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    # 2. Get list of class folders
    all_items = os.listdir(SOURCE_DIR)
    all_classes = []

    for item in all_items:
        full_path = os.path.join(SOURCE_DIR, item)
        # Check if it is a folder (ignore hidden files)
        if os.path.isdir(full_path) and not item.startswith('.'):
            all_classes.append(item)

    print(f"Found {len(all_classes)} total classes.")

    # 3. First Pass: Count images in each class and filter out classes with < MIN_CLASS_SIZE
    valid_classes = []

    for class_name in all_classes:
        class_path = os.path.join(SOURCE_DIR, class_name)

        # Count valid images
        valid_images_count = 0
        files_in_class = os.listdir(class_path)

        for filename in files_in_class:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                valid_images_count += 1

        # Only include classes with at least MIN_CLASS_SIZE images
        if valid_images_count >= MIN_CLASS_SIZE:
            valid_classes.append(class_name)
            print(f"  ✓ {class_name}: {valid_images_count} images (included)")
        else:
            print(f"  ✗ {class_name}: {valid_images_count} images (skipped, < {MIN_CLASS_SIZE})")

    print(f"\nProcessing {len(valid_classes)} classes with at least {MIN_CLASS_SIZE} images.")
    print(f"Targeting {TARGET_COUNT} randomly selected images per class.\n")

    # 4. Second Pass: Randomly select and move files
    for class_name in tqdm(valid_classes, desc="Processing Classes"):
        src_class_path = os.path.join(SOURCE_DIR, class_name)
        dest_class_path = os.path.join(DEST_DIR, class_name)

        # Create class folder in destination
        if not os.path.exists(dest_class_path):
            os.makedirs(dest_class_path)

        # Collect all valid images
        all_images = []
        files_in_class = os.listdir(src_class_path)

        for filename in files_in_class:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(filename)

        # Randomly shuffle to get a random selection
        random.shuffle(all_images)

        # Select TARGET_COUNT images (or all if fewer available)
        images_to_move = all_images[:TARGET_COUNT]

        # Move the selected images
        for image_name in images_to_move:
            src_file = os.path.join(src_class_path, image_name)
            dest_file = os.path.join(dest_class_path, image_name)
            shutil.move(src_file, dest_file)

    print(f"\nDone! Your balanced 'artset' is ready with {TARGET_COUNT} images per class.")

if __name__ == '__main__':
    create_balanced_dataset()
