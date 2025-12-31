import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = r"C:\\Users\\abira\\Downloads\\DeepFake-Detector\\real_and_fake_face" #update this path
output_dir = r"C:\\Users\\abira\\Downloads\\DeepFake-Detector\\Dataset" #update this path

# Create output folders
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Categories
categories = ["training_real", "training_fake"]

# Split data into 80% train and 20% test
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    images = os.listdir(category_path)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # Create subfolders
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

    # Move training images
    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))

    # Move testing images
    for img in test_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(test_dir, category, img))

print("Dataset organized successfully into 'train' and 'test' folders!")

