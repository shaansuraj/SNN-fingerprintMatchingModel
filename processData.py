import os
import numpy as np
import imageio.v2 as imageio
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
from skimage import exposure
from skimage import filters

# Define directories
input_dir = "unprocessedThumbs"
output_dir = "processedThumbs"
label_dir = "labelArrays"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define data augmentation transformations
seq = iaa.Sequential([
    iaa.Flipud(0.5),                # Apply vertical flip with 50% probability
    iaa.Affine(rotate=(-45, 45)),   # Random rotation between -45 and 45 degrees
    iaa.Crop(percent=(0, 0.1)),     # Randomly crop between 0% to 10% of image
    iaa.GammaContrast((0.1, 3.0)),  # Adjust gamma and contrast
    iaa.GaussianBlur(sigma=(0.0, 3.0))  # Apply Gaussian blur with sigma between 0 and 3
])

# Preprocess images
processed_images = []

# Iterate over images in input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.BMP')):  # Check if file is an image
        # Read image
        image_path = os.path.join(input_dir, filename)
        image = imageio.imread(image_path)

        # Apply data augmentation
        augmented_images = seq(images=[image] * 5)  # Augment each image 5 times

        # Enhance contrast using histogram equalization
        equalized_images = [exposure.equalize_hist(img) for img in augmented_images]

        # Apply edge detection
        edge_detected_images = [filters.sobel(img) for img in equalized_images]

        # Convert images to uint8 and save
        for idx, img in enumerate(edge_detected_images):
            img = (img * 255).astype(np.uint8)  # Convert to uint8
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{idx}.png")
            imageio.imwrite(output_path, img)

            # Store processed image path for future use
            processed_images.append(output_path)

# Convert processed images to numpy array
processed_images_array = []
for img_path in processed_images:
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize image to desired dimensions
    img_array = np.array(img) / 255.0  # Normalize pixel values
    processed_images_array.append(img_array)

# Convert to numpy array
processed_images_array = np.array(processed_images_array)

# Save processed images array
np.save(os.path.join(label_dir, "processedThumbs.npy"), processed_images_array)

print("Preprocessing complete.")
