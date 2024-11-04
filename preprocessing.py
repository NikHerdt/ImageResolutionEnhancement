import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import pandas as pd
import os

# Function to read in image and metadata
def read_image(image_path):
    image = sitk.ReadImage(image_path)
    
    metadata = {}
    for key in image.GetMetaDataKeys():
        metadata[key] = image.GetMetaData(key)

    image = sitk.GetArrayFromImage(image)[0]
    return image, metadata

# Function to preprocess image
def preprocess_image(image):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Preprocessing: enhance contrast and smooth
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Threshold to create a binary image
    _, binary_img = cv2.threshold(blurred_img, 10, 255, cv2.THRESH_BINARY)

    # Detect contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours: keep only large regions and exclude rectangular text/overlays
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Conditions to filter out text or rectangular overlays:
        # - Exclude small areas that are likely text
        # - Keep contours that are large or have irregular shapes
        if cv2.contourArea(contour) > 100750 and (0.5 < aspect_ratio < 5.0):  # Adjust these thresholds as needed
            filtered_contours.append(contour)

    # Create a mask to isolate only the main ultrasound shape
    ultrasound_mask = np.zeros_like(binary_img)
    cv2.drawContours(ultrasound_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    # Apply the mask to keep only the filtered regions
    isolated_ultrasound = cv2.bitwise_and(gray_img, gray_img, mask=ultrasound_mask)

    return isolated_ultrasound

# Function to add noise to the isolated ultrasound image
def add_noise(image):
    row, col = image.shape
    noise = np.random.rayleigh(1, (row, col)).astype(np.uint8)
    noisy_image = image + image * noise
    return noisy_image

# Function to preprocess and add noise to all images in the folders specified by metadata.csv
def process_all_images(metadata_csv):
    for index, row in metadata_csv.iterrows():
        folder_path = row['File Location']
        for filename in os.listdir(folder_path):
            if filename.endswith('.dcm'):
                image_path = os.path.join(folder_path, filename)
                image, _ = read_image(image_path)
                preprocessed_image = preprocess_image(image)
                noisy_image = add_noise(preprocessed_image)
                
                # Save the noisy image to Noisy Images folder
                save_path = os.path.join('Noisy Images', filename)
                cv2.imwrite(save_path, noisy_image)

# Read in metadata.csv to get path to folders containing images
metadata_csv = pd.read_csv('metadata.csv')
process_all_images(metadata_csv)