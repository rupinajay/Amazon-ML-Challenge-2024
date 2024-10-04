import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import pandas as pd
from src.utils import download_images, extract_image_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src import constants
import multiprocessing
from pathlib import Path

# Step 1: Load the ResNet50 model for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract image features with error handling
def extract_image_features(image_path):
    """Extract features from an image using ResNet50."""
    try:
        print(f"Processing image: {image_path}")
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = resnet_model.predict(x)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Unit mappings for prediction formatting
unit_mapping = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {
        'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'
    },
    'maximum_weight_recommendation': {
        'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'
    },
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {
        'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 
        'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'
    }
}

# Main function to run the code
def main():
    # Step 2: Load the datasets and create a subset of 10 images
    train_image_folder = '/path/to/train_images'
    test_image_folder = '/path/to/test_images'
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)

    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')

    # Select a subset of 10 images from both train and test datasets
    train_data_subset = train_data.sample(n=10, random_state=42)
    test_data_subset = test_data.sample(n=10, random_state=42)

    print("Testing with 10 images from train and test datasets.")

    # Step 3: Check for missing images and download if necessary
    def check_missing_images(image_links, image_folder):
        missing_images = [img for img in image_links if not os.path.exists(os.path.join(image_folder, Path(img).name))]
        return missing_images

    # For train data (subset)
    missing_train_images = check_missing_images(train_data_subset['image_link'], train_image_folder)
    if missing_train_images:
        print(f"Downloading {len(missing_train_images)} missing train images...")
        download_images(missing_train_images, train_image_folder)
    else:
        print("All train images already downloaded.")

    # For test data (subset)
    missing_test_images = check_missing_images(test_data_subset['image_link'], test_image_folder)
    if missing_test_images:
        print(f"Downloading {len(missing_test_images)} missing test images...")
        download_images(missing_test_images, test_image_folder)
    else:
        print("All test images already downloaded.")

    # Step 4: Extract image features for the subset of training data
    print("Extracting features from training images...")
    train_features = []
    valid_train_images = []  # To keep track of valid images

    for img_file in os.listdir(train_image_folder):
        img_path = os.path.join(train_image_folder, img_file)
        features = extract_image_features(img_path)
        if features is not None:  # Only append features if extraction was successful
            train_features.append(features)
            valid_train_images.append(img_file)

    train_features = np.array(train_features)
    print(f"Extracted features from {len(train_features)} training images.")

    # Filter train_data to only include valid images
    train_data_filtered = train_data_subset[train_data_subset['image_link'].apply(lambda x: Path(x).name).isin(valid_train_images)]
    print(f"Filtered training data: {len(train_data_filtered)} rows.")

    y_train = train_data_filtered['entity_value'].values  # Target values
    group_id = train_data_filtered['group_id'].values.reshape(-1, 1)
    entity_name = train_data_filtered['entity_name'].values.reshape(-1, 1)

    # Combine image features with group_id and entity_name for training
    X_train = np.concatenate([train_features, group_id, entity_name], axis=1)
    print(f"Training data shape: {X_train.shape}")

    # Step 5: Train the model using RandomForestRegressor
    print("Training the model...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train,
                                                                  y_train,
                                                                  test_size=0.2,
                                                                  random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)

    val_predictions = model.predict(X_val)
    print(f"Validation predictions (first 5): {val_predictions[:5]}")

    # Step 6: Extract image features for test data subset
    print("Extracting features from test images...")
    test_features = []
    valid_test_images = []

    for img_file in os.listdir(test_image_folder):
        img_path = os.path.join(test_image_folder, img_file)
        features = extract_image_features(img_path)
        if features is not None:
            test_features.append(features)
            valid_test_images.append(img_file)

    test_features = np.array(test_features)
    print(f"Extracted features from {len(test_features)} test images.")

    # Step 7: Predict entity values for test data
    print("Making predictions on test data...")
    test_predictions = model.predict(test_features)

    # Step 8: Format the predictions
    def format_prediction(value, unit):
        return f"{value:.2f} {unit}"

    # Choose appropriate unit based on entity_name
    formatted_predictions = []
    for idx, pred in enumerate(test_predictions):
        entity_name = test_data_subset['entity_name'].iloc[idx]
        possible_units = unit_mapping.get(entity_name, {'gram'})  # Default to 'gram'
        unit = possible_units.pop()  # Choose any unit (can be refined)
        formatted_predictions.append(format_prediction(pred, unit))

    print(f"Formatted predictions (first 5): {formatted_predictions[:5]}")

    # Step 9: Save predictions to CSV for submission
    output_df = pd.DataFrame({
        'index': test_data_subset['index'],
        'prediction': formatted_predictions
    })

    output_df.to_csv('test_out_10_images.csv', index=False)
    print("Output saved to test_out_10_images.csv")

# Ensure multiprocessing works properly
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Fix for macOS multiprocessing
    main()
