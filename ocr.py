import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import pytesseract
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load test data
def get_test_data(file_dir, dataset_name):
    file_name = f"{dataset_name}.csv"
    file_path = os.path.join(file_dir, file_name)
    return pd.read_csv(file_path)

# Function to extract and clean image text
def extract_and_clean_image_text(image_url):
    # Fetch the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Extract text using pytesseract
    text = pytesseract.image_to_string(img)
    
    # Clean text
    text = re.sub(r'[^\w\s\.\-,:\n]', '', text)  # Remove unwanted symbols
    text = re.sub(r'\n+', '\n', text).strip()  # Remove extra newlines
    return text

# Extract key-value pairs
def extract_key_value_pairs(cleaned_text):
    pattern = re.compile(r'([A-Za-z\s]+)[:\-]\s*(.*)')
    key_value_pairs = {}
    
    # Split text into lines and search for key-value pairs
    for line in cleaned_text.split('\n'):
        match = pattern.match(line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            key_value_pairs[key] = value
    
    return key_value_pairs

# Generate features using TF-IDF
def generate_features(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

# Apply the same logic to predict for the test dataset
def process_test_data(test_df):
    processed_texts = []
    
    for idx, row in test_df.iterrows():
        image_url = row['image_link']  # Assuming 'image_link' contains the URL
        print(f"Processing Image {idx+1}/{len(test_df)}: {image_url}")
        
        # Extract and clean image text
        text = extract_and_clean_image_text(image_url)
        
        # Extract key-value pairs (optional, if needed)
        key_value_pairs = extract_key_value_pairs(text)
        
        # Convert key-value pairs into a format you need
        formatted_string = "\n".join([f"{key}: {value}" for key, value in key_value_pairs.items()])
        
        processed_texts.append(formatted_string)
    
    return processed_texts

# Main prediction workflow
def predict(file_dir, test_file_name):
    # Load test dataset
    test_df = get_test_data(file_dir, test_file_name)
    
    # Process the test data
    processed_texts = process_test_data(test_df)
    
    # Generate TF-IDF features
    features = generate_features(processed_texts)
    
    # You can now apply these features to your model for prediction (e.g., using a trained classifier)
    # Example: model.predict(features)

# Example usage
file_dir = r'/Users/rupinajay/Developer/Amazon ML Challenge 24/dataset'
predict(file_dir, 'sample_test')

