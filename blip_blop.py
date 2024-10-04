from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from sentence_transformers import SentenceTransformer, util

"""Load the BLIP model and processor"""

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

"""Load sentence transformers"""

text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

"""Generating a caption for a local image"""

def generate_caption(image_path):
    # Open the image
    image = Image.open(image_path)

    # Process the image
    inputs = processor(image, return_tensors="pt")

    # Generate the caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

def generate_caption_from_url(image_url):
    # Load image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Reuse the local image captioning function
    return generate_caption(image)

"""Checking the caption against the text provided by the user"""

def check_complaint_match(complaint_text, image_path):
    # Generate a caption for the image
    image_caption = generate_caption(image_path)

    # Calculate similarity between complaint and image caption
    complaint_embedding = text_model.encode(complaint_text, convert_to_tensor=True)
    caption_embedding = text_model.encode(image_caption, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(complaint_embedding, caption_embedding).item()

    # Return results
    return image_caption, similarity_score

def caption_from_local_image(image_path):
    return generate_caption(image_path)

# Test the function with a sample local image
image_path = '/content/img_1.jpg'  # Replace with your local image path
caption = caption_from_local_image(image_path)
print(f"Generated Caption: {caption}")

complaint_text = "this is the state of the sink on the train"

# Check if the complaint matches the image
image_caption, similarity = check_complaint_match(complaint_text, image_path)
print(f"Complaint: {complaint_text}")
print(f"Generated Caption: {image_caption}")
print(f"Similarity Score: {similarity}")

"""**Video Frame Analysis**

Import necessary libraries
"""

import cv2
from PIL import Image
import torch

"""Extract videos from video"""

import cv2
from PIL import Image

def extract_frame_from_video(video_path, start_second=2, interval=5):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate the frame number to start extracting from the start_second
    frame_number = int(fps * start_second)
    frames = []

    while True:
        ret, frame = video.read()  # Read the next frame

        if not ret:
            break  # Exit the loop when the video ends

        # Process frame at 2nd second, 7th second, 12th second, and so on
        if video.get(cv2.CAP_PROP_POS_FRAMES) == frame_number:
            # Convert frame (which is in BGR format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

            # Update to extract the next frame at (current second + interval)
            frame_number += int(fps * interval)

    video.release()
    return frames

"""Caption from frame"""

def caption_from_frame(frame):
    # Process the image and generate a caption
    inputs = processor(frame, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

"""Test with a local video"""

# Test with a local video
video_path = '/content/stock video.mp4'  # Replace with your video path
frames = extract_frame_from_video(video_path, interval=5)

# Process each extracted frame
for i, frame in enumerate(frames):
    caption = caption_from_frame(frame)
    print(f"Generated Caption for frame {i+1}: {caption}")