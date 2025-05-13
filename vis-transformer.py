import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

if len(sys.argv) < 2:
    raise ValueError("Usage: python vis-transformer.py <image_path>")

image_path = sys.argv[1]
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at: {image_path}")

model = YOLO("best.pt")

img = cv2.imread(image_path)
if img is None:
    raise ValueError("Failed to load image. Check the file path and format.")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model(img, imgsz=1280)

class_indices = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else np.array([])
confidence_scores = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else np.array([])

if class_indices.size > 0:
    class_labels = [model.names[int(idx)] for idx in class_indices]
else:
    class_labels = []

# Print detected classes and confidence scores
print("All Detected Objects:", list(zip(class_labels, confidence_scores)))

vegetable_labels = {
    "broccoli", "carrot", "cauliflower", "cucumber", "lettuce",
    "onion", "bell pepper", "potato", "spinach", "tomato", "zucchini",
    "green pepper", "red pepper", "jalapeno", "eggplant", "radish"
}

for label, score in zip(class_labels, confidence_scores):
    print(f"Detected: {label} (Confidence: {score:.2f})")

threshold = 0.0001
identified_vegetables = [
    label for label, score in zip(class_labels, confidence_scores)
]

print("Identified Vegetables:", identified_vegetables)
