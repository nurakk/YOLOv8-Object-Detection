#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# EXPERIMENT 8: OBJECT DETECTION USING YOLOv8
# Algorithm: YOLOv8 Nano
# ============================================================

# Step 1: Install required libraries
get_ipython().run_line_magic('pip', 'install ultralytics opencv-python matplotlib requests')

# Step 2: Import libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import requests

# Step 3: Download sample input image
url = "https://ultralytics.com/images/bus.jpg"

response = requests.get(url)
response.raise_for_status()

with open("bus.jpg", "wb") as file:
    file.write(response.content)

print("Sample image downloaded successfully!")

# Step 4: Display original input image
image = cv2.imread("bus.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 7))
plt.imshow(image)
plt.title("Original Input Image")
plt.axis("off")
plt.show()

# Step 5: Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

print("YOLOv8 model loaded successfully!")

# Step 6: Perform object detection
results = model.predict(
    source="bus.jpg",
    conf=0.25
)

print("Object detection completed!")

# Step 7: Display detection results
result_image = results[0].plot()
result_image = cv2.cvtColor(
    result_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.title("YOLOv8 Object Detection Result")
plt.axis("off")
plt.show()

# Step 8: Display detected object names and confidence
print("\nDETECTED OBJECTS")
print("--------------------------")

for box in results[0].boxes:

    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])

    print(
        "Object:", class_name,
        "| Confidence:", round(confidence, 3)
    )

print("\nExperiment completed successfully!")


# In[ ]:




