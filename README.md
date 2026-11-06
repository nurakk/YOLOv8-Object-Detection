# YOLOv8 Object Detection Using Python

## Experiment 8: Detect Common Objects in Images Using YOLO and Display Bounding Boxes

### 1. Project Overview

This project demonstrates object detection using the YOLOv8 (You Only Look Once) deep learning algorithm. The program detects common objects in an input image and displays bounding boxes, object labels, and confidence scores.

The experiment uses the YOLOv8 Nano model (`yolov8n.pt`) with Python in a Jupyter Notebook environment.

### 2. Objectives

- Understand the working principle of the YOLO object detection algorithm.
- Detect multiple objects in a single image.
- Identify object classes.
- Display bounding boxes around detected objects.
- Obtain confidence scores for each detection.

### 3. Algorithm Used

**YOLOv8 – You Only Look Once**

YOLO is a single-stage, deep-learning-based object detection algorithm. It processes an entire image in a single forward pass and predicts object locations and their corresponding classes.

### 4. Technologies and Libraries

- Programming Language: Python
- Deep Learning Model: YOLOv8 Nano
- Framework: Ultralytics
- Image Processing: OpenCV
- Visualization: Matplotlib
- Environment: Jupyter Notebook

### 5. Sample Input

The experiment uses the official Ultralytics sample image:

https://ultralytics.com/images/bus.jpg

The image contains a bus and people, which are used to demonstrate object detection.

### 6. Installation

Install the required libraries:

```bash
pip install ultralytics opencv-python matplotlib requests
```

### 7. How to Run

1. Clone or download this repository.
2. Open `Experiment_8_YOLO_Object_Detection.ipynb` in Jupyter Notebook.
3. Install the required libraries.
4. Run the notebook cells sequentially.
5. The program downloads the sample image and YOLOv8 Nano model weights if they are not already available.
6. View the detected objects with bounding boxes and confidence scores.

### 8. Expected Output

The program displays:

- Original input image.
- Detected objects with bounding boxes.
- Object class labels.
- Confidence scores for each detected object.

### 9. Project Files

- `Experiment_8_YOLO_Object_Detection.ipynb` – Complete Jupyter Notebook program.
- `Experiment_8_YOLO_Object_Detection.py` – Python source code.
- `bus.jpg` – Sample input image.
- `requirements.txt` – Required Python libraries.

### 10. Applications

- Autonomous mobile robots
- Industrial inspection
- Traffic monitoring
- Surveillance systems
- Robotics vision applications

### 11. Author

**Arun Kumar K**  
Department of Robotics and Automation

### 12. References

- Ultralytics Documentation: https://docs.ultralytics.com/
- YOLO GitHub Repository: https://github.com/ultralytics/ultralytics
