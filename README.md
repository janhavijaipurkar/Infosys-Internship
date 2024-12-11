Smart Safety Gear Detection

This project is designed to enhance workplace safety by detecting whether individuals are wearing helmets using YOLO (You Only Look Once) object detection. By identifying non-compliance in real time, it aims to reduce workplace accidents and enforce safety protocols effectively.

-- Dataset
The dataset used for this project is sourced from Kaggle:
[Construction Safety Detection Dataset.](https://www.kaggle.com/code/harinuu/construction-safety-detection)

It includes annotated images of individuals with and without helmets, enabling robust model training.

-- Usage
Dataset Preparation:

1. Download the dataset from Kaggle.
2. Place the dataset in the data/ folder in your project directory.
   
Model Training:

1. Use YOLOv5 or a similar object detection framework to train the model on the dataset.

Detection:

1. Run the detection script to analyze images or video streams.
2. Alerts will be generated for individuals not wearing helmets.
