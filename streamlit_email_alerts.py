#app.py
# streamlit run d:/safety-gear-cdl/streamlit_email_alerts.py

# Import necessary libraries for Streamlit and YOLO model
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
import yagmail
import base64

detection_flag = False #line 1 correct 

# Load the model
model = YOLO(r'D:\safety-gear-cdl\safety_colabdl\safety_gear_detection_final\results_yolov8n_100e\kaggle\working\runs\detect\train\weights\best.pt')
# Define colors
colors = {
    'safe': (0, 255, 0),       # Green for wearing safety gear
    'unsafe': (0, 0, 255)      # Red for not wearing safety gear
}
st.set_page_config(page_title="Safety Gear Detection System", layout="wide")

def add_background(image_file):
    with open(image_file, "rb") as image:
        base64_image = base64.b64encode(image.read()).decode()
    css_code = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)


# Call the function to set background
add_background(r'D:\Download\img4.jpeg')  # Replace with your image file path


# Set up layout
st.title("🚨 Safety Gear Detection System")
st.write("Identify individuals missing safety gear in uploaded images or videos.")

# Choose file type
upload_choice = st.radio("Choose input type:", ("Image", "Video"))

# Main content
if upload_choice == "Image":
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image file (JPEG or PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_image)
        
        # Convert image to OpenCV format
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run detection
        results = model.predict(source=img)

        # Display results
        detection_flag = False
        for result in results:
            boxes = result.boxes  # Access the boxes attribute
            if len(boxes.xyxy) > 0:  # Check if any boxes were detected
                labels = boxes.cls.tolist()  # Get detected class indices
                detected_labels = [result.names[int(label)] for label in labels]

                # Annotate the image with bounding boxes and labels
                for box, label in zip(boxes.xyxy, detected_labels):
                    x1, y1, x2, y2 = map(int, box)
                    color = colors['unsafe'] if label in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] else colors['safe']
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Check for missing safety gear
                if 'NO-Hardhat' in detected_labels or 'NO-Mask' in detected_labels or 'NO-Safety Vest' in detected_labels:
                    detection_flag = True

        annotated_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        st.subheader("Detection Result")
        st.image(annotated_image, caption="Processed Image", use_column_width=True)

        # Display alert if safety violation is detected
        if detection_flag:
            st.error("⚠ Alert: Some individuals are not wearing safety gear!")
        else:
            st.success("✅ All individuals are wearing safety gear.")

elif upload_choice == "Video":
    st.header("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        st.video(uploaded_video)

        # Create temporary files for input and output video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
            temp_input_file.write(uploaded_video.read())
            temp_input_path = temp_input_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
            temp_output_path = temp_output_file.name

        # Process video
        cap = cv2.VideoCapture(temp_input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Video processing progress
        st.subheader("Processing Video...")
        progress = st.progress(0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        detection_flag = False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process each frame
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection on the frame
            results = model.predict(source=frame)

            for result in results:
                boxes = result.boxes
                if len(boxes.xyxy) > 0:
                    labels = boxes.cls.tolist()
                    detected_labels = [result.names[int(label)] for label in labels]

                    for box, label in zip(boxes.xyxy, detected_labels):
                        x1, y1, x2, y2 = map(int, box)
                        color = colors['unsafe'] if label in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] else colors['safe']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if 'NO-Hardhat' in detected_labels or 'NO-Mask' in detected_labels or 'NO-Safety Vest' in detected_labels:
                        detection_flag = True

            out.write(frame)
            frame_num += 1
            progress.progress(frame_num / total_frames)

        # Release resources
        cap.release()
        out.release()
        st.success("Video processing complete.")

        # Display processed video and download button
        with open(temp_output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

        # Display alert if safety violation is detected
        if detection_flag:
            st.error("⚠ Alert: Some individuals are not wearing safety gear in the video!")
        else:
            st.success("✅ All individuals are wearing safety gear in the video.")



def send_email_alert(recipient, violation_details):
    try:
        # Sender's email credentials (use your email here)
        yag = yagmail.SMTP("janhavione@gmail.com", "rizo dizs joge dqkj")
        
        # Email content
        subject = "🚨 Safety Violation Detected"
        content = f"Dear Employee,\n\nThe system has detected a safety violation:\n\n{violation_details}\n\nPlease address this immediately."
        
        # Send the email
        yag.send(to=recipient, subject=subject, contents=content)
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Failed to send email alert: {e}")

# Usage in your detection system
if detection_flag:  # If safety violation is detected
    violation_details = "Detected individuals without hardhats, masks, or safety vests."
    employee_email = "jaipurkarjanhavi@gmail.com"  # Replace with the actual employee's email
    send_email_alert(employee_email, violation_details)
