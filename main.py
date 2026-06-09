import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AI Age & Gender Detector",
    page_icon="🧑",
    layout="wide"
)

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp{
    background: linear-gradient(to right,#141E30,#243B55);
}
h1{
    text-align:center;
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>🤖 AI Gender & Age Detector</h1>
<h4 style='text-align:center;color:#94a3b8'>
Upload an image and let AI predict gender and age
</h4>
""", unsafe_allow_html=True)

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader("Choose a file:")

# ---------------- MODEL PATHS ----------------
face_model_path = "opencv_face_detector_uint8.pb"
face_txt_path = "opencv_face_detector.pbtxt"

age_model_path = "age_net.caffemodel"
age_txt_path = "age_deploy.prototxt"

gender_model_path = "gender_net.caffemodel"
gender_txt_path = "gender_deploy.prototxt"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 🔥 FIXED AGE CLASSES (IMPORTANT)
age_classes = [
    '(0-2)',
    '(3-5)',
    '(6-15)',
    '(16-24)',
    '(25-34)',
    '(35-44)',
    '(45-59)',
    '(60-100)'
]

gender_classes = ['Male', 'Female']

# ---------------- LOAD MODELS ----------------
face_net = cv2.dnn.readNet(face_model_path, face_txt_path)
age_net = cv2.dnn.readNetFromCaffe(age_txt_path, age_model_path)
gender_net = cv2.dnn.readNetFromCaffe(gender_txt_path, gender_model_path)

# ---------------- FACE DETECTION ----------------
def get_face_box(net, frame, conf_threshold=0.7):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()

    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            boxes.append([x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, boxes

# ---------------- MAIN ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frameFace, boxes = get_face_box(face_net, frame)

    if not boxes:
        st.warning("No face detected!")

    padding = 20

    for bbox in boxes:

        face = frame[
            max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]),
            max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1])
        ]

        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )

        # ---------------- GENDER ----------------
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_index = gender_preds[0].argmax()
        gender = gender_classes[gender_index]

        # ---------------- AGE ----------------
        age_net.setInput(blob)
        age_preds = age_net.forward()

        age_index = int(age_preds[0].argmax())

        # 🔥 SAFE CHECK (PREVENT CRASH)
        if age_index >= len(age_classes):
            age = "Unknown"
        else:
            age = age_classes[age_index]

        # ---------------- LABEL ----------------
        label = f"{gender}, {age}"

        cv2.putText(
            frameFace,
            label,
            (bbox[0], bbox[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # ---------------- UI ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image)

        with col2:
            st.subheader("🧠 Detection Result")
            st.image(cv2.cvtColor(frameFace, cv2.COLOR_BGR2RGB))

    st.success("Detection Completed!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<center>
<h4 style='color:#94a3b8'>
🚀 Developed using OpenCV + Streamlit + Deep Learning
</h4>
</center>
""", unsafe_allow_html=True)