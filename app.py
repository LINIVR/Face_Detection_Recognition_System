import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import os
from mtcnn import MTCNN
from keras_facenet import FaceNet
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import logging

# Logging Setup 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceRecognitionApp")

#  Streamlit Config 
st.set_page_config(page_title="Face Recognition System", layout="wide")
st.title(" Real-Time Face Detection & Recognition")

#  WebRTC Setup 
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

#  Model Loading 
@st.cache_resource
def load_models():
    try:
        logger.info("Loading models...")
        detector = MTCNN()
        embedder = FaceNet()
        with open("models/facerecognitionDL.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Models loaded successfully.")
        return detector, embedder, classifier, label_encoder
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        st.error("Could not load models.")
        st.stop()

detector, embedder, classifier, label_encoder = load_models()

#  Face Recognition 
def recognize_faces(image: np.ndarray):
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        annotated = image.copy()
        results = []

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            roi = rgb[y:y+h, x:x+w]

            if roi.shape[0] < 10 or roi.shape[1] < 10:
                logger.warning("Skipped tiny face.")
                continue

            resized = cv2.resize(roi, (160, 160))
            embedding = embedder.embeddings(np.expand_dims(resized, axis=0))
            prediction = classifier.predict(embedding)[0]
            probability = np.max(classifier.predict_proba(embedding))
            name = label_encoder.inverse_transform([prediction])[0] if probability > 0.3 else "Unknown"

            # Annotate image
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f"{name} ({probability*100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            results.append({
                "name": name,
                "confidence": f"{probability*100:.2f}%",
                "box": [x, y, w, h]
            })

        return annotated, results

    except Exception as e:
        logger.error(f"Face recognition failed: {e}")
        return image, []

#  Webcam Processor 
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            result, _ = recognize_faces(img)
            return av.VideoFrame.from_ndarray(result, format="bgr24")
        except Exception as e:
            logger.error(f"Webcam error: {e}")
            return frame

#  Streamlit Tabs 
tab1, tab2 = st.tabs(["📷 Real-Time Webcam", "🖼️ Upload Image"])

#  Webcam Mode 
with tab1:
    st.info("Allow camera access when prompted .")
    webrtc_streamer(
        key="realtime",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Image Upload Mode 
with tab2:
    st.subheader("Upload a photo for recognition:")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        try:
            image = Image.open(uploaded)
            img_array = np.array(image)
            bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            annotated, predictions = recognize_faces(bgr_img)

            with st.container():
                st.markdown("###  Uploaded Image &  Predictions")
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.image(image, caption="Original", use_container_width=True)

                with col2:
                    st.image(annotated, caption="Recognized", use_container_width=True)

                st.markdown("---")
                if predictions:
                    for person in predictions:
                        st.success(f"**{person['name']}** — Confidence: {person['confidence']}")
                else:
                    st.warning("No recognizable faces detected.")

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            st.error("Failed to process uploaded image.")
