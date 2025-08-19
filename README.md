Face Detection and  Recognition System 🎥👤

A real-time face detection and recognition system built with Streamlit, MTCNN, FaceNet, and a custom-trained classifier.
Supports both image-based recognition and live webcam recognition using WebRTC.

✨ Features

🔍 Face Detection → Detects multiple faces in an image or video stream.

🧠 Face Recognition → Uses FaceNet embeddings + trained classifier for identity prediction.

📷 Image Upload → Recognize faces in uploaded images.

🎥 Live Webcam Recognition → Real-time recognition via WebRTC.

⚡ Fallback Option → Uses OpenCV if WebRTC is not supported.

📝 Logging → Structured logging for debugging and monitoring.

🛠️ Tech Stack

Streamlit
 → Web interface

OpenCV
 → Image processing

MTCNN
 → Face detection

FaceNet
 → Face embeddings

Pickle
 → Model + LabelEncoder storage

streamlit-webrtc
 → Real-time video streaming