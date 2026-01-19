from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
IMAGE_HEIGHT = 96   # change if different
IMAGE_WIDTH = 96
IMAGE_CHANNELS = 1
CONFIDENCE_THRESHOLD = 0.80
MODEL_PATH = "/mnt/d/Smart_Attendence_System/cnn_model/models/face_recognition_cnn.h5"
# ----------------------------------------

app = FastAPI()

# Load model ONCE at startup
model = load_model(MODEL_PATH)
print("✅ Face recognition model loaded")

# ---------------- HELPERS ----------------
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    gray = cv2.resize(gray, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Normalize
    gray = gray.astype("float32") / 255.0

    # Add channel dimension (H, W, 1)
    gray = np.expand_dims(gray, axis=-1)

    # Add batch dimension (1, H, W, 1)
    gray = np.expand_dims(gray, axis=0)

    return gray



def recognize_face(img):
    """
    Runs classifier model and returns
    (student_id, confidence)
    """
    processed = preprocess_image(img)
    preds = model.predict(processed, verbose=0)[0]

    student_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return student_id, confidence
# -----------------------------------------


@app.post("/attendance/scan")
async def scan_attendance(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    # Read image bytes
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "status": "error",
            "message": "Invalid image"
        }

    # ---- MODEL INFERENCE ----
    student_id, confidence = recognize_face(img)

    # ---- DECISION LOGIC ----
    if confidence >= CONFIDENCE_THRESHOLD:
        # TODO: DB insert here (mark attendance)
        return {
            "status": "present",
            "student_id": student_id,
            "confidence": round(confidence, 3),
            "session_id": session_id
        }
    else:
        # TODO: DB insert here (red flag)
        return {
            "status": "unknown",
            "confidence": round(confidence, 3),
            "session_id": session_id
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
