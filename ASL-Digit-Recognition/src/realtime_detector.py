import cv2
import numpy as np
import os
import time
from keras.models import load_model
from collections import deque

# ==== Setup Relative Path for Portability ====
# This ensures the model loads correctly regardless of the computer used
current_directory = os.path.dirname(os.path.abspath(__file__))
# Moves up one level to find the 'model' folder and the specific file
model_path = os.path.join(current_directory, "..", "model", "final_asl_model.keras")

# ==== Load Trained Model ====
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully from relative path!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure the model file is in the 'model' folder.")
    exit()

# ==== Constants and Configuration ====
ROI_top, ROI_bottom = 100, 300
ROI_right, ROI_left = 150, 350
image_size = (96, 96)
accumulated_weight = 0.5
wait_time = 20
font = cv2.FONT_HERSHEY_SIMPLEX

# ==== Background Initialization Variables ====
background = None
num_frames = 0

# ==== Prediction smoothing queue to reduce flickering ====
pred_queue = deque(maxlen=5)

# ==== Gesture class labels ====
gesture_names = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

# ==== Core Functions ====

def cal_accum_avg(frame, accumulated_weight):
    """Calculates the weighted average of the background for subtraction."""
    global background
    if background is None:
        background = frame.copy().astype("float")
    else:
        cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    """Segments the hand from the background using absolute difference."""
    global background
    if background is None:
        return None
    
    # Calculate difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Finding contours
    contours_info = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    
    if len(contours) == 0:
        return None
    
    # Return the largest contour (the hand)
    return thresholded, max(contours, key=cv2.contourArea)

def preprocess_frame(roi):
    """Prepares the ROI image for model prediction."""
    img = cv2.resize(roi, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB as expected by MobileNetV2
    img = img.astype("float32") / 255.0       # Normalization
    return np.expand_dims(img, axis=0)        # Add batch dimension

def display_info(frame, text, position, color=(0, 255, 0), size=0.7):
    """Helper function to overlay text on the video frame."""
    cv2.putText(frame, text, position, font, size, color, 2, cv2.LINE_AA)

# ==== Main Execution Loop ====
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🖐️ Initializing Camera... Please keep the hand out of the ROI for calibration.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Error: Failed to capture image.")
        break

    frame = cv2.flip(frame, 1) # Flip frame horizontally for natural movement
    frame_copy = frame.copy()

    # Extract Region of Interest (ROI)
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Background calibration phase
    if num_frames < wait_time:
        cal_accum_avg(gray_blurred, accumulated_weight)
        display_info(frame_copy, f"⏳ Calibrating... ({num_frames}/{wait_time})", (10, 40), (0, 0, 255))
    else:
        # Prediction phase
        hand = segment_hand(gray_blurred)
        if hand is not None:
            thresholded, hand_segment = hand

            # Model inference
            input_data = preprocess_frame(roi)
            prediction = model.predict(input_data, verbose=0)[0]
            pred_queue.append(prediction)

            # Averaging predictions for stability
            avg_pred = np.mean(pred_queue, axis=0)
            pred_class = np.argmax(avg_pred)
            confidence = np.max(avg_pred) * 100

            # UI Text update
            gesture_text = gesture_names.get(pred_class, f"Class {pred_class}")
            text = f"Gesture: {gesture_text} ({confidence:.1f}%)"
            display_info(frame_copy, text, (10, 40))
            
            # Show processed windows
            cv2.imshow("Thresholded", thresholded)
            cv2.imshow("Grayscale ROI", gray_blurred)
            
            # Draw contours around the detected hand
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 2)
        else:
            display_info(frame_copy, "No hand detected", (10, 40), (0, 0, 255))

    # UI Overlay: Draw ROI box and show final frame
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 2)
    cv2.imshow("ASL Digit Recognition - Real Time", frame_copy)

    # Input handling
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press ESC to exit
        break
    elif key == ord('r'):  # Press 'r' to reset calibration
        background = None
        num_frames = 0
        print("🔄 Background reset triggered!")

    num_frames += 1

# Cleanup
cam.release()
cv2.destroyAllWindows()
print("👋 Program closed.")