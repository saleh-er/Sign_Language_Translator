import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# CONFIGURATION: Change this for every new sign you record
LABEL = "THANK_YOU"  # Example: "HELLO", "THANK_YOU", "YES", "NO"
DATA_PATH = "data/gestures.csv"

cap = cv2.VideoCapture(0)
data = []

print(f"Starting collection for: {LABEL}")
print("Position your hand. Recording starts in 3 seconds...")
time.sleep(3)

while len(data) < 200: # Collect 200 samples per sign
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract 21 landmarks (x, y, z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            landmarks.append(LABEL)
            data.append(landmarks)

    cv2.putText(frame, f"Samples: {len(data)}/200", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Collecting Data', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(DATA_PATH, mode='a', header=not os.path.exists(DATA_PATH), index=False)
print(f"Done! Saved {LABEL} data to {DATA_PATH}")

cap.release()
cv2.destroyAllWindows()