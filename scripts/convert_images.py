import cv2
import mediapipe as mp
import pandas as pd
import os

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# CONFIGURATION
IMAGE_DIR = r"C:\Users\otman\Downloads\archive (1).zip\processed_combine_asl_dataset" # Update this to your images folder
OUTPUT_CSV = "data/external_gestures.csv"

data_list = []

print("Processing images... this might take a minute.")

# Loop through each folder (A, B, C, etc.)
for label in os.listdir(IMAGE_DIR):
    label_path = os.path.join(IMAGE_DIR, label)
    
    if not os.path.isdir(label_path):
        continue

    print(f"Extracting landmarks for: {label}")
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)
        
        if image is None: continue

        # Convert to RGB for MediaPipe
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                landmarks.append(label)
                data_list.append(landmarks)

# Save to CSV
df = pd.DataFrame(data_list)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Successfully converted {len(data_list)} images to {OUTPUT_CSV}")