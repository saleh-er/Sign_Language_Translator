import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Load the data you collected
if not os.path.exists('data/asl_data.csv'):
    print("Error: data/asl_data.csv not found. Please run collect_data.py first!")
    exit()

df = pd.read_csv('data/asl_data.csv')
X = df.iloc[:, :-1].values  # Landmark coordinates (x, y, z)
y = df.iloc[:, -1].values   # Gesture names (labels)

# 2. Convert text labels (HELLO, YES) into numbers (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the class names so the app knows what the numbers mean later
if not os.path.exists('models'):
    os.makedirs('models')
np.save('models/classes.npy', label_encoder.classes_)

# 3. Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Build the Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)), # 21 points * 3 coords
    Dropout(0.2), # Prevents overfitting
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax') # Final prediction
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train!
print("Training started...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 6. Save the model
model.save('models/sign_model.h5')
print("Success! Model saved to models/sign_model.h5")