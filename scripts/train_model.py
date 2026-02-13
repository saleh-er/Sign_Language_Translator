import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load data
df = pd.read_csv('data/gestures.csv')
X = df.iloc[:, :-1].values  # All columns except the last (coordinates)
y = df.iloc[:, -1].values   # The last column (labels)

# Encode labels (e.g., "HELLO" becomes 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
np.save('models/classes.npy', label_encoder.classes_) # Save labels for later

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(63,)), # 21 landmarks * 3 (x,y,z)
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax') # Output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

model.save('models/sign_model.h5')
print("Model saved to models/sign_model.h5")