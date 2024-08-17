import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import mediapipe as mp
import pyttsx3
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to get all sign classes from the dataset directory
def get_sign_classes(data_dir):
    return [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.mp4')]

# Create and compile the model
def create_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(30, 21 * 3)),  # 30 frames, 21 landmarks with x, y, z coordinates
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Process landmarks
def process_landmarks(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

# Data generation function
def generate_data(data_dir, sign_classes):
    X = []
    y = []
    for class_index, class_name in enumerate(sign_classes):
        video_path = os.path.join(data_dir, f"{class_name}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video for '{class_name}' not found.")
            continue
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = process_landmarks(hand_landmarks.landmark)
                frames.append(landmarks.flatten())  # Flatten the landmarks
        cap.release()
        
        # Ensure we have at least 30 frames
        if len(frames) < 30:
            print(f"Warning: Video for '{class_name}' has less than 30 frames with detected hands.")
            continue
        
        # Use sliding window to create multiple samples from one video
        for i in range(0, len(frames) - 29):
            X.append(np.array(frames[i:i+30]))
            y.append(class_index)
    
    return np.array(X), np.array(y)

# Train the model
def train_model(model, X, y, epochs=50, batch_size=32):
    y_categorical = keras.utils.to_categorical(y)
    model.fit(X, y_categorical, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save_weights('sign_language_model.weights.h5')

# Predict sign
def predict_sign(model, landmarks_sequence, sign_classes):
    # Ensure landmarks_sequence has the correct shape (30, 21 * 3)
    if len(landmarks_sequence) < 30:
        # Pad with zeros if less than 30 frames
        padding = np.zeros((30 - len(landmarks_sequence), 21 * 3))
        landmarks_sequence = np.vstack((landmarks_sequence, padding))
    elif len(landmarks_sequence) > 30:
        # Take the last 30 frames if more than 30
        landmarks_sequence = landmarks_sequence[-30:]
    
    # Reshape to (1, 30, 21 * 3) for model input
    reshaped_sequence = np.expand_dims(landmarks_sequence, axis=0)
    prediction = model.predict(reshaped_sequence)
    return sign_classes[np.argmax(prediction)]

# Text-to-speech function
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Main function
def main():
    data_dir = 'dataset'  # Replace with the path to your dataset
    sign_classes = get_sign_classes(data_dir)
    
    # Create model
    model = create_model(len(sign_classes))

    print("Training new model...")
    # Generate and preprocess data
    X, y = generate_data(data_dir, sign_classes)
    
    # Train the model
    train_model(model, X, y)

   

    # Open video capture
    cap = cv2.VideoCapture(0)
    transcription = ""
    audio_enabled = False
    last_recognition_time = time.time()
    current_word = ""
    landmarks_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - last_recognition_time

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Add landmarks to sequence
                landmarks = process_landmarks(hand_landmarks.landmark)
                landmarks_sequence.append(landmarks.flatten())  # Flatten the landmarks
                if len(landmarks_sequence) > 30:
                    landmarks_sequence.pop(0)

                if len(landmarks_sequence) == 30 and elapsed_time >= 3:  # 3-second delay between recognitions
                    # Predict sign
                    predicted_word = predict_sign(model, np.array(landmarks_sequence), sign_classes)
                    current_word = predicted_word

                    # Display predicted word
                    cv2.putText(frame, f"Predicted: {predicted_word}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press Enter to confirm", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display transcription
        cv2.putText(frame, f"Transcription: {transcription}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display instructions
        cv2.putText(frame, "Press 'A' to toggle audio", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Press Spacebar to add blank", (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Sign Language Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            audio_enabled = not audio_enabled
            if audio_enabled:
                text_to_speech(transcription)
        elif key == 13:  # Enter key
            if current_word:
                transcription += current_word + " "
                current_word = ""
                last_recognition_time = time.time()
                landmarks_sequence = []
        elif key == 32:  # Spacebar
            transcription += " "
            last_recognition_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()