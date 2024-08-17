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

# Define dataset directory
dataset_dir = 'custom_dataset'

# Create and compile the model
def create_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(21, 3)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Process landmarks
def process_landmarks(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

# Data generation function
def generate_data(data_dir):
    X = []
    y = []
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = process_landmarks(hand_landmarks.landmark)
                X.append(landmarks)
                y.append(i)
    return np.array(X), np.array(y), classes

# Train the model
def train_model(model, X, y, num_classes, epochs=50, batch_size=32):
    y_categorical = keras.utils.to_categorical(y, num_classes=num_classes)
    model.fit(X, y_categorical, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save_weights('custom_sign_language_model.weights.h5')

# Predict sign
def predict_sign(model, landmarks, classes):
    processed_landmarks = process_landmarks(landmarks)
    prediction = model.predict(np.expand_dims(processed_landmarks, axis=0))
    return classes[np.argmax(prediction)]

# Text-to-speech function
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Function to create or update dataset
def manage_dataset():
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    while True:
        print("\nDataset Management:")
        print("1. Add new word")
        print("2. Add images to existing word")
        print("3. View existing words")
        print("4. Return to main menu")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            word = input("Enter the new word: ").strip()
            word_dir = os.path.join(dataset_dir, word)
            if os.path.exists(word_dir):
                print(f"The word '{word}' already exists in the dataset.")
            else:
                os.makedirs(word_dir)
                capture_images(word, word_dir)
        elif choice == '2':
            words = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            if not words:
                print("No words in the dataset yet.")
                continue
            print("Existing words:", ', '.join(words))
            word = input("Enter the word to add images to: ").strip()
            word_dir = os.path.join(dataset_dir, word)
            if os.path.exists(word_dir):
                capture_images(word, word_dir)
            else:
                print(f"The word '{word}' does not exist in the dataset.")
        elif choice == '3':
            words = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            if words:
                print("Existing words:", ', '.join(words))
            else:
                print("No words in the dataset yet.")
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

# Function to capture images for a word
def capture_images(word, word_dir):
    cap = cv2.VideoCapture(0)
    count = len([f for f in os.listdir(word_dir) if f.endswith('.jpg')])
    
    print(f"Capturing images for '{word}'. Press 'C' to capture, 'Q' to finish.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Word: {word}, Count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Capture Sign', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img_name = os.path.join(word_dir, f"{word}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Captured {img_name}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing images for '{word}'.")

# Main function
def main():
    while True:
        print("\nChoose an option:")
        print("1. Manage dataset")
        print("2. Train and run sign language recognition")
        print("3. Quit")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            manage_dataset()
        elif choice == '2':
            # Generate and preprocess data
            X, y, classes = generate_data(dataset_dir)
            
            if len(classes) == 0:
                print("No data in the dataset. Please add some words and images first.")
                continue
            
            # Create model
            model = create_model(len(classes))

            # Check if weights file exists
            if os.path.exists('custom_sign_language_model.weights.h5'):
                print("Loading existing model weights...")
                model.load_weights('custom_sign_language_model.weights.h5')
            else:
                print("Training new model...")
                # Train the model
                train_model(model, X, y, len(classes))

            # Open video capture
            cap = cv2.VideoCapture(0)
            transcription = ""
            audio_enabled = False
            last_recognition_time = time.time()
            current_sign = ""

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

                if results.multi_hand_landmarks and elapsed_time >= 3:  # 3-second delay between recognitions
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Predict sign
                        predicted_sign = predict_sign(model, hand_landmarks.landmark, classes)
                        current_sign = predicted_sign

                        # Display predicted sign
                        cv2.putText(frame, f"Predicted: {predicted_sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
                    if current_sign:
                        transcription += current_sign
                        current_sign = ""
                        last_recognition_time = time.time()
                elif key == 32:  # Spacebar
                    transcription += " "
                    last_recognition_time = time.time()

            cap.release()
            cv2.destroyAllWindows()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()