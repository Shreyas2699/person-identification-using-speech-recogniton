# main.py

import os
from record_audio import create_user_dataset
from data_cleaning import clean_all_audio_data
from generate_spectrograms import generate_all_spectrograms
from train_model import train_model
from identify_speaker import recognize_speaker
import numpy as np
import tensorflow as tf

def load_model_and_encoder(model_path, encoder_path):
    model = tf.keras.models.load_model(model_path)
    label_encoder = np.load(encoder_path, allow_pickle=True)
    return model, label_encoder

def main_menu():
    while True:
        print("\n--- Speaker Identification System ---")
        print("1. Record Audio Samples for a New User")
        print("2. Clean Audio Data")
        print("3. Generate Spectrograms")
        print("4. Train the Dense Model")
        print("5. Train LSTM Model")
        print("6. Train DenseNet Model")
        print("7. Identify Speaker in Real-Time")
        print("8. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            user_name = input("Enter the name of the user: ")
            num_samples = int(input("Enter the number of samples to record: "))
            duration = int(input("Enter the duration of each sample (seconds): "))
            create_user_dataset(user_name, num_samples=num_samples, duration=duration)

        elif choice == '2':
            clean_all_audio_data()

        elif choice == '3':
            augment = input("Do you want to apply data augmentation (yes/no)? ").strip().lower() == 'yes'
            generate_all_spectrograms(augment)

        elif choice == '4':
            if not os.path.exists('data') or len(os.listdir('data')) == 0:
                print("No data found. Please record audio samples first.")
            else:
                train_model('data', 'models', 'encoders', model_type='dense',augment=True)

        elif choice == '5':
            if not os.path.exists('data') or len(os.listdir('data')) == 0:
                print("No data found. Please record audio samples first.")
            else:
                train_model('data', 'models', 'encoders', model_type='lstm',augment=True)

        elif choice == '6':
            if not os.path.exists('data') or len(os.listdir('data')) == 0:
                print("No data found. Please record audio samples first.")
            else:
                train_model('data', 'models', 'encoders', model_type='densenet',augment=True)

        elif choice == '7':
            if not os.path.exists('models/speaker_identification_model.h5') or not os.path.exists(
                    'encoders/label_encoder.npy'):
                print("Model or label encoder not found. Please train the model first.")
            else:
                model, label_encoder = load_model_and_encoder('models/speaker_identification_model.h5',
                                                              'encoders/label_encoder.npy')
                recognize_speaker(model, label_encoder)

        elif choice == '8':
            print("Exiting the system.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
