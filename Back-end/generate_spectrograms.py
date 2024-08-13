import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs


def extract_and_save_spectrogram(audio_path, output_path, augment=False):
    y, sr = librosa.load(audio_path, sr=None)

    if augment:
        # Add noise
        noise = np.random.randn(len(y))
        y = y + 0.005 * noise

        # Pitch shift (corrected with keyword argument)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

    mfccs = extract_features(audio_path)
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.savefig(output_path)
    plt.close()


def generate_user_spectrograms(user_name, augment=False):
    user_path = os.path.join('data', user_name)
    if not os.path.exists(user_path):
        print(f"No data found for user {user_name}.")
        return

    print(f"Generating spectrograms for user '{user_name}'...")
    spectrogram_path = os.path.join('data', user_name, 'spectrograms')
    os.makedirs(spectrogram_path, exist_ok=True)

    for file_name in os.listdir(user_path):
        if file_name.endswith('.wav'):
            audio_path = os.path.join(user_path, file_name)
            output_path = os.path.join(spectrogram_path, f"{file_name.split('.')[0]}.png")
            extract_and_save_spectrogram(audio_path, output_path, augment)
            print(f"Generated spectrogram for {file_name}")

    print(f"Spectrogram generation completed for user '{user_name}'.")


def generate_all_spectrograms(augment=False):
    for user_name in os.listdir('data'):
        generate_user_spectrograms(user_name, augment)
    print("Spectrograms generated for all users.")
