import os
import librosa
import numpy as np
import soundfile as sf

def remove_silence(audio, sr, top_db=20):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return non_silent_audio

def normalize_volume(audio):
    max_volume = np.max(np.abs(audio))
    return audio / max_volume

def reduce_noise(audio, sr):
    return librosa.effects.preemphasis(audio)

def remove_distortions(audio, sr):
    harmonic, percussive = librosa.effects.hpss(audio)
    return harmonic

def resample_audio(audio, sr, target_sr=16000):
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

def clean_audio_file(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=None)

    y = remove_silence(y, sr)
    y = normalize_volume(y)
    y = reduce_noise(y, sr)
    y = remove_distortions(y, sr)
    y = resample_audio(y, sr, target_sr)

    sf.write(file_path, y, target_sr)
    print(f"Cleaned and saved: {file_path}")

def clean_user_audio(user_name, target_sr=16000):
    user_path = os.path.join('data', user_name)
    if not os.path.exists(user_path):
        print(f"No data found for user {user_name}.")
        return

    print(f"Cleaning audio files for user '{user_name}'...")
    for file_name in os.listdir(user_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(user_path, file_name)
            clean_audio_file(file_path, target_sr)

    print(f"Audio cleaning completed for user '{user_name}'.")

def clean_all_audio_data(target_sr=16000):
    for user_name in os.listdir('data'):
        clean_user_audio(user_name, target_sr)
    print("All audio data cleaned.")
