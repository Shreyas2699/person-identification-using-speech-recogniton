import os
import sounddevice as sd
import wavio

def create_user_dataset(user_name, num_samples=40, duration=3):
    if not os.path.exists('data'):
        os.makedirs('data')

    user_path = os.path.join('data', user_name)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    existing_samples = [f for f in os.listdir(user_path) if f.startswith('sample_') and f.endswith('.wav')]
    existing_sample_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_samples]
    start_num = max(existing_sample_numbers) + 1 if existing_sample_numbers else 1

    print(f"Recording {num_samples} samples for user '{user_name}' starting from sample {start_num}...")

    for i in range(num_samples):
        print(f"Recording sample {start_num + i}/{num_samples + start_num - 1}...")
        sample = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='int16')
        sd.wait()
        file_path = os.path.join(user_path, f"sample_{start_num + i}.wav")
        wavio.write(file_path, sample, 44100, sampwidth=2)
        print(f"Saved to {file_path}")

    print("Recording completed.")
