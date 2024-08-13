import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as img_preprocessing
import speech_recognition as sr
from generate_spectrograms import extract_and_save_spectrogram

def load_model_and_encoder(model_path, encoder_path):
    model = tf.keras.models.load_model(model_path)
    label_encoder = np.load(encoder_path, allow_pickle=True)
    return model, label_encoder

def recognize_speaker(model, label_encoder):
    import sounddevice as sd
    import wavio

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Recording...")
            audio = recognizer.listen(source)
            with open("temp.wav", "wb") as f:
                f.write(audio.get_wav_data())
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return

    print("Processing...")
    extract_and_save_spectrogram("temp.wav", "temp_spectrogram.png", augment=False)

    img = img_preprocessing.load_img("temp_spectrogram.png", color_mode='grayscale', target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    if model.input_shape[1:] == (224, 224, 3):  # Check if model expects RGB input
        img_array = np.repeat(img_array, 3, axis=-1)  # Convert grayscale to RGB

    model_predictions = model.predict(img_array)
    predicted_class = np.argmax(model_predictions, axis=1)
    predicted_label = label_encoder[predicted_class[0]]

    print(f"Recognized speaker: {predicted_label}")
