import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.preprocessing.image as img_preprocessing
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path, model_type):
    img = img_preprocessing.load_img(image_path, color_mode='grayscale', target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img) / 255.0

    if model_type == 'densenet':
        img_array = np.repeat(img_array, 3, axis=-1)  # Convert grayscale to RGB

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def load_data(data_dir, model_type, augment=False):
    images = []
    labels = []

    # Initialize the ImageDataGenerator with augmentation if enabled
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    for user_name in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_name, 'spectrograms')
        if not os.path.exists(user_path):
            continue

        for file_name in os.listdir(user_path):
            if file_name.endswith('.png'):
                img_path = os.path.join(user_path, file_name)
                img = img_preprocessing.load_img(img_path, color_mode='grayscale', target_size=(224, 224))
                img_array = img_preprocessing.img_to_array(img) / 255.0

                if model_type == 'densenet':
                    img_array = np.repeat(img_array, 3, axis=-1)  # Convert grayscale to RGB

                images.append(img_array)
                labels.append(user_name)

                # Augment the image if enabled
                if augment:
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                    for _ in range(5):  # Generate 5 augmented images per original image
                        aug_iter = datagen.flow(img_array, batch_size=1)
                        aug_img = next(aug_iter)[0]
                        images.append(aug_img)
                        labels.append(user_name)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def build_dense_model(hp, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(224, 224, 1)),
        tf.keras.layers.Dense(units=hp.Int('dense_units1', min_value=128, max_value=512, step=128), activation='relu'),
        tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(units=hp.Int('dense_units2', min_value=64, max_value=256, step=64), activation='relu'),
        tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_densenet_model(hp, num_classes):
    input_shape = (224, 224, 3)  # DenseNet expects RGB inputs
    base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Ensure a single output tensor
    x = tf.keras.layers.Dense(hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1))(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_lstm_model(hp, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((224, 224 * 1), input_shape=(224, 224, 1)),
        tf.keras.layers.LSTM(hp.Int('lstm_units1', min_value=64, max_value=256, step=64), return_sequences=True),
        tf.keras.layers.LSTM(hp.Int('lstm_units2', min_value=32, max_value=128, step=32)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_hyperparameters(X_train, y_train_enc, num_classes, model_type):
    if model_type == 'dense':
        build_model = lambda hp: build_dense_model(hp, num_classes)
    elif model_type == 'densenet':
        build_model = lambda hp: build_densenet_model(hp, num_classes)
    elif model_type == 'lstm':
        build_model = lambda hp: build_lstm_model(hp, num_classes)
    else:
        raise ValueError("Unknown model type")

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,  # Limit the number of trials
        directory='tuner',
        project_name='speaker_identification'
    )

    tuner.search(X_train, y_train_enc, epochs=35, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Best Hyperparameters: {best_hps.values}")

    return best_hps

def train_model(data_dir, model_dir, encoder_dir, model_type, augment=False):
    images, labels = load_data(data_dir, model_type, augment=augment)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    os.makedirs(encoder_dir, exist_ok=True)
    np.save(os.path.join(encoder_dir, 'label_encoder.npy'), label_encoder.classes_)

    num_classes = len(label_encoder.classes_)

    best_hps = tune_hyperparameters(X_train, y_train_enc, num_classes, model_type)

    if model_type == 'dense':
        model = build_dense_model(best_hps, num_classes)
    elif model_type == 'densenet':
        model = build_densenet_model(best_hps, num_classes)
    elif model_type == 'lstm':
        model = build_lstm_model(best_hps, num_classes)
    else:
        raise ValueError("Unknown model type")

    model.fit(X_train, y_train_enc, validation_data=(X_test, y_test_enc), epochs=35, batch_size=32)

    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'speaker_identification_model.h5'))
    print("Model training completed and saved.")
