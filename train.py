import tensorflow as tf
from keras import models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the dataset
dataset, info = tfds.load('oxford_flowers102', split=['train', 'test', 'validation'], with_info=True, as_supervised=True)

# Create training, validation, and test sets
train_dataset, test_dataset, val_dataset = dataset

# Preprocess the images
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

def prepare_dataset(dataset, batch_size=32, shuffle_buffer_size=1000):
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 32
train_dataset = prepare_dataset(train_dataset, batch_size)
test_dataset = prepare_dataset(test_dataset, batch_size)
val_dataset = prepare_dataset(val_dataset, batch_size)

# Build the model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(info.features['label'].num_classes, activation='softmax')  # Adjust the number of output classes if needed
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Function to plot the training history
def plot_training_history(history):
    # Plot the training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()

# Call the function to plot the training history
plot_training_history(history)

# Save the model in the native Keras format
model.save('flower_classifier_model.keras')
