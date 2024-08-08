# Necessary imports
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import models 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the dataset with TensorFlow Datasets => Load the Oxford Flowers 102 dataset with splits
dataset, info = tfds.load('oxford_flowers102', split=['train', 'test', 'validation'], with_info=True, as_supervised=True)

# Create a training set, a validation set, and a test set
train_dataset, test_dataset, val_dataset = dataset

# Print dataset info
print(info)

# Get the number of examples in each set from the dataset info
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples
num_validation_examples = info.splits['validation'].num_examples

print(f"Number of training examples: {num_train_examples}")
print(f"Number of test examples: {num_test_examples}")
print(f"Number of validation examples: {num_validation_examples}")

# Get the number of classes in the dataset from the dataset info
num_classes = info.features['label'].num_classes

print(f"Number of classes: {num_classes}")

# Print the shape and corresponding label of 3 images in the training set
for image, label in train_dataset.take(3):
    print(f"Image shape: {image.shape}, Label: {label}")

# Plot 1 image from the training set. Set the title of the plot to the corresponding image label.
image, label = next(iter(train_dataset))

# Plot the image
plt.figure(figsize=(5, 5))
plt.imshow(image.numpy().astype("uint8"))
plt.title(f"Label: {label}")
plt.axis('off')  # Hide axes
plt.show()

# Load the class names from a JSON file
with open('label_map.json', 'r') as f:
    class_names = json.load(f)

# Plot 1 image from the training set. Set the title of the plot to the corresponding class name.
image, label = next(iter(train_dataset))

# Convert label to class name
class_name = class_names[str(label.numpy())]

# Plot the image
plt.figure(figsize=(5, 5))
plt.imshow(image.numpy().astype("uint8"))
plt.title(f"Class: {class_name}")
plt.axis('off')  # Hide axes
plt.show()

# Preprocess the images
def preprocess_image(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

def prepare_dataset(dataset, batch_size=32, shuffle_buffer_size=1000):
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Prepare the datasets
batch_size = 32
train_dataset = prepare_dataset(train_dataset, batch_size)
test_dataset = prepare_dataset(test_dataset, batch_size)
val_dataset = prepare_dataset(val_dataset, batch_size)

# Print some info about the dataset
print(f"Train dataset size: {info.splits['train'].num_examples}")
print(f"Test dataset size: {info.splits['test'].num_examples}")
print(f"Validation dataset size: {info.splits['validation'].num_examples}")

# Build and train your network

# Build the model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Create the model using the Sequential API
model = models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Using num_classes from info.features['label'].num_classes
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

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)

# Print the loss and accuracy values
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the model in the native Keras format
model.save('flower_classifier_model.keras')

# Load the Keras model
loaded_model = tf.keras.models.load_model('flower_classifier_model.keras')

# Verify that the model has been loaded correctly by printing its summary
loaded_model.summary()

# Create the process_image function
def process_image(image):
    # Convert the image to a TensorFlow Tensor
    image = tf.convert_to_tensor(image)
    # Resize the image to (224, 224)
    image = tf.image.resize(image, (224, 224))
    # Normalize the pixel values to be in the range [0, 1]
    image = image / 255.0
    # Convert the image back to a NumPy array
    return image.numpy()

# Example usage of process_image function
image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()

# Create the predict function
def predict(image_path, model, top_k=5):
    # Load the image using PIL
    image = Image.open(image_path)
    # Convert the image to a NumPy array
    image = np.asarray(image)
    # Preprocess the image
    processed_image = process_image(image)
    # Add an extra batch dimension since the model expects a batch of images
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Use the model to predict the class probabilities
    predictions = model.predict(processed_image)
    
    # Get the top K probabilities and class labels
    top_k_probs = np.sort(predictions[0])[-top_k:][::-1]
    top_k_classes = np.argsort(predictions[0])[-top_k:][::-1]
    
    # Convert class indices to strings
    top_k_classes = [str(cls) for cls in top_k_classes]
    
    return top_k_probs, top_k_classes

# Plot the input image along with the top 5 classes
def plot_image_with_predictions(image_path, model, top_k=5):
    # Load and preprocess the image
    image = Image.open(image_path)
    processed_image = np.asarray(image)
    processed_image = process_image(processed_image)
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Get predictions
    probs, classes = predict(image_path, model, top_k)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the image
    ax[0].imshow(Image.open(image_path))
    ax[0].axis('off')
    ax[0].set_title('Input Image')
    
    # Plot the top K predictions
    ax[1].barh(range(top_k), probs, tick_label=classes, color='skyblue')
    ax[1].set_yticks(range(top_k))
    ax[1].set_yticklabels(classes)
    ax[1].set_xlabel('Probability')
    ax[1].set_title(f'Top {top_k} Predictions')
    
    plt.tight_layout()
    plt.show()

# Example usage of plot_image_with_predictions function
image_path = './test_images/wild_pansy.jpg'
plot_image_with_predictions(image_path, loaded_model, top_k=5)