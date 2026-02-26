import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras import layers, models

data_dir = r"D:\PythonProject\mnist_detect\digits updated"

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=(28, 28),
    color_mode='grayscale',
    batch_size=32
)


def visualize_dataset(dataset, num_images=9):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):

        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(tf.squeeze(images[i]), cmap='gray')
            plt.title(f'Label: {labels[i].numpy()}')
            plt.axis("off")
        plt.show()

    # Visualize the dataset

def visualize_dataset(dataset, num_images=9):
    plt.figure(figsize=(16, 6))
    for images, labels in dataset.take(1):
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)

            # Add border color based on label (convert labels to an int to choose color)
            border_color = 'green' if labels[i].numpy() % 2 == 0 else 'red'
            plt.imshow(tf.squeeze(images[i]), cmap='gray')  # Show grayscale image

            # Add title with different font size and style
            plt.title(f'Label: {labels[i].numpy()}', fontsize=12, color=border_color)

            # Create colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)  # Make the border thicker

            plt.axis("off")

    plt.tight_layout()
    plt.show()


visualize_dataset(dataset)
batch_size = 32
img_size = (128, 128)
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=img_size,
    seed=123
)
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    color_mode="grayscale",
    image_size=img_size,
    batch_size=batch_size,
    seed=123

)
for images, labels in train_data.take(1):
    print(f'Images batch shape: {images.shape}')
    print(f'Labels batch shape: {labels.shape}')
normalization_layer = Rescaling(1. / 255)

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizorntal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])
train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))




visualize_dataset(dataset)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1), padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.Flatten(),

    layers.Dense(64, activation="relu", ),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.summary()
history = model.fit(train_data, validation_data=val_data, epochs=10)
model.save("mnistv2.h5")