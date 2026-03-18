# Import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'dataset/',            # IMPORTANT: folder name must be 'dataset'
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# -----------------------------
# CNN MODEL
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

# -----------------------------
# EVALUATE
# -----------------------------
loss, acc = model.evaluate(val_data)
print("\n✅ Final Accuracy:", acc)

# -----------------------------
# PLOT GRAPH
# -----------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()

# -----------------------------
# FILTER VISUALIZATION
# -----------------------------
filters, biases = model.layers[0].get_weights()

print("\n🔍 Showing Filters...")
for i in range(6):
    f = filters[:, :, :, i]
    plt.imshow(f[:, :, 0], cmap='gray')
    plt.title(f'Filter {i}')
    plt.axis('off')
    plt.show()

# -----------------------------
# CAM (Class Activation Map)
# -----------------------------
def generate_cam(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128,128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    last_conv_layer = model.layers[4]  # last conv layer

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    plt.imshow(heatmap, cmap='jet')
    plt.title("CAM Heatmap 🔥")
    plt.axis('off')
    plt.show()

generate_cam(model, 'dataset/yes/Y1.jpg')   