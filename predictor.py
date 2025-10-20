import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils import class_weight

# ------------------------------
# 1️⃣ Dataset Configuration
# ------------------------------
dataset_dir = r"C:\Users\pranav.ghuge\Desktop\Safas\SAFAS\thermal images UL"
img_height, img_width = 128, 128
batch_size = 32

# ------------------------------
# 2️⃣ Load Dataset (train/test split)
# ------------------------------
train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Detected classes:", class_names)

# ------------------------------
# 3️⃣ Compute Class Weights to Handle Imbalance
# ------------------------------
# Flatten labels to compute class weights
train_labels = np.concatenate([y for x, y in train_ds], axis=0)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)

# ------------------------------
# 4️⃣ Data Augmentation
# ------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ------------------------------
# 5️⃣ Optimize Dataset Performance
# ------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ------------------------------
# 6️⃣ Build CNN Model
# ------------------------------
model = models.Sequential([
    data_augmentation,  # Apply augmentation
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------
# 7️⃣ Model Checkpoint
# ------------------------------
os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint(
    filepath="models/thermal_paddy_disease_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# ------------------------------
# 8️⃣ Train the Model with Class Weights
# ------------------------------
epochs = 25  # you can increase if GPU available
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint],
    class_weight=class_weights
)

# ------------------------------
# 9️⃣ Evaluate Best Model
# ------------------------------
best_model = tf.keras.models.load_model("models/thermal_paddy_disease_model.keras")
loss, acc = best_model.evaluate(val_ds)
print(f"\n✅ Best Model Accuracy: {acc*100:.2f}%")

# ------------------------------
# 🔟 Predict New Image
# ------------------------------
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)  # add batch dimension

    predictions = best_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[tf.argmax(score)]
    print(f"Predicted: {predicted_class} ({100 * tf.reduce_max(score):.2f}% confidence)")

# Example usage
predict_image(r"C:\Users\pranav.ghuge\Desktop\Safas\SAFAS\thermal images UL\Blast\Thermalimage1a.jpg")
