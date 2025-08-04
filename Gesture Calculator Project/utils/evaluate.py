import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# === Load the model ===
model = load_model("model/model.h5")
print("✅ Model loaded successfully!")

# === Load MNIST test data ===
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# === Evaluate model on test set ===
loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n📊 Test Accuracy: {accuracy*100:.2f}%")

# === Predictions for metrics ===
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# === Classification Report ===
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred_classes, digits=4))

# === Confusion Matrix ===
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))
