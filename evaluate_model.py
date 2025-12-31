import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# 1. LOAD MODEL
# -------------------------
model = load_model("model/deepfake_detector.h5")

# -------------------------
# 2. LOAD TEST DATASET
# -------------------------
test_dir = "dataset/test"

img_height = 128
img_width = 128
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),   # IMPORTANT: same as training
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# -------------------------
# 3. PREDICT
# -------------------------
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int)

# -------------------------
# 4. CONFUSION MATRIX
# -------------------------
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# -------------------------
# 5. CLASSIFICATION REPORT
# -------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# -------------------------
# 6. ROC CURVE
# -------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %.4f" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# -------------------------
# 7. PRECISION–RECALL CURVE
# -------------------------
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.show()
