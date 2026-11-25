
import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET_PATH = "dataset"

X = []
y = []

print("Loading skin disease images...")

for disease in os.listdir(DATASET_PATH):
    disease_folder = os.path.join(DATASET_PATH, disease)

    if not os.path.isdir(disease_folder):
        continue

    for filename in os.listdir(disease_folder):
        image_path = os.path.join(disease_folder, filename)

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((128, 128))

            features = hog(
                np.array(image),
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                channel_axis=-1
            )

            X.append(features)
            y.append(disease)

        except Exception as error:
            print("Skipped:", image_path, error)

if len(X) < 4 or len(set(y)) < 2:
    print("Error: Add at least 2 disease categories with images.")
    print("Add at least 2 images in each category.")
    raise SystemExit

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training AI model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Accuracy:", accuracy * 100, "%")
print(classification_report(y_test, predictions))

joblib.dump(model, "skin_disease_model.pkl")

print("Model saved successfully!")
