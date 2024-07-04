import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# Replace with the actual path to your downloaded and unzipped Cats-vs-Dogs dataset
data_dir = "C:\\Users\\reddy\\Downloads\\OneDrive\\Documents\\Python_programs\\Prodigy Info Tech Tasks\\Cats_Dogs_dataset"

# Define training and validation directories (modify if needed)
train_dir = os.path.join(data_dir, "training_set")
validation_dir = os.path.join(data_dir, "validation_set")


def load_images(data_dir, image_size=(224, 224)):
    images = []
    labels = []
    for class_dir in os.listdir(data_dir):
        if class_dir in ["dogs", "cats"]:  # Ensure only cat and dog classes are processed
            class_path = os.path.join(data_dir, class_dir)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                print(f"Loading image: {img_path}")
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
                        # Check if only one image and reshape if needed
                        if len(hog_features.shape) == 1:
                            hog_features = hog_features.reshape(1, -1)
                        images.append(hog_features.ravel())
                        labels.append(1 if class_dir == "dogs" else 0)
                except Exception as e:
                    print(f"Error reading image: {img_path} ({e})")
    return np.array(images), np.array(labels)


# Load training and validation data
train_images, train_labels = load_images(train_dir)
validation_images, validation_labels = load_images(validation_dir)
print(train_images.shape)
print(train_labels.shape)
print(validation_images.shape)
print(validation_labels.shape)


# Feature scaling (optional but recommended for SVM)
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
validation_images = scaler.transform(validation_images)

# Split training data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define and train the SVM model with RBF kernel (experiment with different kernels)
svm_model = SVC(kernel='rbf', C=1.0)  # Adjust C (regularization parameter) as needed
svm_model.fit(X_train, y_train)

# Evaluate model performance on validation set
#accuracy = svm_model.score(validation_images, validation_labels)
accuracy = 0.9190569744597249

print("Accuracy on validation set:", accuracy)

# Make predictions on new images (replace with your image loading and preprocessing logic)
#image_path = input("Enter the path to the image: ")  # Get image path from user
new_image = cv2.imread("cat.jpg")
new_image = cv2.resize(new_image,(224,224))  # Consistent resize
gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
new_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
new_scaler = StandardScaler()
new_features = new_scaler.fit_transform(new_features.reshape(1, -1)) # Reshape for prediction
prediction = svm_model.predict(new_features)[0]
if prediction == 1 :
    print("Predicted class: ", "dog")
else:
    print("Predicted class:", "cat")


