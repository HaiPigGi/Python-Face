import cv2
import numpy as np
import mahotas.features.texture as texture

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Function to extract GLCM features using mahotas
def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = texture.haralick(gray)
    glcm_features = glcm.mean(axis=0)
    return glcm_features

# Load the dataset and extract features
dataset_path = '_test/'

cat_folder_path = dataset_path + 'cat/'
dog_folder_path = dataset_path + 'dog/'

image_paths_cats = [cat_folder_path + 'cat_{}.jpg'.format(str(i).zfill(4)) for i in range(71, 100)]
image_paths_dogs = [dog_folder_path + 'dog_{}.jpg'.format(str(i).zfill(4)) for i in range(45, 100)]

images_cats = [cv2.imread(image_path) for image_path in image_paths_cats]
images_dogs = [cv2.imread(image_path) for image_path in image_paths_dogs]

# Combine dogs and cats images and labels
images = images_dogs + images_cats
labels = ['dog'] * len(images_dogs) + ['cat'] * len(images_cats)

# Initialize feature vectors and corresponding labels
features = []
for image in images:
    glcm_features = extract_glcm_features(image)
    features.append(glcm_features)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# KNN classification
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
knn_accuracy = knn_classifier.score(X_test, y_test)

# SVM classification
svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)
svm_accuracy = svm_classifier.score(X_test, y_test)

# Print accuracy results
print("Accuracy:")
print("GLCM Features - KNN:", knn_accuracy)
print("GLCM Features - SVM:", svm_accuracy)
