import tkinter as tk
from tkinter import ttk
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

# Extract HOG features
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (64, 128))  # Resize image to compatible dimensions
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(resized_image)
    hog_features = hog_features.flatten()
    return hog_features


# Extract histogram features
def extract_histogram_features(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_features = hist.flatten()
    return hist_features

# Extract vector features (concatenation of GLCM, HOG, and histogram features)
def extract_vector_features(image):
    glcm_features = extract_glcm_features(image)
    hog_features = extract_hog_features(image)
    histogram_features = extract_histogram_features(image)
    vector_features = np.concatenate((glcm_features, hog_features, histogram_features))
    return vector_features

# Load the dataset and extract features
dataset_path = '_test/'
cat_folder_path = dataset_path + 'cat/'
dog_folder_path = dataset_path + 'dog/'
image_paths_cats = [cat_folder_path + 'cat_{}.jpg'.format(str(i).zfill(4)) for i in range(71, 80)]
image_paths_dogs = [dog_folder_path + 'dog_{}.jpg'.format(str(i).zfill(4)) for i in range(45, 70)]
images_cats = [cv2.imread(image_path) for image_path in image_paths_cats]
images_dogs = [cv2.imread(image_path) for image_path in image_paths_dogs]
# Combine dogs and cats images and labels
images = images_dogs + images_cats
labels = ['dog'] * len(images_dogs) + ['cat'] * len(images_cats)

# Initialize feature vectors and corresponding labels for each extraction method
glcm_features = []
hog_features = []
histogram_features = []
vector_features = []

# Extract features for each image
for image in images:
    glcm_feature = extract_glcm_features(image)
    hog_feature = extract_hog_features(image)
    histogram_feature = extract_histogram_features(image)
    vector_feature = extract_vector_features(image)
    
    glcm_features.append(glcm_feature)
    hog_features.append(hog_feature)
    histogram_features.append(histogram_feature)
    vector_features.append(vector_feature)

# Convert features to numpy arrays
glcm_features = np.array(glcm_features)
hog_features = np.array(hog_features)
histogram_features = np.array(histogram_features)
vector_features = np.array(vector_features)

# Split the dataset into training and testing sets for each extraction method
X_train_glcm, X_test_glcm, y_train_glcm, y_test_glcm = train_test_split(glcm_features, labels, test_size=0.25, random_state=42)
X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(hog_features, labels, test_size=0.25, random_state=42)
X_train_histogram, X_test_histogram, y_train_histogram, y_test_histogram = train_test_split(histogram_features, labels, test_size=0.25, random_state=42)
X_train_vector, X_test_vector, y_train_vector, y_test_vector = train_test_split(vector_features, labels, test_size=0.25, random_state=42)

# KNN classification for each extraction method
knn_classifier_glcm = KNeighborsClassifier(n_neighbors=3)
knn_classifier_glcm.fit(X_train_glcm, y_train_glcm)
knn_accuracy_glcm = knn_classifier_glcm.score(X_test_glcm, y_test_glcm)

knn_classifier_hog = KNeighborsClassifier(n_neighbors=3)
knn_classifier_hog.fit(X_train_hog, y_train_hog)
knn_accuracy_hog = knn_classifier_hog.score(X_test_hog, y_test_hog)

knn_classifier_histogram = KNeighborsClassifier(n_neighbors=3)
knn_classifier_histogram.fit(X_train_histogram, y_train_histogram)
knn_accuracy_histogram = knn_classifier_histogram.score(X_test_histogram, y_test_histogram)

knn_classifier_vector = KNeighborsClassifier(n_neighbors=3)
knn_classifier_vector.fit(X_train_vector, y_train_vector)
knn_accuracy_vector = knn_classifier_vector.score(X_test_vector, y_test_vector)

# SVM classification for each extraction method
svm_classifier_glcm = svm.SVC()
svm_classifier_glcm.fit(X_train_glcm, y_train_glcm)
svm_accuracy_glcm = svm_classifier_glcm.score(X_test_glcm, y_test_glcm)

svm_classifier_hog = svm.SVC()
svm_classifier_hog.fit(X_train_hog, y_train_hog)
svm_accuracy_hog = svm_classifier_hog.score(X_test_hog, y_test_hog)

svm_classifier_histogram = svm.SVC()
svm_classifier_histogram.fit(X_train_histogram, y_train_histogram)
svm_accuracy_histogram = svm_classifier_histogram.score(X_test_histogram, y_test_histogram)

svm_classifier_vector = svm.SVC()
svm_classifier_vector.fit(X_train_vector, y_train_vector)
svm_accuracy_vector = svm_classifier_vector.score(X_test_vector, y_test_vector)

# Create the GUI window
window = tk.Tk()
window.title("Classification Results")

# Create a Treeview widget to display the results
tree = ttk.Treeview(window)
tree["columns"] = ("Extractor", "KNN Accuracy", "SVM Accuracy")
tree.heading("#0", text="No.")
tree.heading("Extractor", text="Extractor")
tree.heading("KNN Accuracy", text="KNN Accuracy")
tree.heading("SVM Accuracy", text="SVM Accuracy")

# Insert GLCM results
glcm_index = tree.insert("", "end", text="1", values=("GLCM", knn_accuracy_glcm, svm_accuracy_glcm))

# Insert HOG results
hog_index = tree.insert("", "end", text="2", values=("HOG", knn_accuracy_hog, svm_accuracy_hog))

# Insert histogram results
histogram_index = tree.insert("", "end", text="3", values=("Histogram", knn_accuracy_histogram, svm_accuracy_histogram))

# Insert vector results
vector_index = tree.insert("", "end", text="4", values=("Vector", knn_accuracy_vector, svm_accuracy_vector))

# Configure Treeview column widths
tree.column("#0", width=50)
tree.column("Extractor", width=100)
tree.column("KNN Accuracy", width=100)
tree.column("SVM Accuracy", width=100)

# Pack the Treeview widget
tree.pack()

# Start the GUI event loop
window.mainloop()
