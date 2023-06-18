import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk

# Load the images
image_paths = ['person_0085.jpg', 'person_0091.jpg', 'person_0093.jpg', 'person_0099.jpg', 'person_0106.jpg']

# Initialize a list to store the histograms
histograms = []

# Calculate histograms for each image
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    
    # Add the histogram to the list
    histograms.append(hist)

# Calculate the distances between histograms
distances = np.zeros((len(histograms), len(histograms)))

for i in range(len(histograms)):
    for j in range(len(histograms)):
        if i != j:
            # Calculate the histogram distance using Bhattacharyya coefficient
            distance = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_BHATTACHARYYA)
            distances[i][j] = distance

# Create a pandas DataFrame for the distances
columns = [str(i + 1) for i in range(len(distances))]
df = pd.DataFrame(distances, columns=columns, index=columns)

# Create a Tkinter window
window = tk.Tk()
window.title("Distance Table (Histograms)")

# Create a ttk Treeview widget
tree = ttk.Treeview(window)
tree["columns"] = columns

# Configure the Treeview columns
tree.heading("#0", text="Object")
for column in columns:
    tree.heading(column, text=column)

# Insert data into the Treeview
for i, row in enumerate(distances):
    item_id = tree.insert("", i, text=str(i + 1))
    for j, distance in enumerate(row):
        tree.set(item_id, column=columns[j], value=str(distance))

# Pack the Treeview widget
tree.pack()

# Start the Tkinter event loop
window.mainloop()
