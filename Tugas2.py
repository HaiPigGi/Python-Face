import cv2

# Load the smile cascade classifier
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Load a list of image paths
image_paths = ['person_0085.jpg', 'person_0091.jpg', 'person_0093.jpg', 'person_0099.jpg', 'person_0106.jpg']

# Initialize the counter for smiling people
smiling_people = 0

# Loop through the images and detect smiles
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect smiles in the image
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20)
    
    # Check if any smiles are found
    if len(smiles) > 0:
        smiling_people += 1
        
    # Draw rectangles around the smiles
    for (x, y, w, h) in smiles:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Display the image with smile detections
    cv2.imshow('Image', image)
    cv2.waitKey(0)

# Print the number of people smiling
print(f"Number of people smiling: {smiling_people}")
