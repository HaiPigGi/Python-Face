import cv2

# Load the cascade classifiers for face, eyes, and smile
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Load the image
image = cv2.imread('person_0093.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Define color codes for rectangles and text
face_color = (255, 0, 0)  # Blue color for face detection
eye_color = (0, 255, 0)  # Green color for eye detection
smile_color = (0, 0, 255)  # Red color for smile detection

# Draw rectangles around the faces and detect eyes and smiles
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), face_color, 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, 2)
    
    # Detect smiles within the face region
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), smile_color, 2)

# Add text labels for each detection
cv2.putText(image, 'Face', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, face_color, 2)
cv2.putText(image, 'Eye', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, eye_color, 2)
cv2.putText(image, 'Smile', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, smile_color, 2)

# Display the image with detections
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
