import cv2

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("smile.xml")

# Initialize the video capture
b = cv2.VideoCapture(0)

while True:
    c_rec, d_image = b.read()

    # Convert the image to grayscale for detection
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    f = face_cascade.detectMultiScale(e, 1.3, 6)

    for (x1, y1, w1, h1) in f:
        # Draw rectangle around detected face
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)
        roi_gray = e[y1:y1 + h1, x1:x1 + w1]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Calculate face confidence based on the number of eyes detected
        if len(eyes) == 0:
            face_confidence = -50  # Negative confidence if no eyes are detected
        else:
            face_confidence = min(100, len(eyes) * 25)  # Scale the number of eyes detected to a max of 100%

        # Display face confidence level on the image
        cv2.putText(d_image, f"Face Confidence: {face_confidence}%", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Draw rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(d_image, (x1 + ex, y1 + ey), (x1 + ex + ew, y1 + ey + eh), (0, 255, 0), 2)

        # Detect smiles within the face region
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

        # If smile(s) are detected, calculate smile confidence
        for (sx, sy, sw, sh) in smiles:
            # Draw rectangle around detected smile
            cv2.rectangle(d_image, (x1 + sx, y1 + sy), (x1 + sx + sw, y1 + sy + sh), (0, 255, 255), 2)

            # Estimate smile confidence (based on detected area size)
            smile_confidence = (sw * sh) / (w1 * h1)*100 # Ratio of smile area to face area, scaled to percentage

            # Display smile confidence level on the image
            cv2.putText(d_image, f"Smile Confidence: {smile_confidence:.2f}%", 
                        (x1 + sx, y1 + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Display the resulting image
    cv2.imshow('img', d_image)

    # Exit when the user presses the 'Esc' key
    h = cv2.waitKey(40) & 0xff
    if h == 27:
        break

# Release the capture and close all OpenCV windows
b.release()
cv2.destroyAllWindows()

    
