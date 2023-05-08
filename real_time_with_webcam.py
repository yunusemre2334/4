import cv2
from roboflow import Roboflow

# Initialize the Roboflow API (bu kısım size roboflow tarafından temin edilecektir kendi kod bloklarınızı bu apı kısmına yazın benimkleri silerek.)
rf = Roboflow(api_key="apiyi girin")

# Load the model from Roboflow
project = rf.workspace().project("proje adını girin")
model = project.version(2).model

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Make predictions with the model
    predictions = model.predict(frame, confidence=40, overlap=30).json()

    # Display the predictions on the frame
    for result in predictions['predictions']:
        if 'xmin' in result:
            xmin = result['xmin']
            ymin = result['ymin']
            xmax = result['xmax']
            ymax = result['ymax']
            label = result['label']
            confidence = result['confidence']

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
