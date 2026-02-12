import cv2
from ultralytics import YOLO

# 1. Load your custom trained model
model_path = 'runs/detect/crop_weed_model/weights/best.pt'
model = YOLO(model_path)

# 2. Initialize Webcam (0 is usually the default laptop cam)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting Webcam... Press 'q' to exit.")

while True:
    # Read a frame from the camera
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 inference on the frame
        # conf=0.5 means only show detections with 50% or higher confidence
        results = model(frame, conf=0.5)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow("Real-time Crop/Weed Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()