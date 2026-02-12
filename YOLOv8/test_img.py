import cv2
import os
import random
from ultralytics import YOLO

# 1. Load your custom model
model_path = 'runs/detect/crop_weed_model/weights/best.pt'
model = YOLO(model_path)

# 2. Path to your validation images
val_images_path = "datasets/crop_weed_data/val/images"

# Get list of all image files
files = os.listdir(val_images_path)
images = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print(f"No images found in {val_images_path}")
    exit()

print("Controls:\n  Press 'n' -> Next Image\n  Press 'q' -> Quit")

# 3. Loop to show random images
while True:
    # Pick a random image from the list
    selected_file = random.choice(images)
    image_path = os.path.join(val_images_path, selected_file)
    
    # Run prediction
    results = model(image_path)
    
    # Plot the results (draw bounding boxes)
    annotated_frame = results[0].plot()
    
    # Resize slightly if the image is too big for your screen (optional)
    annotated_frame = cv2.resize(annotated_frame, (800, 800))

    # Show the image
    cv2.imshow("Crop vs Weed Detection (Press 'n' for next)", annotated_frame)

    # Wait for key press
    key = cv2.waitKey(0) & 0xFF
    
    # If 'q' is pressed, break loop
    if key == ord('q'):
        break
    # If 'n' is pressed, the loop continues and picks a new random image
    elif key == ord('n'):
        continue

cv2.destroyAllWindows()