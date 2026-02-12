import cv2
import os
from ultralytics import YOLO

# 1. Load your trained model
# Make sure this path is correct relative to where you are running the command
model = YOLO('runs/detect/crop_weed_model/weights/best.pt')

# 2. Automatically find an image in the val folder
val_images_path = "datasets/crop_weed_data/val/images"

# Get a list of all files in that folder
files = os.listdir(val_images_path)

# Filter for jpg/jpeg images just in case
images = [f for f in files if f.endswith('.jpg') or f.endswith('.jpeg')]

if len(images) > 0:
    # Pick the first image found
    image_name = images[0]
    image_path = os.path.join(val_images_path, image_name)
    print(f"Testing on image: {image_path}")

    # 3. Run inference
    results = model(image_path)

    # 4. Plot the results
    annotated_frame = results[0].plot()

    # 5. Display
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # Wait for key press
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"No images found in {val_images_path}. Check your folder structure!")