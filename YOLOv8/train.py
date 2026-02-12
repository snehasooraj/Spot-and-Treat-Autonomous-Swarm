from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model

    # Train the model
    # device=0 uses your RTX 4050 GPU
    results = model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=512, 
        batch=16, 
        device=0, 
        name='crop_weed_model'
    )

if __name__ == '__main__':
    main()