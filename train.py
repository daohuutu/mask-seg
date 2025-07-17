from ultralytics import YOLO
import torch

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Thiết bị đang sử dụng: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO("yolov8m-seg.pt")

    model.train(
        data="D:\\VS code\\thigiacmaytinh\\BTL\\mask-seg\\data.yaml",
        epochs=50,
        imgsz=1290,       
        batch=2,         
        workers=0,      
        device=device    
    )

if __name__ == "__main__":
    main()
