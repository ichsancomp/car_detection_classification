import argparse
import torch
from car_classifier import CarClassifier
from car_trainer import train_car_classifier, evaluate_model, test_on_image
from PIL import Image
import cv2
import time
import os
CLASS_NAMES=['Car Pickup', 'Carry Car', 'City Car', 'MPV', 'Off-Road', 'SUV', 'Sedan', 'Truck']
def run_training():
    model, class_names, val_loader, transform = train_car_classifier(
        dataset_path="Indonesia",
        model_arch="resnet50",
        num_epochs=15,
        batch_size=16,
        lr=1e-4,
        output_path="car_classifier_indonesia_new.pth"
    )
    return model, class_names, val_loader, transform

#'Car Pickup', 'Carry Car', 'City Car', 'MPV', 'Off-Road', 'SUV', 'Sedan', 'Truck'
def run_video_demo(model_path):
    car_system = CarClassifier(
        classifier_model_path=model_path,
        class_names=CLASS_NAMES
    )

    cap = cv2.VideoCapture("traffic_test.mp4")
    out = cv2.VideoWriter('output_with_labels.avi',
                          cv2.VideoWriter_fourcc(*'XVID'),
                          30,
                          (int(cap.get(3)), int(cap.get(4))))

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = car_system.detect_cars(pil_image)

        for box in boxes:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            label = car_system.classify_car(crop)
            fps = 1.0 / (time.time() - start_time)
            start_time = time.time()

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow("Car Detection and Classification", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Car Classifier System")
    parser.add_argument('--mode', choices=['train', 'video'], required=True, help="Mode: train or video")
    parser.add_argument('--model', type=str, default='car_classifier_indonesia.pth', help="Path to model weights")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        model, class_names, val_loader, transform = run_training()

        evaluate_model(model=model, val_loader=val_loader, class_names=class_names, device=device)

        test_image = "Indonesia/Carry Car/images (1).jpeg"
        test_on_image(model=model, class_names=class_names, transform=transform, image_path=test_image, device=device)


    elif args.mode == 'video':
        run_video_demo(args.model)
        
    

if __name__ == "__main__":
    main()
