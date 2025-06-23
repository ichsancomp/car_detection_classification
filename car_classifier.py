import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


class CarClassifier:
    def __init__(self, classifier_model_path: str, class_names: list, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names

        # Load classification model
        self.classifier = torchvision.models.resnet50(weights=None)
        self.classifier.fc = torch.nn.Linear(self.classifier.fc.in_features, len(class_names))
        self.classifier.load_state_dict(torch.load(classifier_model_path, map_location=self.device))
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        # Load SSD detector
        self.detector = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        self.detector.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.COCO_CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            # ... add the rest if needed
        ]

    def detect_cars(self, image: Image.Image, threshold: float = 0.5):
        tensor = transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            predictions = self.detector(tensor)[0]

        boxes = []
        for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
            if label < len(self.COCO_CLASSES) and self.COCO_CLASSES[label] == 'car' and score > threshold:
                boxes.append(box.int().tolist())
        return boxes

    def classify_car(self, image: np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.classifier(tensor)
            _, pred = torch.max(output, 1)
        return self.class_names[pred.item()]
