import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import numpy as np
import torchvision
import cv2


def train_car_classifier(
    dataset_path: str,
    model_arch: str = "resnet50",
    num_epochs: int = 0,
    batch_size: int = 16,
    lr: float = 1e-4,
    output_path: str = "car_classifier.pth",
    image_size: int = 224,
    device: torch.device = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=dataset_path, transform=transform)
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model_func = getattr(models, model_arch)
    model = model_func(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return model, class_names, val_loader, transform


def evaluate_model(model, val_loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    for i, class_name in enumerate(class_names):
        total = sum([1 for label in all_labels if label == i])
        correct = sum([1 for pred, label in zip(all_preds, all_labels) if pred == label and label == i])
        acc = 100 * correct / total if total > 0 else 0
        print(f"{class_name}: {acc:.2f}% accuracy")

    
    
   
def detect_cars(image, threshold=0.6):
    ssd = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    ssd.eval()
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    ]
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        output = ssd(tensor)[0]

    boxes = []
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if label < len(COCO_CLASSES) and COCO_CLASSES[label] == 'car' and score > threshold:
            boxes.append(box.int().tolist())
    return boxes

 
def test_on_image(model, class_names, transform, image_path, device):
    image = Image.open(image_path).convert("RGB")
    img_cv = np.array(image)[:, :, ::-1].copy()
    boxes = detect_cars(image)

    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image.crop((x1, y1, x2, y2))
        crop_tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(crop_tensor)
            _, predicted = output.max(1)
            label = class_names[predicted]

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
