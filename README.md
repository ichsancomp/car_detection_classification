# 🚗 Car Detection and Classification with PyTorch

This project implements a complete pipeline to detect and classify cars using deep learning. The system uses:

- **SSD MobileNet V3** for car detection (pretrained on COCO)
- **ResNet-50** fine-tuned on a custom dataset for car type classification

---

## 🔧 Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Evaluation

After training, the script will output:

- Validation Accuracy
- Classification Report (Precision, Recall, F1)
- Confusion Matrix (Visual)

---

## 🎥 Run Inference on Video

To run detection and classification on a video:

```bash
python3 run_demo.py	--mode train #Trains and evaluates model.
python3 run_demo.py	--mode video #Runs detection on video using trained model.

```



Output:
- Annotated video with bounding boxes and labels: `output_with_labels.avi`

---




---

## 🧠 Classes Used (Example)

```python
['MPV', 'Mobil Pickup', 'Off-Road', 'SUV', 'Sedan', 'Truck']
```

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- Custom vehicle dataset from Indonesian traffic scenes
