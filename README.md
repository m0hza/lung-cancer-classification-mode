# 🦥 Lung Cancer Stage Classification - Inference Only

> Minimal inference pipeline using a pretrained CNN model for classifying lung cancer stages (normal, benign, malignant) from chest scan images.

---

## 🚀 Overview

This repository contains **only the final trained model and an inference notebook**. It is intended for generating predictions from a set of test images in a Kaggle-style submission format.

---

## 📂 Project Structure

```
├── lung_cancer_cnn.pth       # saved model weights (uploaded)
├── inference.ipynb           # minimal inference script
└── README.md

# ⚠️ Note:
> The inference notebook includes a placeholder: `# link to your test dataset`. Mount your own test image directory or update the code accordingly.
```

---

## 🔧 Setup

> ⚠️ Recommended: Run this in Google Colab for easy Drive access and GPU support.

1. **Clone the repo** or upload the two files to Colab.
2. **Mount Google Drive**:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Install dependencies**:

```bash
!pip install -q torch torchvision matplotlib pandas
```

---

## 🔍 Inference

> You must have your test images available (in `/test/` or any path you mount). Update the inference notebook to reflect your directory structure.

```python
model = SimpleCNN()
model.load_state_dict(torch.load("/content/drive/MyDrive/lung_cancer_cnn.pth"))
model.eval()

# Load test images from your dataset
# Replace this line in the notebook:
# test_dataset = LungCancerDataset("/path/to/your/test", transform=transform, mode='test')

# Run inference
submission_df.to_csv("submission.csv", index=False)
```

**Submission format:**

```csv
id,label
img_0001,2
img_0002,1
...
```

---

## 📅 Credits

Model and pipeline created for a medical imaging classification challenge using PyTorch.

---

## 🌐 License

MIT License — use, share, or modify as you wish.

---

## ✨ Contributions

If you adapt the model to new datasets or improve the inference logic, feel free to open a pull request.
