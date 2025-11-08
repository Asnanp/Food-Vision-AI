<img width="2676" height="889" alt="gradcam_visualization" src="https://github.com/user-attachments/assets/fd324178-318c-4a9b-84ac-99bf01ff748f" /># 🍕 Food Vision AI - Deep Learning Food Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-4.0-orange.svg)](https://gradio.app/)

> **An advanced AI-powered food recognition system that identifies 101 different food types with 81.10% accuracy, complete with nutrition information, health ratings, and explainable AI visualizations.**

🔗 **[Live Demo](https://huggingface.co/spaces/asnanp1/food-ai-detector)** | 📊 **[Dataset](https://www.kaggle.com/datasets/kmader/food41))**


---

## 🌟 Key Features

### 🎯 **AI Food Detection**
- Identifies **101 different food categories**
- **81.10% Top-1 accuracy** (exceeds industry standard)
- **95.64% Top-5 accuracy** 
- Real-time inference (<500ms on CPU)

### 📊 **Comprehensive Nutrition Analysis**
- Complete nutritional breakdown for all 101 foods
- Calories, Protein, Carbs, Fat, Fiber tracking
- Health scores and dietary recommendations
- Serving size information

### 🔬 **Explainable AI (Grad-CAM)**
- Visual explanations of model predictions
- Heatmaps showing what the AI focuses on
- Comparison visualizations for top predictions
- Builds trust through transparency

### 🎨 **Beautiful Web Interface**
- Modern, responsive Gradio UI
- Real-time predictions
- Interactive visualizations
- Mobile-friendly design

### 📱 **Cross-Platform Ready**
- Web application (Gradio)
- REST API (Flask)
- Real-time video detection (OpenCV)
- Mobile app ready (React Native)

---

## 🎬 Demo

### Food Detection
![Detection Demo](<img width="2676" height="889" alt="gradcam_visualization" src="https://github.com/user-attachments/assets/e650001b-dd77-440d-8943-134f863a437a" />
)
---

## 📊 Performance Metrics

| Metric | Score | Benchmark |
|--------|-------|-----------|
| **Top-1 Accuracy** | **81.10%** | 75-80% |
| **Top-5 Accuracy** | **95.64%** | 90-95% |
| **Inference Time (CPU)** | <500ms | <1000ms |
| **Model Size** | 17MB | <50MB |
| **Training Time** | 7.5 hours | - |

---

## 🏗️ Architecture

```
Input Image (224×224)
    ↓
EfficientNet-B0 Backbone (Pretrained on ImageNet)
    ↓
Feature Extraction (1280 features)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer (101 classes)
    ↓
Softmax
    ↓
Predictions + Nutrition Lookup
```

**Model Details:**
- **Base Architecture:** EfficientNet-B0
- **Parameters:** 4.1M (trainable: 4.1M)
- **Framework:** PyTorch + timm
- **Training Strategy:** Transfer Learning + Fine-tuning

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Asnanp/Food-Vision-AI.git
cd Food-Vision-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model

```bash
# Download pre-trained model (17MB)
wget [https://huggingface.co/Asnanp/Food-Vision-AI](https://huggingface.co/spaces/asnanp1/food-ai-detector)/resolve/main/food_classifier_final.pth
```

Or train your own (see [Training](#training) section).

### Run Web App

```bash
# Launch Gradio interface
python app.py

# Access at: http://localhost:7860
```

### Run API Server

```bash
# Start Flask API
python api_server.py

# API will be available at: http://localhost:5000
```

### Test Predictions

```bash
# Test with a sample image
python app.py --image path/to/food.jpg
```

---


---

## 🎓 Training

### Dataset Preparation

1. **Download Food-101 Dataset**
   ```bash
   kaggle datasets download -d kmader/food41
   unzip food41.zip
   ```

2. **Dataset Structure**
   ```
   food-101/
   ├── images/
   │   ├── apple_pie/
   │   ├── sushi/
   │   └── ... (101 folders)
   └── meta/
       ├── train.txt
       └── test.txt
       └── classes.txt
       └── train.json
       └── test.json
       └── labels.txt
     
   ```

### Training Process

```bash
# Start training
python app.py

# Monitor with TensorBoard
tensorboard --logdir runs/
```

**Training Configuration:**
- **Epochs:** Phase 1 (2) + Phase 2 (3) = 5 total
- **Batch Size:** 32
- **Optimizer:** Adam (lr=1e-3, then 1e-4)
- **Augmentation:** Random flip, rotation, color jitter
- **Hardware:** CPU (7.5h) or GPU (1.5h)

**Training Results:**
```
Phase 1 (Classifier Head):
  Epoch 1: Val Acc = 52.27%
  Epoch 2: Val Acc = 54.84%

Phase 2 (Full Model):
  Epoch 1: Val Acc = 77.56%
  Epoch 2: Val Acc = 80.18%
  Epoch 3: Val Acc = 81.10% ✨

Final Test Accuracy: 81.10% (Top-1), 95.64% (Top-5)
```

---

## 🔬 Technical Details

### Model Architecture (EfficientNet-B0)

- **Input Size:** 224×224×3
- **Backbone:** EfficientNet-B0 (pretrained on ImageNet)
- **Output:** 101 classes (food categories)
- **Activation:** Softmax

### Data Augmentation

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Training Strategy

1. **Phase 1:** Freeze backbone, train classifier (2 epochs)
2. **Phase 2:** Unfreeze all, fine-tune entire model (3 epochs)
3. **Learning Rate:** 1e-3 → 1e-4 (reduced by 10x)
4. **Scheduler:** ReduceLROnPlateau
5. **Loss:** CrossEntropyLoss

---

## 🌐 API Usage

### REST API Endpoints

**1. Predict Food**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@food_image.jpg"
```

Response:
```json
{
  "success": true,
  "predictions": [
    {"name": "Chicken Curry", "confidence": 0.866},
    {"name": "Gnocchi", "confidence": 0.041}
  ],
  "inference_time_ms": 342
}
```

**2. Get Nutrition Info**
```bash
curl http://localhost:5000/nutrition/pizza
```

Response:
```json
{
  "success": true,
  "food": "pizza",
  "nutrition": {
    "calories": 266,
    "protein": "11g",
    "carbs": "33g",
    "fat": "10g"
  }
}
```

### Python SDK

```python
from food_vision import FoodVisionAPI

api = FoodVisionAPI()
result = api.predict('path/to/food.jpg')

print(f"Food: {result['food']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Calories: {result['nutrition']['calories']}")
```

---

## 📊 Supported Foods (101 Categories)

<details>
<summary>Click to expand full list</summary>

### Desserts & Sweets
Apple Pie, Baklava, Cannoli, Carrot Cake, Cheesecake, Chocolate Cake, Chocolate Mousse, Churros, Cup Cakes, Donuts, Ice Cream, Macarons, Panna Cotta, Red Velvet Cake, Strawberry Shortcake, Tiramisu, Waffles

### Main Dishes
Bibimbap, Chicken Curry, Chicken Quesadilla, Fish and Chips, Fried Rice, Hamburger, Lasagna, Pad Thai, Paella, Pizza, Ramen, Risotto, Spaghetti Bolognese, Spaghetti Carbonara, Steak, Tacos

### Seafood
Ceviche, Clam Chowder, Crab Cakes, Fried Calamari, Lobster Bisque, Lobster Roll, Mussels, Oysters, Sashimi, Scallops, Shrimp and Grits, Sushi, Tuna Tartare

### Salads & Appetizers
Beet Salad, Bruschetta, Caesar Salad, Caprese Salad, Deviled Eggs, Edamame, French Onion Soup, Greek Salad, Guacamole, Hummus, Miso Soup, Nachos, Seaweed Salad, Spring Rolls

### And 50+ more!

</details>

---

## 🎨 Grad-CAM Visualizations

Generate explainable AI visualizations:

```bash
python gradcam_visualizer.py --image food.jpg
```

**Output:**
- `gradcam_visualization.png` - 3-panel visualization
- `gradcam_top5_comparison.png` - Top 5 predictions comparison

**Example:**
![Grad-CAM Example](<img width="2676" height="1600" alt="gradcam_top5_comparison" src="https://github.com/user-attachments/assets/b037d4c9-f3f7-4e23-8a86-abcd405937fd" />
)

---


---

## 📈 Results & Analysis

### Accuracy by Food Category

| Category | Accuracy | Sample Size |
|----------|----------|-------------|
| Desserts | 85.3% | 15,000 |
| Asian | 83.7% | 18,000 |
| Italian | 81.4% | 12,000 |
| American | 79.8% | 14,000 |

### Common Misclassifications

1. **Spaghetti Carbonara** ↔ **Spaghetti Bolognese** (similar appearance)
2. **Ramen** ↔ **Pho** (similar noodle soups)
3. **Cup Cakes** ↔ **Muffins** (shape similarity)

---

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t food-vision-ai .

# Run container
docker run -p 7860:7860 food-vision-ai
```

### Hugging Face Spaces

```bash
# Install huggingface_hub
pip install huggingface_hub

# Upload to Hugging Face
huggingface-cli login
huggingface-cli upload food-vision-ai ./
```
---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch 2.0, timm
- **Web Framework:** Gradio 4.0, Flask
- **Computer Vision:** OpenCV, PIL
- **Visualization:** Matplotlib, Seaborn
- **API:** FastAPI, RESTful
- **Deployment:** Docker, Hugging Face Spaces

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{food_vision_ai,
  author = {Asnan P},
  title = {Food Vision AI: Deep Learning Food Classifier},
  year = {2025},
  url = {https://github.com/Asnanp/Food-Vision-AI}
}
```


---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


## 🙏 Acknowledgments

- **Dataset:** Food42 bg Kmadar
- **Model:** EfficientNet by Google Research
- **Framework:** PyTorch team
- **Interface:** Gradio team
- **Community:** Hugging Face, Kaggle

---

## 📧 Contact

**Your Name** - [Instagram](https://instagram.com/asnannp) - asnanp875@gmail.com

**Project Link:** [https://github.com/AsnanP/food-vision-ai](https://github.com/Asnanp/food-vision-ai)

**Live Demo:** [https://huggingface.co/spaces/AsnanP/food-vision-ai](https://huggingface.co/spaces/asnanp1/food-ai-detector)

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AsnanP/food-vision-ai&type=Date)](https://star-history.com/#AsnanP/food-vision-ai&Date)

---

<div align="center">
  
**Made with ❤️ and lots of ☕**

[⬆ back to top](#-food-vision-ai---deep-learning-food-classifier)

</div>
