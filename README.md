# 🧠 Brain Tumor MRI Classification with ResNet50 (Transfer Learning)

A deep learning pipeline that classifies brain MRI scans into **four categories** using a fine-tuned **ResNet50** pretrained on ImageNet. The project covers image cropping, noise removal, augmentation, transfer learning with partial fine-tuning, training control with callbacks, and a full evaluation with a classification report and confusion matrix.

> ⚠️ **Disclaimer:** this is an educational/research project. It is **not** a medical device and must not be used for diagnosis.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Tech Stack](#-tech-stack)
4. [Pipeline](#-pipeline)
5. [Preprocessing in Detail](#-preprocessing-in-detail)
6. [Data Splits & Augmentation](#-data-splits--augmentation)
7. [Model Architecture](#-model-architecture)
8. [Training Setup](#-training-setup)
9. [Results](#-results)
10. [Analysis: Why Validation and Test Scores Differ](#-analysis-why-validation-and-test-scores-differ)
11. [Known Issues & Future Work](#-known-issues--future-work)
12. [How to Run](#-how-to-run)
13. [Repository Structure](#-repository-structure)

---

## 🎯 Project Overview

Brain MRI classification is a standard medical-imaging benchmark. This notebook trains a **ResNet50** classifier to distinguish:

| Label | Class |
|:-----:|-------|
| 0 | Glioma tumor |
| 1 | Meningioma tumor |
| 2 | No tumor |
| 3 | Pituitary tumor |

Rather than training from scratch on a small medical dataset, the project uses **transfer learning**: ImageNet-pretrained convolutional features are reused, and the top 100 layers of ResNet50 are fine-tuned on MRI data.

**Headline result:** best validation accuracy of **94.25%**, but accuracy on the held-out test set is **74.87%**. This gap is the most important finding of the project and is analyzed in detail [below](#-analysis-why-validation-and-test-scores-differ).

---

## 📦 Dataset

The notebook expects an MRI dataset organized into class folders (the dataset itself is **not** included in this repository):

```
data/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

| Set | Images |
|-----|:------:|
| Training pool (before validation split) | 2,870 |
| Test | 394 |

Test set class counts (from the classification report): glioma 100, meningioma 115, no tumor 105, pituitary 74.

---

## 🛠 Tech Stack

- **Python**, **NumPy**
- **OpenCV (`cv2`)** and **imutils** for image processing
- **TensorFlow / Keras** for ResNet50, augmentation, training, and callbacks
- **scikit-learn** for shuffling, splitting, and evaluation metrics
- **Matplotlib** and **Seaborn** for visualization, **tqdm** for progress bars

---

## 🔄 Pipeline

```
Raw MRI → Crop brain region → Resize (200×200) → Denoise → Pseudocolor → Normalize
        → Shuffle → One-hot labels → Train/Val split → Augment
        → ResNet50 (fine-tune top 100 layers) → Callbacks → Evaluate on test set
```

---

## 🧹 Preprocessing in Detail

### Step 1: Crop away the background (`crop_img`)

MRI scans contain a large black background. Cropping to the brain region makes the network focus on relevant pixels. The `crop_img` function:

1. Converts the image to grayscale and applies a **3×3 Gaussian blur**
2. **Thresholds** at intensity 45 to create a binary mask
3. Applies **2 erosions + 2 dilations** to remove small noise regions
4. Finds the **largest external contour** (the brain)
5. Locates the **extreme points** (left, right, top, bottom) of that contour
6. **Crops** the rectangle defined by those points

Every training and testing image is cropped, resized to **200×200**, and written to `data/cropped/Training/...` and `data/cropped/Testing/...`. Unreadable files are skipped with a warning, and each image is wrapped in `try/except` so one bad file cannot stop the run. The notebook also visualizes the four cropping steps on a sample image (original → largest contour → extreme points → cropped).

### Step 2: Load, denoise, and colorize

For every cropped image:

| Operation | Details | Purpose |
|-----------|---------|---------|
| Read as grayscale | `cv2.imread(path, 0)` | MRI is inherently single-channel |
| Bilateral filter | `d=2, sigmaColor=50, sigmaSpace=50` | Smooths noise while preserving edges |
| Pseudocolor | `cv2.COLORMAP_BONE` | Converts to 3 channels, which ResNet50 requires |
| Resize | 200 × 200 | Consistent input size |

### Step 3: Normalize

Pixel values are divided by 255 to scale them to `[0, 1]`. Resulting array shapes:

- Training pool: `(2870, 200, 200, 3)`
- Test: `(394, 200, 200, 3)`

---

## ✂️ Data Splits & Augmentation

**Splitting:** the training pool is shuffled (`random_state=42`), labels are one-hot encoded, and 20% is held out for validation.

| Split | Images |
|-------|:------:|
| Train | 2,296 |
| Validation | 574 |
| Test | 394 |

**Augmentation** (training data only, via `ImageDataGenerator`):

| Transform | Setting |
|-----------|---------|
| Rotation | up to ±10° |
| Width shift | 5% |
| Height shift | 5% |
| Horizontal flip | enabled |

These are mild transforms, chosen to add variety without distorting anatomy.

---

## 🏗 Model Architecture

**Base:** `ResNet50(include_top=False, weights='imagenet', input_shape=(200, 200, 3))`

**Partial fine-tuning:** all layers except the **last 100** are frozen. The last 100 layers are trainable.

**Custom classification head:**

```
ResNet50 features
  → GlobalAveragePooling2D
  → Dropout(0.4)
  → Dense(4, softmax)
```

| Parameters | Count |
|------------|------:|
| Total | 23,595,908 |
| Trainable | 22,160,132 |
| Non-trainable | 1,435,776 |

Global average pooling replaces the flatten + large dense layers of older CNNs, which keeps the head small. Dropout (0.4) regularizes the head.

---

## ⚙️ Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | Adam, initial learning rate `5e-4` |
| Loss | Categorical cross-entropy |
| Metric | Accuracy |
| Batch size | 32 |
| Max epochs | 30 |
| Training data | Augmented stream from `datagen.flow` |

**Callbacks:**

- **`EarlyStopping`**: monitors `val_loss`, patience 10, `restore_best_weights=True`
- **`ReduceLROnPlateau`**: monitors `val_loss`, halves the learning rate after 3 epochs without improvement, minimum `1e-5`

All 30 epochs ran (early stopping did not trigger). Because `restore_best_weights=True`, the model was restored to **epoch 21**, which had the lowest validation loss. Training took roughly 220–320 seconds per epoch on the machine used (about 2 hours in total).

### Learning-rate schedule that actually occurred

| Epochs | Learning rate |
|:------:|:-------------:|
| 1–7 | 5.0e-4 |
| 8–11 | 2.5e-4 |
| 12–20 | 1.25e-4 |
| 21–24 | 6.25e-5 |
| 25–27 | 3.125e-5 |
| 28–30 | 1.5625e-5 |

---

## 📊 Results

### Training progress (selected epochs)

| Epoch | Train Acc | Val Acc | Val Loss | LR |
|:-----:|:---------:|:-------:|:--------:|:--:|
| 1 | 42.68% | 32.40% | 5.1641 | 5.0e-4 |
| 2 | 55.23% | 38.50% | 5.7375 | 5.0e-4 |
| 4 | 69.43% | 65.85% | 1.2237 | 5.0e-4 |
| 8 | 85.28% | 84.84% | 0.3903 | 2.5e-4 |
| 12 | 91.81% | 91.81% | 0.2489 | 1.25e-4 |
| 17 | 94.43% | 91.46% | 0.2193 | 1.25e-4 |
| **21 ⭐ (best)** | 97.21% | **94.25%** | **0.1863** | 6.25e-5 |
| 25 | 98.26% | 93.73% | 0.2111 | 3.125e-5 |
| 30 | 99.00% | 94.08% | 0.1934 | 1.5625e-5 |

Early epochs are unstable: validation loss is very high (above 5) while the unfrozen layers adapt, and it only settles once the learning rate is halved a couple of times.

### Summary

| Dataset | Accuracy | Loss |
|---------|:--------:|:----:|
| Validation (best epoch, 21) | **94.25%** | 0.1863 |
| **Test (394 images)** | **74.87%** | 1.8651 |

### Test-set classification report

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Glioma | 0.90 | **0.28** | 0.43 | 100 |
| Meningioma | 0.67 | 0.97 | 0.79 | 115 |
| No tumor | 0.72 | 0.99 | 0.83 | 105 |
| Pituitary | 0.98 | 0.70 | 0.82 | 74 |
| **Macro avg** | 0.82 | 0.73 | 0.72 | 394 |
| **Weighted avg** | 0.80 | 0.75 | 0.72 | 394 |

### Test-set confusion matrix

Rows are true labels and columns are predictions.

| True \ Predicted | Glioma | Meningioma | No tumor | Pituitary |
|------------------|:------:|:----------:|:--------:|:---------:|
| **Glioma** (100) | **28** | 45 | 26 | 1 |
| **Meningioma** (115) | 2 | **111** | 2 | 0 |
| **No tumor** (105) | 1 | 0 | **104** | 0 |
| **Pituitary** (74) | 0 | 9 | 13 | **52** |

295 of 394 test images are classified correctly (74.87%).

---

## 🔬 Analysis: Why Validation and Test Scores Differ

The gap between **94.25% validation** accuracy and **74.87% test** accuracy is large, and the confusion matrix shows where it comes from: **72 of the 100 glioma images are misclassified** (45 as meningioma, 26 as "no tumor"). The model is also biased toward predicting meningioma and "no tumor" on the test set, which is why those classes have high recall but lower precision.

Possible causes, in rough order of likelihood. These are **hypotheses to test**, not confirmed findings:

1. **Optimistic validation set.** Validation images are a random 20% of the training pool. MRI datasets often contain several slices per patient, so near-identical images can end up in both training and validation. Validation accuracy then measures memorization as much as generalization. A patient-level split would give a more honest validation score.
2. **Distribution shift between Training and Testing folders.** The two folders may differ in scanners, slice orientation, or patient population, so the learned features may not transfer cleanly.
3. **Preprocessing mismatch with ImageNet weights.** Keras ResNet50 expects inputs processed with `tensorflow.keras.applications.resnet.preprocess_input` (channel-mean subtraction on a 0–255 scale). This notebook only divides by 255, so the pretrained features see a different input distribution than they were trained on.
4. **Batch normalization while fine-tuning.** With the top 100 layers unfrozen (including BatchNorm layers) and a batch size of 32, BatchNorm statistics can drift on a small dataset. This may explain the very unstable early validation loss.
5. **Overfitting.** Training accuracy reaches 99.0%, so the model fits the training data almost perfectly.

---

## ⚠️ Known Issues & Future Work

**Issues found while reviewing the notebook:**

- **Inference preprocessing does not match training.** The final single-image prediction cell loads the image with `load_img` and divides by 255 only. Training images went through grayscale → bilateral filter → bone colormap. On the sample glioma image this produced a wrong prediction (`no_tumor`, 55.06% confidence; glioma got 42.87%). Any inference or deployment code must reuse the exact training preprocessing.
- **Test set is the only reliable benchmark.** The README headline should be the 74.87% test accuracy, not the validation score.
- **Model saved in legacy HDF5 format** (`bestresnet.h5`). Keras recommends the native `.keras` format.
- **Tidiness:** the last four code cells of the notebook are empty and can be deleted.

**Suggested improvements:**

- Split by **patient** (or at least de-duplicate near-identical slices) before creating the validation set
- Use `resnet.preprocess_input` instead of dividing by 255
- Freeze BatchNorm layers during fine-tuning, or use a **two-stage** schedule (train the head first, then unfreeze with a low learning rate)
- Address the glioma recall problem: class weights, more augmentation, or a loss focused on hard examples
- Add **Grad-CAM** to check whether the model looks at the tumor region
- Compare against other backbones (EfficientNet, DenseNet) under the same split
- Report per-class metrics on validation as well as test, and average over multiple seeds

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Sarthakm811/Brain-Tumor-MRI-ResNet50.git
cd Brain-Tumor-MRI-ResNet50
```

### 2. Install dependencies
```bash
pip install numpy opencv-python imutils matplotlib seaborn scikit-learn tqdm tensorflow
```

### 3. Add the dataset
Download an MRI dataset with the four classes and place it in the `data/Training` and `data/Testing` structure shown [above](#-dataset). The class folder names must match: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`.

### 4. Run the notebook
```bash
jupyter notebook Train_model_resnet.ipynb
```
Run the cells in order: cropping creates `data/cropped/`, training saves `bestresnet.h5`.

> **Compute note:** training the full 30 epochs took about 2 hours on the author's machine (roughly 3 seconds per step). A GPU runtime such as Google Colab will be much faster.

---

## 📁 Repository Structure

```
Brain-Tumor-MRI-ResNet50/
├── Train_model_resnet.ipynb   # Preprocessing, training, evaluation
├── README.md                  # Project documentation
├── data/                      # (not included) Training/ and Testing/ folders
└── bestresnet.h5              # (generated) saved model after training
```

---

## 👤 Author

**Sarthak Mahajan**
GitHub: [@Sarthakm811](https://github.com/Sarthakm811)

****
