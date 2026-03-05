Real-Time ASL Digit Recognition (0-9)

### Developed at Libyan International University (LIMU)
**Author:** Yasmina Ahmad Elmismary


This project implements a Real-Time American Sign Language (ASL) digit recognition system using **Deep Transfer Learning** with **MobileNetV2**.



## 🎯 The Problem (Gap Analysis)

Existing sign language models often struggle with **visual similarity** between specific digits (e.g., 6, 7, 8, and 9). This project addresses this gap by:

1. **Data Fusion:** Merging 3 major datasets (ASL Digits, Numbers-ASL, and BSL) to create a robust training set of **27,000+ images**.

2. **Two-Phase Training:** Fine-tuning the top 20 layers of MobileNetV2 to capture fine-grained finger features.



## 📂 Project Structure

* `src/realtime_detector.py`: Main script for real-time camera inference.

* `src/training_notebook.ipynb`: The complete training pipeline.

* `model/final_asl_model.keras`: The trained weights (not included in Git due to size).

* `docs/Project_Report.pdf`: Official academic report.

## 📚 Datasets (Data Fusion)

The model was trained on a comprehensive fused dataset of **27,000+ images**, combined from three primary sources to resolve visual ambiguity:

1. **American Sign Language Digits Dataset** 🔗 **Download the Dataset:** [👉 Click here to download ASL-Dataset.zip](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset)
2. **Numbers-ASL Dataset**
3. **BSL (British Sign Language) Digits**

The model was trained on a fused dataset of **27,000+ images**.

## 🚀 How to Run

1. Install requirements: `pip install -r requirements.txt`

2. Place your model in the `model/` folder.

3. Run: `python src/realtime_detector.py`



## 📊 Results - **Test Accuracy:** 88.80%

- **Robustness:** High performance in varied lighting conditions due to background subtraction and data augmentation.