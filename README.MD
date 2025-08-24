# Document Segmentation with VOC Archival Page Scans

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-green.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-red.svg)](https://xgboost.readthedocs.io)

This repository contains a comprehensive machine learning pipeline for predicting TANAP boundaries in historical VOC (Dutch East India Company) archival page scans. The goal is to identify sequences of pages within the dataset that constitute single documents by classifying each page as NONE, START, MIDDLE, or END of a document sequence.

## 🎯 Project Overview

The project tackles the challenge of **document boundary detection** in historical archives using machine learning. Given a sequence of scanned pages with rich XML layout annotations and XMI named entity data, the models predict whether each page represents:

- **NONE**: Page is not part of any document
- **START**: First page of a new document  
- **MIDDLE**: Continuation page within a document
- **END**: Final page of a document

## 📁 Project Structure

```
/
├── data/
│   └── train/
│       ├── renate_dataset.csv          # Original dataset (16,774 samples)
│       └── features_dataset.csv        # Extracted features (generated)
├── notebooks/                          # ML model implementations
│   ├── 01_logistic_regression_model.ipynb
│   ├── 02_random_forest_model.ipynb
│   ├── 03_xgboost_model.ipynb
│   ├── 04_neural_network_model.ipynb
│   └── 05_svm_model.ipynb
├── models/                             # Saved models and results
├── feature_extraction.py              # Feature engineering pipeline
├── requirements.txt                   # Python dependencies
├── README.MD                          # This file
└── CLAUDE.md                          # Development documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Globalise Document Segmentation"

# Install dependencies
pip install -r requirements.txt
```

### 2. Feature Extraction

Extract features from the raw XML/XMI data:

```bash
python feature_extraction.py
```

This generates `data/train/features_dataset.csv` with 60+ engineered features.

### 3. Model Training

Launch Jupyter and run the model notebooks:

```bash
jupyter notebook notebooks/
```

Train models in order:
1. **Logistic Regression** - Linear baseline
2. **Random Forest** - Tree ensemble  
3. **XGBoost** - Gradient boosting
4. **Neural Network** - Deep learning
5. **Support Vector Machine** - Kernel methods

### 4. Results Analysis

All models save results to the `models/` directory:
- Performance metrics (`*_results.csv`)
- Feature importance (`*_feature_importance.csv`)  
- Trained models (`*.pkl` or `*.h5`)

## 📊 Dataset Details

### Original Dataset (`renate_dataset.csv`)
- **Size**: 16,774 historical page scans
- **Source**: Dutch National Archives (NL-HaNA) VOC collection
- **Annotations**: Created by Dutch historians using Laypa tool
- **Target Distribution**: 
  - MIDDLE: 13,058 (78%)
  - NONE: 1,535 (9%)
  - START: 1,091 (7%)  
  - END: 1,090 (6%)

### Data Formats
- **XML Data**: PAGE format with text regions, coordinates, layout structure
- **XMI Data**: UIMA annotations with named entities, sentences, tokens
- **Missing Data**: ~1,424 pages lack XMI annotations

## 🔧 Feature Engineering

The pipeline extracts three categories of features:

### XML Layout Features (25+ features)
- **Geometric**: Text region counts, areas, spatial distribution
- **Structural**: Reading order complexity, region types (header/paragraph/table)
- **Layout**: Page coverage ratio, region variance, spatial positioning

### XMI NER Features (20+ features)  
- **Entities**: Person, location, organization, date, money counts
- **Linguistic**: Sentence/token counts, average lengths, punctuation density
- **Language**: Dutch language indicators, formal language scoring

### Sequence Features (15+ features)
- **Temporal**: Previous/next page feature comparisons  
- **Statistical**: Rolling window means and standard deviations
- **Positional**: Document sequence position, delta features

## 🤖 Machine Learning Models

### Model Comparison

| Model | Strengths | Best For |
|-------|-----------|----------|
| **Logistic Regression** | Fast, interpretable, linear relationships | Baseline, feature analysis |
| **Random Forest** | Handles non-linearity, robust, feature importance | General-purpose ensemble |
| **XGBoost** | High performance, regularization, handles missing values | Competition-grade results |  
| **Neural Network** | Complex patterns, end-to-end learning | Non-linear relationships |
| **SVM** | High-dimensional data, kernel trick, memory efficient | Text/document classification |

### Model Features

Each notebook includes:
- **Hyperparameter tuning** with grid search
- **Cross-validation** for robust evaluation  
- **Feature importance** analysis
- **Confusion matrices** and classification reports
- **Model-specific insights** (decision boundaries, learning curves, etc.)

## 📈 Performance Evaluation

Models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1 for class imbalance
- **Precision/Recall**: Per-class performance
- **Cross-validation**: 3-5 fold CV for generalization
- **Confidence analysis**: Prediction certainty assessment

## 🏛️ Historical Context

### TANAP Project
The **Towards a New Age of Partnership (TANAP)** project digitizes VOC archives, making 17th-18th century Dutch colonial documents accessible for research.

### VOC Archives
The **Dutch East India Company (VOC)** produced extensive administrative records during its operations (1602-1799), including:
- Trading correspondence
- Administrative reports  
- Legal documents
- Ships' logs and inventories

### Document Types
Historical documents in the dataset include:
- **Letters**: Official and personal correspondence
- **Newspapers**: Colonial period publications
- **Administrative records**: Trade, legal, and governmental documents
- **Inventories**: Cargo and property listings

## 💻 Technical Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM (for feature extraction)
- 2GB storage space

### Recommended  
- Python 3.9+
- 16GB+ RAM (for neural network training)
- GPU support (for TensorFlow models)
- SSD storage for faster I/O

### Key Dependencies
- **pandas, numpy**: Data manipulation
- **scikit-learn**: Traditional ML algorithms
- **tensorflow/keras**: Deep learning
- **xgboost**: Gradient boosting
- **matplotlib, seaborn**: Visualization

## 🔍 Development Notes

### Performance Considerations
- **Feature extraction**: Processes 16K+ samples with XML/XMI parsing (~30-60 minutes)
- **SVM training**: Limited to 5K samples for computational efficiency
- **Neural networks**: Benefit significantly from GPU acceleration
- **Memory usage**: XGBoost and Random Forest require substantial RAM

### Data Handling
- **Class imbalance**: 78% of samples are MIDDLE class
- **Missing data**: XMI features have ~8% missing values
- **Scalability**: Pipeline designed for datasets up to 100K samples

### Model Selection Tips
- Start with **Random Forest** for quick, reliable results
- Use **XGBoost** for maximum performance  
- Try **Neural Networks** if you have sufficient computational resources
- Use **Logistic Regression** for interpretability and speed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the TANAP digitization initiative. Please respect the historical nature of the source materials and cite appropriately in academic work.

## 🙏 Acknowledgments

- **TANAP Project**: For digitizing and providing access to VOC archives
- **Dutch National Archives**: For preserving and maintaining historical documents
- **Dutch Historians**: For creating detailed annotations using the Laypa tool
- **Laypa Team**: For developing the annotation platform

## 📚 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{voc_document_segmentation_2024,
  title={Document Segmentation with VOC Archival Page Scans},
  author={[Your Name]},
  year={2024},
  note={TANAP Project - Towards a New Age of Partnership}
}
```

---

**Keywords**: Historical Documents, Document Segmentation, Machine Learning, VOC Archives, TANAP, Named Entity Recognition, Layout Analysis, Dutch Colonial History
