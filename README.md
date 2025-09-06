# Transformer-BERT Fake News Classifier

This project implements a deep learning pipeline for detecting fake news on social media using **BERT-based transformer models**.  
The dataset is from **Constraint@AAAI-2021**, focusing on COVID-19-related misinformation.  

## 📂 Project Overview
- **Goal**: Classify posts as `Real` (verified) or `Fake` (misinformation).  
- **Models Used**:
  - `bert-base-uncased`
  - `covid-twitter-bert`
  - `twhin-bert-base`
  - `socbert`
- **Frameworks**: PyTorch, Hugging Face Transformers, Scikit-learn, Optuna  

## 🚀 Features
- Data preprocessing and cleaning for social media posts  
- Fine-tuning of transformer models for text classification  
- Hyperparameter tuning using Optuna/manual search  
- Evaluation using Accuracy, Precision, Recall, and F1-score  
- Visualization with confusion matrices and reports  

## 📊 Results
- Achieved robust performance across BERT variants  
- Best-performing model selected based on validation F1-score  

## ⚙️ Requirements
```bash
torch
transformers
scikit-learn
pandas
numpy
optuna
```

## ▶️ Usage
```bash
jupyter notebook 24CS60R47_transfomer.ipynb
```

## 📑 Report
See `Report.pdf` for details on dataset, methodology, results, and findings.
