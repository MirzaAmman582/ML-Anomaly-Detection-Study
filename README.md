# ML-Anomaly-Detection-Study

## 📌 Project Overview
This repository contains a comparative analysis of three supervised machine learning algorithms—**Logistic Regression**, **Random Forest**, and **XGBoost**—applied to the detection of rare fraudulent transactions in a highly imbalanced dataset.

### 🤖 Relevance to Robotics
In robotics, finding a "fraudulent" transaction is mathematically identical to detecting **rare-event failures**, such as sensor faults, motor anomalies, or collision risks. This project demonstrates proficiency in:
* **Imbalanced Data Handling:** Using SMOTE and cost-sensitive learning.
* **Model Selection:** Balancing precision vs. recall for safety-critical applications.
* **Performance Metrics:** Using ROC-AUC and F1-score over simple accuracy.

## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Pandas, Matplotlib, Seaborn.

## 📊 Performance Comparison
Based on the experimental results, **XGBoost** provided the most robust balance of discriminative power and recall.


| Model | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.06 | 0.92 | 0.11 | 0.95 |
| Random Forest | 0.96 | 0.73 | 0.83 | 0.87 |
| **XGBoost** | **0.33** | **0.87** | **0.48** | **0.98** |

## 📁 Repository Structure
* `fraud_detection.py`: Python source code for data preprocessing and model training.
* `dataset_sample.png`: Visual overview of the processed dataset.
* `A Comparative Study...pdf`: Full technical project report.
* `LICENSE`: MIT License.
* `README.md`: Project overview and results.

## ⚙️ Usage
1. Clone this repository.
2. Install dependencies: `pip install pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn`.
3. Download the dataset from [Kaggle](https://www.kaggle.com).
4. Run the analysis script: `python fraud_detection.py`.

---
*Note: This project was completed as part of postgraduate studies.*
