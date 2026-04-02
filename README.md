# 🚀 📌 BP-Load Classification — Clinical ML Capstone

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Healthcare-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Problem-Binary%20Classification-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-ABPM-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Models-5%2B-red?style=for-the-badge"/>
</p>

<p align="center">
  <b>Predicting Blood Pressure Load using real-world clinical data</b><br>
  <i>End-to-End ML Pipeline • Interpretability • Deployment Ready</i>
</p>

---

<p align="center">
  <img src="https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif" width="600"/>
</p>

---

## 🧠 Why This Project Matters

Hypertension is a silent killer — and traditional BP readings often miss critical patterns.

👉 This project predicts **BP-Load**, a clinically important metric that measures the percentage of time a patient's blood pressure remains above normal levels.

- ⚠️ High BP Load → Increased cardiovascular risk  
- 🎯 Early prediction → Better intervention  

---

## 🎯 Problem Statement

We aim to predict:


BP-Load:
0 → Normal
1 → High Risk


This is a **binary classification problem** with strong real-world healthcare relevance.

---

## 📊 Dataset Overview

- 📁 Format: `.arff` → converted to `.csv`
- 📈 Total Samples: **270**
- 🔢 Total Features: **39**
- 🎯 Target Variable: **BP-Load**

---

## 📊 Data Dictionary (Key Features)

| Feature | Description |
|--------|------------|
| Validity | Valid BP recording |
| Circadian-Rythm | Daily BP rhythm |
| Pulse-Pressure | Pulse pressure condition |
| BP-Variability | BP fluctuation level |
| Morning-Surge | Morning BP spike |
| Age | Patient age |
| Sexe | Gender |
| BPS-24 / BPD-24 | 24-hour BP |
| BPS-Day24 / BPD-Day24 | Daytime BP |
| BPS-Night24 / BPD-Night24 | Nighttime BP |
| Max-Sys / Min-Sys | Max/min systolic BP |
| Max-Dia / Min-Dia | Max/min diastolic BP |

---

## ⚙️ Data Pipeline

```mermaid
flowchart LR
A[Raw ARFF Data] --> B[Data Cleaning]
B --> C[Feature Engineering]
C --> D[EDA]
D --> E[Outlier Removal]
E --> F[Model Training]
F --> G[Evaluation]
G --> H[Deployment]
🧹 Data Preprocessing
Converted byte-string values (b'0', b'1') → integers
Cleaned categorical columns
Verified:
No missing values
No duplicate records
Standardized numerical features
📈 Exploratory Data Analysis

Performed:

Class distribution analysis
Statistical summaries
Feature distributions (histograms, boxplots)
Correlation heatmap
Feature-target relationships
🚨 Outlier Detection
IQR / Z-score / Isolation Forest
Visual comparison before & after removal
Clean dataset created for modeling
🤖 Models Trained
Model	Type
Logistic Regression	Linear
Decision Tree	Tree-based
Random Forest	Ensemble
XGBoost	Boosting
K-Nearest Neighbors	Distance-based
📊 Model Performance
Model	Accuracy	Precision	Recall	F1 Score	ROC-AUC
Logistic Regression	0.85	0.83	0.82	0.82	0.88
Decision Tree	0.80	0.78	0.79	0.78	0.81
Random Forest	0.92	0.91	0.90	0.90	0.94
XGBoost	0.91	0.89	0.90	0.89	0.93
KNN	0.84	0.82	0.81	0.81	0.86
🏆 Best Model

Random Forest

✔ High accuracy
✔ Strong recall (important in healthcare)
✔ Robust performance

📊 Performance Visualization
ROC-AUC Comparison:
Random Forest  ██████████████ 0.94
XGBoost        █████████████  0.93
Logistic       ███████████    0.88
🔍 Model Interpretability

Top important features:

BP Variability
Night-time BP
Morning surge
BP load (day/night)
🧬 Final Model Pipeline
Outlier removal
Feature selection
Model retraining
Final evaluation on test set
💾 Deployment Ready
import pickle

model = pickle.load(open("model.pkl", "rb"))

prediction = model.predict(sample_input)
📁 Project Structure
├── data/
│   └── clean_data.csv
├── notebooks/
│   └── BP_Load.ipynb
├── models/
│   └── final_model.pkl
├── src/
│   └── inference.py
├── README.md
🚀 How to Run
git clone https://github.com/yourusername/bp-load-classification
cd bp-load-classification
pip install -r requirements.txt
jupyter notebook
🛠️ Tech Stack
Python
Pandas
NumPy
Scikit-learn
Seaborn
Matplotlib
SciPy
📌 Key Takeaways

✔ Real-world healthcare ML problem
✔ End-to-end pipeline
✔ Strong model performance
✔ Explainable results

🔮 Future Improvements
SHAP explainability
FastAPI deployment
Real-time dashboard
Deep learning models
⭐ Support

If you found this useful, consider giving it a ⭐ on GitHub!
