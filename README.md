<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=BP-Load%20Classification&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Ambulatory%20Blood%20Pressure%20Monitoring%20%7C%20Machine%20Learning%20Pipeline&descAlignY=56&descSize=15" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FB5?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **A clinical machine learning pipeline trained on Ambulatory Blood Pressure Monitoring (ABPM) data to predict BP-Load — the percentage of blood pressure readings exceeding normal thresholds — using 5 classifiers, Yeo-Johnson transformation, statistical significance testing, and a Decision Tree achieving perfect accuracy with only 5 features.**

<br/>

[🔍 Explore the Notebook](#-project-structure) · [📊 See Results](#-model-performance) · [🚀 Run Locally](#-getting-started) · [🧪 Predict on a Patient](#-predict-on-a-new-patient)

<br/>
</div>

---

## 🫀 Problem Statement

**BP-Load** is defined as the percentage of blood pressure readings that exceed normal thresholds during a 24-hour ambulatory monitoring session. A high BP-Load is a strong predictor of hypertension-related organ damage — often more informative than a single clinic measurement.

This project builds a binary classification pipeline to predict:

```
BP-Load = 1  →  High BP burden (elevated % readings above normal)
BP-Load = 0  →  Normal BP burden
```

The goal is accurate, interpretable early detection using routine ABPM data.

---

## 📁 Project Structure

```
bp-load-classification/
│
├── 📓 BP_Load_Classification_Capstone_Project_8.ipynb   # Full pipeline
│
├── 📊 data/
│   ├── ABPM-dataset.arff                                # Raw ARFF dataset
│   ├── clean_data.csv                                   # Converted CSV
│   └── final_dataset.csv                                # Reduced 5-feature dataset
│
├── 🤖 model.pkl                                         # Final saved Decision Tree
│
└── 📄 README.md
```

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| **Source** | ABPM (Ambulatory Blood Pressure Monitoring) Dataset — ARFF format |
| **Shape** | 270 rows × 39 columns |
| **After dropping `BP-Variability`** | 270 rows × 38 columns |
| **Missing values** | ✅ None |
| **Duplicates** | ✅ None |
| **Format challenge** | All categorical columns were byte-strings (`b'0'`, `b'1'`) — converted to integers |
| **Outlier strategy** | ⚠️ No removal — dataset is small (270 rows); removal would hurt model performance |

### Feature Reference (39 Columns)

| Feature | Description | Type |
|---|---|---|
| `Validity` | Whether BP recording is valid (1 = valid) | Binary |
| `Circadian-Rythm` | Normal BP circadian rhythm present | Binary |
| `Pulse-Pressure` | Elevated pulse pressure status | Binary |
| `BP-Variability` | BP variability classification *(dropped)* | Binary |
| **`BP-Load`** | **Target — % readings above normal** | **Binary** |
| `Morning-Surge` | Morning BP surge present | Binary |
| `Interrupt` | Recording was interrupted | Binary |
| `Sexe` | Patient sex (0 = Female, 1 = Male) | Binary |
| `HRecord` | Hours of total recording | Numeric |
| `Perc` | Percentage of valid readings | Numeric |
| `Age` | Patient age | Numeric |
| `Height` / `Weight` | Anthropometric measurements | Numeric |
| `BPS-24` / `BPD-24` | Mean systolic/diastolic BP over 24h (mmHg) | Numeric |
| `BPS-Day24` / `BPD-Day24` | Mean daytime systolic/diastolic BP (mmHg) | Numeric |
| `BPS-Night24` / `BPD-Night24` | Mean nighttime systolic/diastolic BP (mmHg) | Numeric |
| `BPS-load-Day` / `BPD-load-Day` | % readings above normal (daytime) | Numeric |
| `BPS-load-Night` / `BPD-load-Night` | % readings above normal (nighttime) | Numeric |
| `Max-Sys` / `Min-Sys` | Maximum/minimum systolic BP recorded | Numeric |
| `Max-Dia` / `Min-Dia` | Maximum/minimum diastolic BP recorded | Numeric |
| `Sys-Night-Des` / `Dia-Night-Des` | Nighttime BP dipping % | Numeric |
| `BPS-CV-all` … `BPD-CV-Night` | Coefficient of variation for BP variability | Numeric |
| `BPS-wakeUp` / `BPD-wakeUp` | BP at wake-up | Numeric |
| `low-BPS-Night` / `low-BPD-Night` | Count of low BP episodes at night | Numeric |

---

## 🔬 Full Pipeline

```
Raw ARFF Data (270 × 39)
        │
        ▼
Phase 2: Data Setup ────── scipy.io.arff import, byte-string decoding
        │                   (b'0'/b'1' → int), dtype validation
        ▼
Phase 3: EDA ───────────── class distribution, statistical summary,
        │                   KDE plots, violin/boxplots by class,
        │                   correlation heatmap, BP-Load correlation bar chart
        ▼
Phase 4: Outlier Decision ─ No removal (270 rows — too small to shrink further)
        │                   Yeo-Johnson power transform on all numeric cols
        ▼
Phase 5: Preprocessing ─── drop BP-Variability, stratified 80/20 split,
        │                   StandardScaler, t-tests + chi-square significance tests
        ▼
Phase 6: Model Training ─── 5 classifiers trained and evaluated
        │
        ▼
Phase 7: Model Selection ── Decision Tree chosen (perfect metrics + interpretability)
        │
        ▼
Phase 8: Feature Importance ─ BPS-load-Day dominates at 96.04%,
        │                      Top 5 features selected
        ▼
Phase 9: Reduced Model ───── Retrained on 5 features → identical perfect performance
        │
        ▼
Phase 10: Deployment ──────── Pickle save, interactive CLI inference demo
```

---

## 📊 Class Distribution

| Class | Count | Percentage |
|---|---|---|
| 1 — High BP-Load | 173 | **64.07%** |
| 0 — Normal BP-Load | 97 | 35.93% |

> The dataset is **moderately imbalanced** — High BP-Load is the majority class at 64%. Stratified splitting was used to preserve this ratio.
> Test set breakdown: 35 High BP-Load, 19 Normal BP-Load (54 total).

---

## 🧪 Statistical Analysis

### Point-Biserial Correlation (Numeric Features vs. BP-Load)

| Feature | Correlation | P-Value | Significant? |
|---|---|---|---|
| **BPS-load-Day** | **+0.844** | 1.38e-74 | ✅ Yes |
| **BPS-Day24** | **+0.749** | 7.11e-50 | ✅ Yes |
| **BPS-24** | **+0.743** | 1.12e-48 | ✅ Yes |
| **BPD-load-Day** | **+0.670** | 1.44e-36 | ✅ Yes |
| **BPD-Day24** | **+0.642** | 8.86e-33 | ✅ Yes |
| **BPD-24** | +0.630 | 2.81e-31 | ✅ Yes |
| **BPS-load-Night** | +0.615 | 1.71e-29 | ✅ Yes |
| **Max-Sys** | +0.550 | 9.61e-23 | ✅ Yes |
| **BPD-load-Night** | +0.432 | 9.86e-14 | ✅ Yes |
| BPS-Night24 | +0.385 | 5.99e-11 | ✅ Yes |
| Weight | +0.198 | 0.00109 | ✅ Yes |
| Age | +0.184 | 0.00239 | ✅ Yes |
| HRecord | -0.022 | 0.71682 | ❌ No |
| Height | +0.065 | 0.28539 | ❌ No |
| Sys-Night-Des | +0.025 | 0.68675 | ❌ No |
| Dia-Night-Des | +0.022 | 0.71458 | ❌ No |

### Chi-Square Test (Binary Features vs. BP-Load)

| Feature | P-Value | Significant? |
|---|---|---|
| **Pulse-Pressure** | **0.02484** | ✅ Yes |
| Validity | 0.17528 | ❌ No |
| Circadian-Rythm | 1.00000 | ❌ No |
| Morning-Surge | 0.30298 | ❌ No |
| Interrupt | 0.24238 | ❌ No |
| Sexe | 0.53989 | ❌ No |

> Among all binary features, only **Pulse-Pressure** has a statistically significant relationship with BP-Load.

---

## 🔑 Key EDA Insights

| Feature | Clinical Interpretation |
|---|---|
| **BPS-load-Day** | Strongest predictor — daytime systolic BP load almost fully determines BP classification |
| **BPS-Day24 / BPS-24** | Higher average daytime/24h systolic BP strongly associated with elevated BP-Load |
| **BPD-Day24 / BPD-24** | Diastolic pressure during day is also highly relevant |
| **BPS-load-Night / BPD-load-Night** | Nighttime BP burden adds significant predictive value |
| **Max-Sys / Max-Dia** | Peak BP values indicate hypertensive episodes |
| **BPS-CV-all** | Systolic variability across 24h increases BP-Load risk |
| **BPS-wakeUp / BPD-wakeUp** | Wake-up BP reflects morning surge and circadian risk |

---

## 🤖 Model Performance

All 5 models trained on **216 samples** (80%), tested on **54 samples** (20%), stratified split, StandardScaler applied.

| Model | Accuracy | Precision | Recall | F1-Score | MCC |
|---|---|---|---|---|---|
| Logistic Regression | 0.98148 | 1.00000 | 0.97143 | 0.98551 | 0.96065 |
| **Decision Tree** ⭐ | **1.00000** | **1.00000** | **1.00000** | **1.00000** | **1.00000** |
| Random Forest | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| KNN | 0.92593 | 0.94286 | 0.94286 | 0.94286 | 0.83759 |
| XGBoost | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

> **Decision Tree** was selected as the final model — perfect metrics combined with full interpretability (every decision node is human-readable).

**Final Confusion Matrix (Test Set — 54 samples):**
```
Predicted →      0    1
Actual 0  →  [ 19    0 ]   ← 0 False Positives
Actual 1  →  [  0   35 ]   ← 0 False Negatives
```

---

## 🏆 Feature Importance (Decision Tree)

The Decision Tree assigned nearly all predictive power to a single feature:

| Rank | Feature | Importance |
|---|---|---|
| 1 | **BPS-load-Day** | **96.04%** |
| 2 | Age | 1.82% |
| 3 | Min-Sys | 1.51% |
| 4 | Min-Dia | 0.46% |
| 5 | Max-Dia | 0.17% |
| All others (32 features) | — | 0.00% |

### Reduced Model (Top 5 Features Only)

| Metric | Full Model (37 features) | Reduced Model (5 features) |
|---|---|---|
| Accuracy | 1.0000 | **1.0000** |
| Precision | 1.0000 | **1.0000** |
| Recall | 1.0000 | **1.0000** |
| F1-Score | 1.0000 | **1.0000** |
| MCC | 1.0000 | **1.0000** |

> ✅ **Identical performance with 86% fewer features.** A model using only `BPS-load-Day`, `Age`, `Min-Sys`, `Min-Dia`, and `Max-Dia` is sufficient for perfect classification.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/bp-load-classification.git
cd bp-load-classification
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy jupyter
```

### 3. Place the dataset

Put `ABPM-dataset.arff` in `data/` (or update the path in the notebook).

### 4. Run the notebook

```bash
jupyter notebook BP_Load_Classification_Capstone_Project_8.ipynb
```

---

## 🧪 Predict on a New Patient

```python
import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Features: BPS-load-Day, Age, Min-Sys, Min-Dia, Max-Dia
# (Yeo-Johnson transformed + StandardScaler applied)
sample = [[0.27617, 0.535684, -2.623383, -0.651608, -0.776941]]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print(f"Prediction: {prediction[0]} — {'High BP-Load' if prediction[0] == 1 else 'Normal BP-Load'}")
```

**Interactive CLI demo (from notebook):**
```
Enter BPS Load Day  : 0.27617
Enter Age           : 0.535684
Enter Min_Sys       : -2.623383
Enter Min_Dia       : -0.651608
Enter Max_Dia       : -0.776941

→ The Prediction for the patient is: [1]  (High BP-Load)
```

---

## 📌 Key Findings & Insights

- **Zero missing values and duplicates** in the raw ARFF file — minimal preprocessing overhead
- **Byte-string encoding** (`b'0'`, `b'1'`) in all categorical columns required a custom decoding step before any analysis
- **`BP-Variability` was dropped** — constant column providing no discriminative signal
- **Outlier removal was intentionally skipped** — at 270 rows, IQR removal would shrink the dataset too aggressively and risk underfitting
- **Yeo-Johnson transformation** significantly normalized skewed numeric features (Height had extreme skew of -2.72 pre-transform)
- **`BPS-load-Day` alone accounts for 96% of feature importance** — daytime systolic load is the near-sole driver of BP-Load classification
- **Only 1 of 6 binary features** (Pulse-Pressure, p=0.025) showed statistically significant association with BP-Load via chi-square test
- **Logistic Regression** also performed exceptionally well (98.15% accuracy) — confirming strong linear separability in the transformed feature space

---

## 🧠 Skills Demonstrated

| Area | Details |
|---|---|
| **Data Engineering** | ARFF format parsing with `scipy.io.arff`, byte-string decoding, dtype conversion |
| **EDA** | KDE plots, violin plots, boxplots stratified by class, correlation heatmap |
| **Statistical Testing** | Point-biserial correlation (numeric vs binary), chi-square test (binary vs binary) |
| **Feature Engineering** | Yeo-Johnson power transform, StandardScaler, binary/numeric column auto-detection |
| **ML Modeling** | 5 classifiers trained and rigorously compared with consistent evaluation framework |
| **Model Evaluation** | Accuracy, Precision, Recall, F1, MCC, confusion matrix |
| **Feature Selection** | Decision Tree importance ranking, 96% concentration in 1 feature, reduced model validation |
| **Deployment** | Pickle persistence, scaler + model inference pipeline, interactive CLI demo |
| **Clinical Reasoning** | ABPM domain knowledge, outlier retention decision, BP-Load interpretation |

---

## 📚 References

- Dataset: ABPM (Ambulatory Blood Pressure Monitoring) — ARFF format
- Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org)
- XGBoost Documentation: [xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- Clinical Reference: [WHO Hypertension Guidelines](https://www.who.int/news-room/fact-sheets/detail/hypertension)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

**Made with ❤️ and 270 blood pressure readings**

⭐ Star this repo if you found it useful!

</div>
