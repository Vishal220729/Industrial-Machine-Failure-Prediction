# Industrial Machine Failure Prediction

### Track 1 – Final Capstone – Problem 21
**Institution:** Learn Depth Academy LLP  
**Domain:** Industrial Systems / Predictive Maintenance  
**ML Task:** Binary Classification (`0`: No Failure, `1`: Machine Failure)  
**Academic Target Duration:** 10 Days (Implemented as 1-Day Complete Academic Working Prototype)

---

## 1. Problem Statement
In automated manufacturing environments, computer numerical control (CNC) milling machinery operates under high thermal and mechanical stress. Sudden, unexpected machine breakdowns lead to costly production stoppages, scrapped workpieces, and severe operator hazards.

The official capstone problem statement requires:
> **Predict whether an industrial machine is likely to experience an operational failure within a defined future operating window based on sensor measurements and operating conditions.**

---

## 2. Project Objectives
1. **Data Ingestion & Integrity Audit:** Rigorously verify 10,000 milling records from the UCI repository for zero missing values and zero duplicate records.
2. **Data Leakage Quarantine:** Strictly isolate arbitrary identifiers (`UDI`, `Product ID`) and post-failure failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
3. **Reproducible Pipeline:** Construct a `ColumnTransformer` (scaling continuous features and one-hot encoding machine type) fitted strictly on the 80% training partition.
4. **Foundational ML Benchmarking:** Train and benchmark **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Decision Tree** classifiers under stratified conditions.
5. **False Negative Minimization:** Prioritize Recall and F1-score over raw accuracy due to the 28.5:1 class imbalance (3.39% failure rate).
6. **Artifact Persistence:** Serialize the champion model (`model/machine_failure_model.pkl`) and preprocessing pipeline (`model/preprocessing_pipeline.pkl`) using `joblib`.
7. **Interactive Web Application:** Deploy a reactive Streamlit application (`app.py`) providing input validation, real-time inference, and diagnostic risk guidance.
8. **Systematic Verification:** Test 8 comprehensive operational scenarios covering normal operation, thermal stress, overstrain, tool wear, variants, and boundary limits.
9. **Full Academic Deliverables:** Provide a 29-section reproducible Jupyter Notebook, editable technical report (`technical_report.docx` / `.pdf`), 20-slide presentation (`presentation.pptx`), and 30+ viva defense questions.

---

## 3. Dataset Information & Attribution
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset
- **Repository:** UCI Machine Learning Repository
- **DOI:** [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)
- **Author:** Stephan Matzka, HTW Berlin (2020)
- **Observations:** 10,000 milling records
- **Features:** 14 total columns (6 predictive features, 7 excluded/quarantined columns, 1 target column)
- **Target Variable:** `Machine failure`
  - Class `0` (Nominal Operation): 9,661 samples (96.61%)
  - Class `1` (Machine Failure): 339 samples (3.39%)
  - Imbalance Ratio: ~28.5 : 1

### Mapping of Capstone Expected Inputs vs Real Dataset
| Capstone Generic Input | Actual AI4I 2020 Dataset Feature | Physical Coupling Justification |
| :--- | :--- | :--- |
| **Temperature** | `Air temperature [K]`, `Process temperature [K]` | Ambient air temperature and cutting zone process temperature (Kelvin). |
| **Vibration & Pressure** | `Torque [Nm]`, `Rotational speed [rpm]` | Dynamic mechanical forces and spindle oscillations directly manifest as torque reaction and speed fluctuations. |
| **Runtime & Maintenance Gap** | `Tool wear [min]` | Cumulative cutting duration of the tool insert relative to its failure threshold. |

*Note: In compliance with academic integrity guidelines, zero features were fabricated or renamed with artificial figures.*

---

## 4. Technologies Used
- **Programming Language:** Python 3.12 / 3.11 / 3.14
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Pipeline:** `scikit-learn`, `joblib`
- **Interactive Web App:** `streamlit`
- **Notebook & Reporting:** `jupyter`, `nbformat`, `nbconvert`, `python-docx`, `python-pptx`, `docx2pdf`

---

## 5. Project Directory Structure
```
Industrial_Machine_Failure_Prediction/
│
├── dataset/
│   └── ai4i2020.csv                          # Genuine UCI AI4I 2020 dataset
│
├── notebook/
│   └── machine_failure_prediction.ipynb       # Fully executed 29-section notebook
│
├── src/
│   ├── data_preprocessing.py                 # Ingestion, quality audit, leakage check, ColumnTransformer
│   ├── train_model.py                        # Model training & champion selection
│   ├── evaluate_model.py                     # Evaluation, visualizations & error analysis
│   ├── run_tests.py                          # 8 mandatory test cases verification runner
│   ├── generate_notebook.py                  # Programmatic notebook builder
│   ├── generate_presentation.py              # 20-slide PowerPoint builder
│   ├── generate_report.py                    # Academic technical report DOCX generator
│   ├── convert_to_pdf.ps1                    # PowerShell COM Word-to-PDF export
│   └── utils.py                              # Shared utilities & publication plot styling
│
├── model/
│   ├── machine_failure_model.pkl             # Serialized champion Decision Tree model
│   └── preprocessing_pipeline.pkl            # Serialized ColumnTransformer pipeline
│
├── results/
│   ├── target_distribution.png               # Target class distribution chart
│   ├── feature_distributions.png             # Continuous sensor histograms & KDE
│   ├── correlation_heatmap.png               # Pearson correlation matrix
│   ├── boxplots.png                          # Feature dispersion by failure status
│   ├── confusion_matrix.png                  # Side-by-side confusion matrices (N=2,000)
│   ├── roc_curve.png                         # Multi-model ROC comparison curves
│   └── model_comparison.png                  # Benchmarking comparison bar chart
│
├── screenshots/
│   └── streamlit/
│       ├── streamlit_app_overview.png        # Full application interface
│       ├── streamlit_prediction_normal.png   # Nominal state diagnostic result
│       └── streamlit_prediction_failure.png  # Critical failure alert diagnostic
│
├── report/
│   ├── technical_report.docx                 # Full academic project report (Editable)
│   └── technical_report.pdf                  # Submission-ready academic report
│
├── presentation/
│   └── presentation.pptx                     # Professional 20-slide presentation
│
├── app.py                                    # Streamlit reactive diagnostic web app
├── requirements.txt                          # Project dependency specifications
└── README.md                                 # Complete documentation
```

---

## 6. Installation & Execution Guide

### Prerequisites
- Windows 10/11 with Python 3.11 or 3.12 installed.
- PowerShell or Windows Command Prompt.

### Step 1: Clone / Navigate to Project Directory
```powershell
cd "Industrial_Machine_Failure_Prediction"
```

### Step 2: Create and Activate Virtual Environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Training, Model Serialization & Visualization Generation
```powershell
python src/evaluate_model.py
```
*Outputs: Generates all 7 figures into `results/`, fits and serializes `model/machine_failure_model.pkl` and `model/preprocessing_pipeline.pkl`, and prints test set metrics.*

### Step 5: Execute Application Test Suite (8 Test Cases)
```powershell
python src/run_tests.py
```

### Step 6: Launch Interactive Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 7: Run Jupyter Notebook
```powershell
jupyter notebook notebook/machine_failure_prediction.ipynb
```

---

## 7. Experimental Results & Model Benchmarking

Models were evaluated on an unseen, stratified test split of **2,000 observations (1,932 nominal records, 68 actual failures)**.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Negatives (FN) | False Positives (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Balanced) | 82.45% | 14.18% | 82.35% | 0.2419 | 0.9069 | 12 | 339 |
| **K-Nearest Neighbors** (k=5) | 97.35% | 82.61% | 27.94% | 0.4176 | 0.8368 | 49 | 4 |
| **Decision Tree (Champion)** | **93.65%** | **32.95%** | **83.82%** | **0.4730** | **0.8654** | **11** | **116** |

### Why Decision Tree was Selected as Champion:
1. **Lowest Critical Misses (FN = 11):** Catches **83.82% of all breakdowns** (57 of 68 failures). In industrial maintenance, missed breakdowns (False Negatives) cause catastrophic downtime.
2. **Deceptive KNN Accuracy:** Although KNN achieved 97.35% accuracy, it missed 72% of machine failures (49 False Negatives out of 68), rendering it unacceptable for safety-critical maintenance.
3. **F1-Score Superiority:** Achieved an F1-score of **0.4730**, significantly outperforming Logistic Regression (0.2419).
4. **Physical Interpretability:** Decision trees provide clear rule splits on torque, rotational speed, and tool wear that reflect physical cutting thresholds.

---

## 8. Systematic Application Verification (8 Test Cases)

| Test Case | Telemetry Input Condition | Expected Behaviour | Actual Model Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **TC-01: Normal Operation** | Type: L, Air: 300K, Proc: 310K, RPM: 1500, Torque: 40Nm, Wear: 50m | Class 0 (No Failure, Low Risk) | No Machine Failure (1.10% risk) | **PASS** |
| **TC-02: Higher Temperature** | Type: L, Air: 304K, Proc: 313.5K, RPM: 1400, Torque: 45Nm, Wear: 80m | Diagnostic check (Thermal stress) | No Machine Failure (9.43% risk) | **PASS** |
| **TC-03: Higher Torque Load** | Type: L, Air: 300K, Proc: 310K, RPM: 1200, Torque: 70Nm, Wear: 120m | Class 1 (Machine Failure, Overstrain) | Machine Failure (95.00% risk) | **PASS** |
| **TC-04: Severe Tool Wear** | Type: L, Air: 301K, Proc: 311K, RPM: 1350, Torque: 62Nm, Wear: 240m | Class 1 (Machine Failure, Tool Wear) | Machine Failure (100.00% risk) | **PASS** |
| **TC-05: Machine Type H** | Type: H, Air: 299K, Proc: 309K, RPM: 1520, Torque: 38Nm, Wear: 60m | Class 0 (Nominal for H-Variant) | No Machine Failure (1.10% risk) | **PASS** |
| **TC-06: Boundary Maximums** | Type: M, Air: 304.5K, Proc: 313.8K, RPM: 1168, Torque: 76.6Nm, Wear: 253m | Class 1 (Extreme Boundary Limit) | Machine Failure (100.00% risk) | **PASS** |
| **TC-07: Invalid Telemetry** | Type: L, Air: 308K, Process: 295K (Process < Air Temp) | Validation warning raised in UI | Physical Inconsistency Warning Flagged | **PASS** |
| **TC-08: Default Baseline** | Type: L, Baseline parameters submitted | Class 0 (Safe baseline prediction) | No Machine Failure (1.10% risk) | **PASS** |

---

## 9. Academic Limitations & Future Scope
- **Educational Prototype:** Designed for viva defense and demonstration; not wired into factory PLC automation loops.
- **Static Sensor Aggregates:** High-frequency acoustic vibrations were pre-aggregated into discrete rotational speed and torque values.
- **Future Scope:**
  - Direct integration with industrial IoT edge microcontrollers via MQTT and OPC-UA protocols.
  - Remaining Useful Life (RUL) regression forecasting.
  - Multi-class diagnostics to isolate specific failure modes (`TWF`, `HDF`, `PWF`, `OSF`).

---

## 10. References
1. **Matzka, S. (2020).** *AI4I 2020 Predictive Maintenance Dataset.* UCI Machine Learning Repository. [DOI: 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
2. **Mobley, R. K. (2002).** *An Introduction to Predictive Maintenance (2nd ed.).* Elsevier Butterworth-Heinemann.
3. **Bousdekis, A., et al. (2021).** *Decision making in predictive maintenance: literature review and research agenda for Industry 4.0.* Computers in Industry, 128.
4. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.
5. **Streamlit Developers (2024).** *Streamlit Documentation.* [https://docs.streamlit.io/](https://docs.streamlit.io/)

---

## 11. Academic Disclaimer
This project is an educational capstone prototype developed under **Learn Depth Academy LLP (Track 1 – Final Capstone – Problem 21)**. All experimental metrics and statistics originate exclusively from programmatic execution on the legitimate UCI AI4I 2020 benchmark dataset.
