# 🏭 Industrial Predictive Maintenance

## 📌 Project Overview

Unexpected machine failures can cause expensive production downtime in industrial environments.

This project develops a machine learning system for **predictive maintenance** using the AI4I 2020 Predictive Maintenance Dataset.

The main objective is not only to predict machine failures accurately, but also to minimize **false maintenance alarms**.

A false positive may cause unnecessary inspection or production shutdown. Therefore, the project focuses particularly on reducing the **False Discovery Rate (FDR)** while maintaining useful failure detection capability.

---

## 🎯 Business Problem

The system predicts whether a machine is likely to experience a failure based on operational sensor measurements such as:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The main business constraint is:

> Minimize false-positive maintenance alerts while preserving useful failure detection.

For this reason, model evaluation focuses on **Precision, Recall, False Positives, and False Discovery Rate (FDR)** rather than relying only on Accuracy.

---

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

The dataset contains:

- 10,000 observations
- Machine operating conditions
- Sensor measurements
- Machine failure target
- Individual failure-type indicators

### Target Distribution

| Class | Observations |
|---|---:|
| No Failure | 9,661 |
| Machine Failure | 339 |

Only **3.39%** of the observations represent machine failures.

This strong class imbalance makes Accuracy alone unsuitable as the main evaluation metric.

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## 🔍 Project Workflow

The project includes:

1. Data inspection and cleaning
2. Class imbalance analysis
3. Exploratory data analysis
4. Sensor correlation analysis
5. Feature engineering
6. Categorical feature encoding
7. Train / Validation / Test splitting
8. Random Forest modeling
9. XGBoost experimentation
10. Classification threshold optimization
11. FDR analysis
12. Final held-out test evaluation
13. Feature importance analysis
14. Permutation importance
15. Failure-type analysis
16. Time-to-Failure feasibility assessment

---

## 🧠 Feature Engineering

Three additional operational features were investigated.

### Temperature Difference

```python
Temperature Difference = Process Temperature - Air Temperature
```

This captures the thermal difference between the machine process and surrounding air.

### Mechanical Power

Approximate mechanical power was calculated from torque and rotational speed:

```python
Power = Torque × RPM × 2π / 60
```

### Torque-Wear Interaction

```python
Torque_Wear_Interaction = Torque × Tool Wear
```

This engineered feature represents the interaction between mechanical load and accumulated tool wear.

---

## 📈 Sensor Analysis

Among the original sensor variables analyzed, **Torque** showed the strongest individual linear correlation with machine failure.

However, model-based feature importance showed that engineered load-related features also contained substantial predictive information.

In particular:

- Torque-Wear Interaction
- Mechanical Power
- Rotational Speed
- Temperature Difference

were important to the Random Forest model.

Because several engineered features are derived from the original sensors, their information is correlated. Therefore, feature importance should not be interpreted as evidence of causality.

---

## ⚠️ Preventing Data Leakage

The dataset contains individual failure indicators:

- TWF — Tool Wear Failure
- HDF — Heat Dissipation Failure
- PWF — Power Failure
- OSF — Overstrain Failure
- RNF — Random Failure

These variables were **not used as input features** when predicting the main `Machine failure` target.

Using them as predictors would introduce target leakage because they already encode failure outcomes.

Identifiers such as `UDI` and `Product ID` were also excluded from the predictive feature set.

---

## ✂️ Train / Validation / Test Strategy

The dataset was divided into:

| Dataset | Percentage | Samples |
|---|---:|---:|
| Training | 70% | 7,000 |
| Validation | 15% | 1,500 |
| Test | 15% | 1,500 |

Stratified splitting was used to preserve the rare failure proportion across all three sets.

The workflow was designed to prevent test-set leakage:

**Training Set → Train Model**

**Validation Set → Select Classification Threshold**

**Test Set → Final Evaluation Only**

The final test set was not used to select the operating threshold.

---

## 🎚️ Threshold Optimization

The standard binary classification threshold of `0.50` was not automatically assumed to be the best operating point.

Different thresholds were evaluated using the **validation set**.

At threshold `0.50`:

- Precision: 95.56%
- Recall: 84.31%
- False Positives: 2
- FDR: 4.44%

At threshold `0.70`:

- Precision: 97.22%
- Recall: 68.63%
- False Positives: 1
- FDR: 2.78%

A threshold of **0.70** was selected as the operating point because the project prioritizes reducing false maintenance alarms while retaining meaningful failure detection.

The threshold was locked before evaluating the final test set.

---

## 🏆 Final Model Results

### Random Forest — Held-Out Test Set

**Selected Threshold: 0.70**

| Metric | Result |
|---|---:|
| Accuracy | **99.00%** |
| Precision | **100.00%** |
| Recall | **70.59%** |
| F1 Score | **82.76%** |
| False Discovery Rate | **0.00%** |
| True Positives | **36** |
| False Positives | **0** |
| False Negatives | **15** |
| True Negatives | **1,449** |

---

## 💼 Business Interpretation

The final model detected:

**36 of 51 actual machine failures**

while producing:

**0 false-positive maintenance alerts**

on the held-out test set.

This means every maintenance alert generated by the model in the final test sample corresponded to an actual failure.

However, the conservative operating threshold introduces an important trade-off:

**15 actual failures were not detected.**

Therefore, the system prioritizes reducing unnecessary maintenance alerts at the cost of lower failure recall.

The observed **0% FDR applies specifically to the held-out test sample** and should not be interpreted as a guarantee of zero false alarms in future production data.

---

## 🔧 Failure-Type Analysis

The project also investigated individual failure mechanisms:

| Code | Failure Type |
|---|---|
| TWF | Tool Wear Failure |
| HDF | Heat Dissipation Failure |
| PWF | Power Failure |
| OSF | Overstrain Failure |
| RNF | Random Failure |

Failure types are not mutually exclusive, so the problem cannot be treated as a standard multiclass classification task.

Separate binary classification experiments were therefore performed for each failure type.

HDF, PWF, and OSF showed strong separation in the experimental split.

TWF and RNF were substantially more difficult to detect reliably because of their rarity and probability overlap.

These results should be interpreted cautiously because some failure types contain very few positive observations.

---

## ⏳ Time-to-Failure Limitation

A Time-to-Failure / Remaining Useful Life regression extension was considered.

However, the AI4I dataset does not provide a clearly defined longitudinal run-to-failure history for individual machines or a ground-truth Remaining Useful Life target.

Therefore, an artificial RUL target was intentionally **not constructed from row order or identifiers**, as this could create a misleading regression problem.

A future extension could use a true run-to-failure time-series dataset with machine-level degradation histories.

---

## 💡 Key Findings

- Machine failures represent only **3.39%** of the dataset, creating a significant class imbalance problem.
- Accuracy alone is therefore misleading for this application.
- Torque showed the strongest individual linear association with machine failure among the analyzed original sensors.
- Engineered load-related features provided important predictive information.
- Failure-type variables must be excluded from the main model to prevent target leakage.
- Classification threshold selection can materially change the operational behavior of a predictive-maintenance system.
- Increasing the threshold reduced false alarms but also reduced failure recall.
- The final Random Forest produced **zero false positives on the held-out test sample** at a threshold of `0.70`.
- TWF and RNF remain challenging because of extreme rarity.
- A defensible Remaining Useful Life model requires longitudinal run-to-failure data not provided by this dataset.

---

## 🚀 Future Improvements

Future work could include:

- Evaluation on larger real-world industrial datasets
- Cost-based threshold optimization using actual maintenance and failure costs
- Probability calibration
- Cross-validation for rare failure types
- Anomaly detection approaches
- Additional cost-sensitive learning strategies
- Model monitoring for production data drift
- Remaining Useful Life prediction using longitudinal machine histories

```
```
---

## 👤 Shrouk

Machine Learning Portfolio Project

**Industrial Predictive Maintenance using Machine Learning**

Focus areas:

`Predictive Maintenance` • `Random Forest` • `XGBoost` • `Imbalanced Classification` • `Feature Engineering` • `Threshold Optimization` • `FDR`
