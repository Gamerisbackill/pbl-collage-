# SmartMach – Machine Learning based Predictive Maintenance

**Institute:** Jaypee Institute of Information Technology, Noida – Department of Computer Applications
**Course:** Machine Learning Lab (24M15CA114) | **Program:** MCA, 1st Year 1st Sem (2026–2027)
**Team:** Karan Malik (MCG260243), Sachit Rawat (MCG260307), Hardik Sharma (MCG260585), Pankaj Singh Negi (MCG260518)

## 1. What this project does

SmartMach predicts whether an industrial machine is likely to fail, using five sensor/operating parameters
(air temperature, process temperature, rotational speed, torque, tool wear) plus the machine type (L/M/H).
It trains and compares five ML models, picks the best one, shows which parameters matter most, and finally
gives a **failure prediction and a Low / Medium / High risk level** for any machine the user enters.

Dataset: **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository), 10,000 rows.

## 2. Files

| File | Purpose |
|---|---|
| `smartmach.py` | The complete project code (one script, divided into numbered sections) |
| `README.md` | This file |
| `ai4i2020.csv` | Dataset – downloaded automatically on first run  |
| `smartmach_outputs/` | Created on run: 7 plots (`.png`), `model_comparison.csv`, `smartmach_best_model.joblib` |

## 3. How to run

**Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
(all are pre-installed in Google Colab; elsewhere run `pip install pandas numpy matplotlib seaborn scikit-learn joblib`).

- **Google Colab:** upload `smartmach.py`, then run `!python smartmach.py`; or paste the code into a notebook
  (each `# %%` line marks a cell). If the download fails, you will be asked to upload `ai4i2020.csv`.
- **VS Code / Jupyter:** put `smartmach.py` in a folder and run `python smartmach.py`. The `# %%` markers also let
  VS Code run it cell by cell.
- **Offline:** download `ai4i2020.csv` from the UCI repository and keep it in the same folder as the script.

The script ends with an interactive prompt asking for machine values. Set `RUN_INTERACTIVE = False` (Section 10) to skip it.

## 4. How it works – section by section

| Section | What happens | Synopsis item covered |
|---|---|---|
| 0. Imports & settings | Libraries, random seed, 80/20 split, risk thresholds, output folder | – |
| 1. Dataset collection | Loads local CSV → else downloads from UCI → else asks for upload | Dataset Collection |
| 2. Preprocessing | Renames columns, checks/removes missing and duplicate rows, drops ID columns and the 5 failure-mode columns, encodes `Type` (L=0, M=1, H=2) | Handling missing/duplicate values, encoding categorical data |
| 3. EDA | Class balance, histograms and boxplots of each sensor split by failure, failure rate per machine type | Exploratory data analysis, analyse factors linked to failure |
| 4. Feature engineering | Adds `Temp_diff` (process − air temperature) and `Power` (torque × speed); correlation heatmap | EDA & Feature Engineering |
| 5. Train/test split | 80% train / 20% test, stratified so both keep the same failure percentage | – |
| 6. Model training | Logistic Regression, Decision Tree, Random Forest, SVM, KNN – each in a Pipeline with `StandardScaler` | Multiple ML models, feature scaling |
| 7. Evaluation | Accuracy, Precision, Recall, F1-score, classification report, confusion matrices, comparison bar chart | Evaluation metrics & visualisations |
| 8. Best model | Chosen by highest F1-score on the failure class | Best-performing model selection |
| 9. Feature importance | Permutation importance (works for any model) + Random Forest built-in importance | Important parameters influencing failure |
| 10. Prediction | `predict_machine()` returns failure yes/no, probability and risk level; sample demos; interactive input with validation | Failure/Risk prediction, Low/Medium/High risk |
| 11. Save | Saves the best model (`joblib`) and the comparison table | – |

## 5. Key design decisions (read before presenting)

1. **No data leakage.** The columns `TWF, HDF, PWF, OSF, RNF` are the individual failure causes that *define*
   `Machine failure`. Using them as inputs would make the models look perfect but useless, so they are dropped.
   `UDI` and `Product ID` are only identifiers and are also dropped.
2. **Scaling inside a Pipeline.** The scaler is fitted on training data only, so test data never influences it.
3. **Imbalanced data.** Failures are only about 3–4% of the real dataset, so accuracy alone is misleading
   (predicting "no failure" every time gives ~96% accuracy). Hence Precision/Recall/F1 are reported for the failure
   class, and the best model is chosen by F1.
4. **Prediction vs risk.** "Failure predicted" means probability ≥ 0.5. The risk level uses the probability bands below.

## 6. Assumptions (not specified in the synopsis)



| Assumption | Where to change |
|---|---|
| Risk bands: **Low < 0.30, Medium 0.30–0.60, High ≥ 0.60** failure probability | `LOW_RISK_MAX`, `MEDIUM_RISK_MAX` in Section 0 |
| `class_weight="balanced"` for Logistic Regression, Decision Tree, Random Forest, SVM (KNN has no such option) | `models` dictionary, Section 6 |
| 80/20 train-test split, random seed 42 | `TEST_SIZE`, `RANDOM_STATE` |
| Model hyper-parameters (tree depth 6, 200 forest trees, KNN k=5) are simple defaults, no tuning | Section 6 |
| Best model chosen by F1-score of the failure class | Section 8 |
| Two engineered features (`Temp_diff`, `Power`) – the synopsis only says "Feature Engineering" | Section 4 |

## 7. Sample output (format only)

```
=== MODEL COMPARISON (test set) ===
                     Accuracy  Precision  Recall  F1-score
Random Forest           ...       ...      ...      ...
...
Best model: Random Forest

=== SAMPLE PREDICTIONS ===
Healthy machine: No failure expected | Probability = 0.00% | Risk = Low Risk
Heavily worn tool: FAILURE LIKELY | Probability = 99.50% | Risk = High Risk
```

Exact numbers depend on the real dataset, so we have run the script and use data which we able to find results in the report.
(The code was tested end to end on a simulated file with the same columns, because the real dataset could not be
downloaded in the development environment. The scores from that test are **not** real results.)

## 8. Limitations

- The AI4I dataset is synthetic, so results show the method working, not real factory performance.
- Very few failure rows exist, so test-set metrics can vary a little with a different random seed.
- Risk thresholds are a design choice, not learned from data.
