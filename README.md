# Disease risk: Logistic Regression and statistical baselines

The project uses Logistic Regression as its only predictive model. Run the
pipeline from `src` in this order:

```powershell
python split_data.py
python train.py
python evaluate.py
python statistical_analysis.py
python generate_shap_reports.py
```

`evaluate.py` compares Logistic Regression with a constant training-prevalence
baseline using ROC-AUC, PR-AUC, F1, Brier score, calibration, sensitivity, and
specificity. The classification threshold is selected on validation data and
then applied unchanged to the test data.

`statistical_analysis.py` uses the training split only and writes descriptive
statistics, classical group tests, and univariable/multivariable Logistic
Regression tables (odds ratios, 95% confidence intervals, and p-values) to
`results/statistical_analysis`.
