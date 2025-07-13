# AI-Week-7

# COMPAS Recidivism Bias Audit 🔍

This project audits racial bias in the COMPAS Recidivism Dataset using IBM's [AI Fairness 360 (AIF360)](https://aif360.mybluemix.net/) toolkit. It includes analysis, bias metrics, visualizations, and a mitigation strategy using reweighing.

## Contents

- `compas_bias_audit.py` – Python script for dataset analysis and bias mitigation.
- `compas_audit_report.txt` – A 300-word summary report of key findings and recommendations.
- `fpr_by_race.png` – Visualization showing disparity in false positive rates (generated when running the script).

## Objective

To detect and mitigate racial bias in COMPAS risk scores — specifically disparities in **false positive rates** between African-American and Caucasian individuals.

## Setup & Installation

Install the required libraries:

pip install aif360 pandas numpy scikit-learn matplotlib seaborn

Make sure you have the latest version of AIF360, and that the COMPAS dataset loads correctly via its built-in loader.

## Usage
Run the audit script:

python compas_bias_audit.py
The script will:

Load and process the COMPAS dataset.

Calculate fairness metrics (Statistical Parity, Disparate Impact, False Positive Rate).

Generate a bar chart comparing FPR across racial groups.

Apply reweighing to reduce bias and evaluate the model again.

## Findings Summary
African-American individuals have a significantly higher false positive rate than Caucasians.

Disparate Impact and Statistical Parity metrics also show unfair treatment.

The reweighing technique reduces this disparity, demonstrating improved fairness.

## Recommendations
Avoid using biased models in high-stakes decisions like parole.

Always audit datasets for fairness before model deployment.

Apply preprocessing techniques (like reweighing) and monitor fairness metrics continually.

## Report
See compas_audit_report.txt for a concise 300-word summary of findings and ethical recommendations.

Author: [Karabo J Masipa]
Toolkit: IBM AI Fairness 360
