import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataset = CompasDataset()

privileged_groups = [{'race': 1}]  
unprivileged_groups = [{'race': 0}]  

metric_orig = BinaryLabelDatasetMetric(dataset, 
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

print("Disparate Impact:", metric_orig.disparate_impact())
print("Statistical Parity Difference:", metric_orig.statistical_parity_difference())

df = dataset.convert_to_dataframe()[0]
X = df.drop(columns=['two_year_recid', 'race'])
y = df['two_year_recid']

X_scaled = StandardScaler().fit_transform(X)

clf = LogisticRegression(solver='liblinear')
clf.fit(X_scaled, y)

y_pred = clf.predict(X_scaled)

dataset_pred = dataset.copy()
dataset_pred.labels = y_pred.reshape(-1, 1)

class_metric = ClassificationMetric(dataset, dataset_pred,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)

fpr_priv = class_metric.false_positive_rate(privileged=True)
fpr_unpriv = class_metric.false_positive_rate(privileged=False)

plt.bar(['Privileged (White)', 'Unprivileged (Black)'], [fpr_priv, fpr_unpriv], color=['blue', 'red'])
plt.title('False Positive Rate by Race')
plt.ylabel('FPR')
plt.savefig('fpr_by_race.png')
plt.close()

RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset)

df_transf = dataset_transf.convert_to_dataframe()[0]
X_rw = df_transf.drop(columns=['two_year_recid', 'race'])
y_rw = df_transf['two_year_recid']
X_rw_scaled = StandardScaler().fit_transform(X_rw)
clf.fit(X_rw_scaled, y_rw)
y_rw_pred = clf.predict(X_rw_scaled)

dataset_rw_pred = dataset_transf.copy()
dataset_rw_pred.labels = y_rw_pred.reshape(-1, 1)
class_metric_rw = ClassificationMetric(dataset_transf, dataset_rw_pred,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

print("FPR after reweighing:")
print("Privileged:", class_metric_rw.false_positive_rate(privileged=True))
print("Unprivileged:", class_metric_rw.false_positive_rate(privileged=False))
