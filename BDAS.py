# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
## step2

file = r"C:\A3\dataset.xlsx"

import pandas as pd

df = pd.read_excel(file, 'dataset')

df.shape

df.head()

df.info()

df_desc = df.describe().T



import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df, x='Target')


ds = sns.countplot(df, x='Curricular units 2nd sem (grade)')
sns.displot(df, x="Curricular units 2nd sem (grade)")


sns.displot(df, x="Curricular units 2nd sem (grade)", hue="Gender")
// plt.ylabel("Gender (0=Female, 0=Male)")

sns.relplot(data=df, x="Curricular units 2nd sem (grade)", y="Age at enrollment")

sns.displot(df, x="Target", hue="Scholarship holder")

import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

## step3

df = df.fillna(999)
df.info()

df = df.drop("Nacionality", axis='columns')
info=df.info()

df = df[(df["Curricular units 1st sem (enrolled)"] > 0)
   |
   (df["Curricular units 2nd sem (enrolled)"] > 0)]
df = df[df["Target"] != "Enrolled"]
uselesscolumns = [
    "Application mode",
    "Application order",
    "Tuition fees up to date",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    ]
for x in uselesscolumns:
    df = df.drop(x, axis='columns')
df.info()

g = df.groupby('Target')
bdf = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
sns.countplot(bdf, x='Target')


df = df.drop("Inflation rate", axis='columns')

from sklearn import utils
from pprint import pprint

classifiers = [est[0] for est in utils.all_estimators(type_filter="classifier")]
print("Classifiers:")
print(classifiers)

regressors = [est[0] for est in utils.all_estimators(type_filter="regressor")]
print("\nRegressors:")
print(regressors)

socialFeaturecols = [
    'Unemployment rate',
    'GDP'
    ]
personalFeaturecols = [
    'Daytime/evening attendance',
    'Educational special needs',
    'Scholarship holder',
    'Age at enrollment',
    'Previous qualification'
    ]
xSocialForGrade = bdf.loc[:, socialFeaturecols]
xPersonalForGrade = bdf.loc[:, personalFeaturecols]

xSocialForGrade.head()
xPersonalForGrade.head()

yGrade = bdf['Total Grade']
yGrade.head()

socialFeaturecols = [
    'Unemployment rate',
    'GDP',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
    ]
personalFeaturecols = [
    'International',
    'Educational special needs',
    'Marital status',
    'Age at enrollment',
    'Previous qualification',
    'Debtor',
    'Gender',
    'Course'
    ]
xSocialForTarget = bdf.loc[:, socialFeaturecols]
xPersonalForTarget = bdf.loc[:, personalFeaturecols]

xSocialForTarget.head()
xPersonalForTarget.head()

yTarget = bdf['Target']
yTarget.head()

from sklearn.model_selection import train_test_split

xSocialForGradeTrain, xSocialForGradeTest, yGradeSocialTrain, yGradeSocialTest = train_test_split(xSocialForGrade, yGrade, test_size=0.5)
xPersonalForGradeTrain, xPersonalForGradeTest, yGradePersonalTrain, yGradePersonalTest = train_test_split(xPersonalForGrade, yGrade, test_size=0.5)
xSocialForTargetTrain, xSocialForTargetTest, yTargetSocialTrain, yTargetSocialTest = train_test_split(xSocialForTarget, yTarget, test_size=0.5)
xPersonalForTargetTrain, xPersonalForTargetTest, yTargetPersonalTrain, yTargetPersonalTest = train_test_split(xPersonalForTarget, yTarget, test_size=0.5)

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, r2_score

ridge = Ridge(alpha=1.0)

ridge.fit(xPersonalForGradeTrain, yGradePersonalTrain)

y_pred = ridge.predict(xPersonalForGradeTest)

mse = mean_squared_error(yGradePersonalTest, y_pred)
r2 = r2_score(yGradePersonalTest, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")



ridge = Ridge(alpha=1.0)

ridge.fit(xSocialForGradeTrain, yGradeSocialTrain)

y_pred = ridge.predict(xSocialForGradeTest)

mse = mean_squared_error(yGradeSocialTest, y_pred)
r2 = r2_score(yGradeSocialTest, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")



from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

linear_svc = LinearSVC(random_state=32)

linear_svc.fit(xPersonalForTargetTrain, yTargetPersonalTrain)

y_pred = linear_svc.predict(xPersonalForTargetTest)

accuracy = accuracy_score(yTargetPersonalTest, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(yTargetPersonalTest, y_pred))


print("Coefficients:", linear_svc.coef_)
xPersonalForTargetTrain.info()

sns.displot(df, hue="International",x="Target" )

sns.relplot(df, x="Debtor", y="Gender")


linear_svc = LinearSVC(random_state=32)

linear_svc.fit(xSocialForTargetTrain, yTargetSocialTrain)

y_pred = linear_svc.predict(xSocialForTargetTest)

accuracy = accuracy_score(yTargetSocialTest, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(yTargetSocialTest, y_pred))

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery-nogrid')


sns.boxplot(df, x="Gender", y="Total Grade")
sns.boxplot(df, x="Debtor", y="Total Grade")

sns.boxplot(df, x="International", y="Total Grade")
sns.boxplot(df, x="Educational special needs", y="Total Grade")

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(yTargetPersonalTest, y_pred)
auc_value = auc(fpr, tpr)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(xPersonalForTargetTrain, yTargetPersonalTrain)

y_pred = knn.predict(xPersonalForTargetTest)

accuracy = accuracy_score(yTargetPersonalTest, y_pred)
print("Accuracy:", accuracy)

report = classification_report(yTargetPersonalTest, y_pred)
print("Classification Report:\n", report)





corr = df.corr(numeric_only=True)
df["Total Grade"] = df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]
file1 = r"C:\A3\dataset_to_be_merge_1.xlsx"
file2 = r"C:\A3\dataset_to_be_merge_2.xlsx"
df1 = pd.read_excel(file1, 'Sheet1')
df2 = pd.read_excel(file2, 'Sheet1')
df1.shape
df2.shape
dfm = pd.concat([df1, df2])
dfm.shape
