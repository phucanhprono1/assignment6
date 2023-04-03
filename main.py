import inline as inline
import matplotlib as matplotlib
import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')
df.info()

print("Nulls")
print("=====")
print(df.isnull().sum())

print("0s")
print("==")
print(df.eq(0).sum())

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0,
                                                                                                                np.NaN)
df.fillna(df.mean(), inplace=True)
print(df.eq(0).sum())
# Check correlation
corr = df.corr()
print(corr)
# Vẽ sơ đồ sự tương quan

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(df.columns), 1)
ax.set_xticks(ticks)
ax.set_xticklabels(df.columns)
plt.xticks(rotation=90)
ax.set_yticklabels(df.columns)
ax.set_yticks(ticks)
# ---print the correlation factor---
for i in range(df.shape[1]):
    for j in range(9):
        text = ax.text(j, i, round(corr.iloc[i][j], 2), ha="center", va="center", color="w")
plt.show()
# Dùng seaborn để vẽ tương quan
import seaborn as sns

sns.heatmap(df.corr(), annot=True)
# ---get a reference to the current figure and set its size---
fig = plt.gcf()
fig.set_size_inches(8, 8)
print(df.corr().nlargest(4, 'Outcome').index)
# ---print the top 4 correlation values---
print(df.corr().nlargest(4, 'Outcome').values[:, 8])
# Đánh giá thuật toán
# Suy diễn lùi
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# ---features---
X = df[['Glucose', 'BMI', 'Age']]
# ---label---
y = df.iloc[:, 8]
log_regress = linear_model.LogisticRegression()
log_regress_score = cross_val_score(log_regress, X, y, cv=10, scoring='accuracy').mean()
print(log_regress_score)

result = []
result.append(log_regress_score)

# K láng giềng gần
from sklearn.neighbors import KNeighborsClassifier

# ---empty list that will hold cv (cross-validates) scores---
cv_scores = []
# ---number of folds---
folds = 10
# ---creating odd list of K for KNN---
ks = list(range(1, int(len(X) * ((folds - 1) / folds)), 2))
# ---perform k-fold cross validation---
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
    cv_scores.append(score)
# ---get the maximum score---
knn_score = max(cv_scores)
# ---find the optimal k that gives the highest score---
optimal_k = ks[cv_scores.index(knn_score)]
print(f"The optimal number of neighbors is {optimal_k}")
print(knn_score)
result.append(knn_score)

# SVM(Support Vector Machine)

# libear
from sklearn import svm

linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm, X, y,
                                   cv=10, scoring='accuracy').mean()
print(linear_svm_score)
result.append(linear_svm_score)

# RBF
rbf = svm.SVC(kernel='rbf')
rbf_score = cross_val_score(rbf, X, y, cv=10, scoring='accuracy').mean()
print(rbf_score)
result.append(rbf_score)

# Đánh giá:
algorithms = ["Logistic Regression", "K Nearest Neighbors", "SVM Linear Kernel", "SVM RBF Kernel"]
cv_mean = pd.DataFrame(result, index=algorithms)
cv_mean.columns = ["Accuracy"]
cv_mean.sort_values(by="Accuracy", ascending=False)

# Training model
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X, y)
import pickle

# ---save the model to disk---
filename = 'diabetes.sav'
# ---write to the file using write and binary mode---
pickle.dump(knn, open(filename, 'wb'))
# ---load the model from disk---
loaded_model = pickle.load(open(filename, 'rb'))
# Test prediction
Glucose = 65
BMI = 70
Age = 50
prediction = loaded_model.predict([[Glucose, BMI, Age]])
print(prediction)
if (prediction[0] == 0):
    print("Non-diabetic")
else:
    print("Diabetic")


proba = loaded_model.predict_proba([[Glucose, BMI, Age]])
print(proba)
print("Confidence: " + str(round(np.amax(proba[0]) * 100 ,2)) + "%")