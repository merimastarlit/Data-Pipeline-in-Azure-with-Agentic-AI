# Project 03: Spam Email Classification

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import BytesIO

from sklearn.decomposition import PCA

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head(5))

# finding out emails are there in the dataset
print(df.shape)


# count how many spam and non-spam emails are there
print(df['spam_label'].value_counts())

# The dataset contains 2788 ham emails (60.6%) and 1813 spam emails (39.4%), 
# which is moderately imbalanced toward ham.
# This matters because a naive model that predicts "not spam" for every email 
# would achieve ~60% accuracy without learning anything useful. Raw accuracy 
# alone is therefore misleading — precision and recall are more informative, 
# especially since false positives (real emails marked as spam) can cause 
# users to miss important messages.


# Boxplot

sns.boxplot(x='spam_label', y='word_freq_free', data=df)
plt.title('Boxplot of Word Frequency "free" by Spam Label')
plt.xlabel('Spam Label (0 = Not Spam, 1 = Spam)')
plt.ylabel('Frequency of "free"')
plt.savefig('assignments_03/outputs/word_freq_free_boxplot.png')
plt.show()

sns.boxplot(x='spam_label', y='char_freq_!', data=df)
plt.title('Boxplot of Character Frequency "!" by Spam Label')
plt.xlabel('Spam Label (0 = Not Spam, 1 = Spam)')
plt.ylabel('Frequency of "!"')
plt.savefig('assignments_03/outputs/char_freq_exclamation_boxplot.png')
plt.show()

sns.boxplot(x='spam_label', y='capital_run_length_total', data=df)
plt.title('Boxplot of Capital Run Length Total by Spam Label')
plt.xlabel('Spam Label (0 = Not Spam, 1 = Spam)')
plt.ylabel('Total Capital Run Length')
plt.savefig('assignments_03/outputs/capital_run_length_total_boxplot.png')
plt.show()

# Most word frequency features are heavily skewed toward zero — the majority of emails
# do not contain most words, so these columns consist mostly of 0.0 with occasional
# high outliers. This is visible in the boxplots where the interquartile range sits
# near zero with long upper whiskers.
#
# Numeric scales also vary widely across features: word frequency values are tiny
# fractions (typically 0.0–1.0), while capital_run_length_total can reach into the
# hundreds or thousands. This matters for models like KNN and Logistic Regression,
# which are sensitive to feature scale — a large-valued feature will dominate distance
# calculations even if it is no more informative than others. This is why we apply
# StandardScaler before those models. Decision Tree and Random Forest are not affected
# by scale because they split on thresholds rather than distances.



# Task 2 Test and train


# we need first drop the spam_label column so that the features are all those other columns, while spam_label is the target
X = df.drop('spam_label', axis=1)
y = df['spam_label']
# splitting the data into training and testing sets, and then scaling the features to better serve PCA afterwards.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# PCA


pca = PCA()
# we need to fit PCA on the training data only, and then we can look at the explained variance ratio to see how much variance each principal component captures. This will help us decide how many components to keep for our model.
pca.fit(X_train_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
# the cumulative variance ratio will show us how much total variance is captured as we include more principal components. This is useful for determining the number of components to retain while still capturing a significant portion of the variance in the data.
cumulative_variance_ratio = explained_variance_ratio.cumsum()

print("Explained Variance Ratio:", explained_variance_ratio)
print("Cumulative Explained Variance Ratio:", cumulative_variance_ratio)

# Plot and save the cumulative explained variance curve to visualize how many
# principal components are needed to capture 95% of the variance in the data.
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.grid()
plt.savefig("assignments_03/outputs/pca_cumulative_variance.png")
plt.show()

n_components_90 = np.argmax(cumulative_variance_ratio >= 0.90) + 1
print(f"Number of components needed to reach 90% variance: {n_components_90}")



# Task 3: A Classifier Comparison

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Classification Report for KNN:")
print(classification_report(y_test, y_pred))
print("Accuracy for KNN:", accuracy_score(y_test, y_pred))


knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("Classification Report for KNN with Scaled Data:")
print(classification_report(y_test, y_pred_scaled))
print("Accuracy for KNN with Scaled Data:",
      accuracy_score(y_test, y_pred_scaled))


pca = PCA()
pca.fit(X_train_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

print("Explained Variance Ratio:", explained_variance_ratio)
print("Cumulative Explained Variance Ratio:", cumulative_variance_ratio)

# number of components to reach 90%
n = np.argmax(cumulative_variance_ratio >= 0.90) + 1
print("Number of components for 90% variance:", n)

# Apply PCA and keep first n components
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)

print("Classification Report for KNN with PCA:")
print(classification_report(y_test, y_pred_pca))
print("Accuracy for KNN with PCA:", accuracy_score(y_test, y_pred_pca))


# Plot cumulative explained variance and save it
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_variance_ratio) + 1),
    cumulative_variance_ratio,
    marker='o'
)
plt.axhline(y=0.90, color='red', linestyle='--', label='90% variance')
plt.axvline(x=n, color='green', linestyle='--', label=f'n = {n}')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('assignments_03/outputs/cumulative_explained_variance.png')
plt.show()


# Decision Tree

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Classification Report for Decision Tree:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy for Decision Tree:", accuracy_score(y_test, y_pred_dt))

# Try with different depths
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print("Depth:", depth)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("-" * 30)

# As depth increases, the model begins to overfit. The tree with no depth limit memorizes the training data, leading to very high training accuracy but lower test accuracy. The best depth is 10 because it achieves the highest test accuracy while still generalizing well.

# Fit final Decision Tree at best depth
dt_final = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_final.fit(X_train, y_train)
y_pred_final = dt_final.predict(X_test)

print("Classification Report for Decision Tree (max_depth=10):")
print(classification_report(y_test, y_pred_final))
print("Accuracy for Decision Tree (max_depth=10):", accuracy_score(y_test, y_pred_final))

# Top 10 feature importances
feature_names = X.columns.tolist()
dt_importances = pd.Series(dt_final.feature_importances_, index=feature_names)
top10_dt = dt_importances.sort_values(ascending=False).head(10)
print("Top 10 features (Decision Tree):")
print(top10_dt)




# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy for Random Forest:", accuracy_score(y_test, y_pred_rf))

# Random Forest performed best because it combines many trees that make different mistakes, which helps reduce random errors and improves overall performance.

# Feature Importances
dt_importances = dt_final.feature_importances_
rf_importances = rf.feature_importances_

rf_importances_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_importances
})

rf_top10 = rf_importances_df.sort_values(
    by="importance", ascending=False).head(10)
print(rf_top10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=rf_top10)
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("assignments_03/outputs/feature_importances.png")
plt.show()

# Do the models agree? Yes, both Decision Tree and Random Forest highlight similar important features.
# Does it match intuition? Yes, the important features (like word_freq_free, char_freq_!, and capital letter usage) match common characteristics of spam emails.


#  Logistic regression

lr = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy for Logistic Regression:", accuracy_score(y_test, y_pred_lr))

# Now with PCA

lr_pca = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_pca.predict(X_test_pca)
print("Classification Report for Logistic Regression with PCA:")
print(classification_report(y_test, y_pred_lr_pca))
print("Accuracy for Logistic Regression with PCA:",
      accuracy_score(y_test, y_pred_lr_pca))

# Conclusions:
# Best model: Random Forest
# Why: It combines many trees that make different mistakes, which helps reduce error and improves overall performance.
# Did PCA help KNN? PCA slightly improved or had similar performance, but it did not outperform Random Forest.
# Did PCA help Logistic Regression? No, PCA slightly reduced performance because some useful information was lost.
# Is accuracy enough? Why or why not? No, accuracy is not enough because it does not capture the difference between false positives and false negatives. In spam detection, marking a real email as spam can be more harmful than letting spam through, so other metrics like precision and recall are important.

# Confusion Matrix for Random Forest


ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf)
plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png")
plt.show()

# Conclusion:
# After analyzing the confusion matrix, the model makes more false negatives than false positives, meaning some spam gets through. However, it keeps false positives low as 9, which is important because marking real emails as spam is more harmful.


# Task 4: Cross Validation


# Decision Tree
dt_cv = DecisionTreeClassifier(max_depth=10, random_state=42)
scores = cross_val_score(dt_cv, X_train, y_train, cv=5)
print("DT Mean:", scores.mean())
print("DT Std:", scores.std())

# Mean tells you the average accuracy of the model across different splits.
# Standard deviation tells you how consistent the model is; a lower value means the model is more stable.

# Random Forest
rf_cv = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
scores = cross_val_score(rf_cv, X_train, y_train, cv=5)
print("RF Mean:", scores.mean())
print("RF Std:", scores.std())

# Logistic Regression (scaled)
lr_cv = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
scores = cross_val_score(lr_cv, X_train_scaled, y_train, cv=5)
print("LR Mean:", scores.mean())
print("LR Std:", scores.std())

# KNN (scaled)
knn_cv = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn_cv, X_train_scaled, y_train, cv=5)
print("KNN Mean:", scores.mean())
print("KNN Std:", scores.std())

# KNN (unscaled)
knn_cv_unscaled = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn_cv_unscaled, X_train, y_train, cv=5)
print("KNN (unscaled) Mean:", scores.mean())
print("KNN (unscaled) Std:", scores.std())

# Logistic Regression with PCA
lr_cv_pca = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
scores = cross_val_score(lr_cv_pca, X_train_pca, y_train, cv=5)
print("LR (PCA) Mean:", scores.mean())
print("LR (PCA) Std:", scores.std())

# KNN with PCA
knn_cv_pca = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn_cv_pca, X_train_pca, y_train, cv=5)
print("KNN (PCA) Mean:", scores.mean())
print("KNN (PCA) Std:", scores.std())


# Task 5: Building a Prediction Pipeline


# Pipeline for best tree-based classifier: Random Forest
# Tree-based models do not need scaling or PCA because they split on feature thresholds.
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf_pipeline = rf_pipeline.predict(X_test)

print("Classification Report for Random Forest Pipeline:")
print(classification_report(y_test, y_pred_rf_pipeline))


# Pipeline for best non-tree-based classifier: Logistic Regression
# Logistic Regression benefits from scaling because feature magnitudes matter.
# PCA is not included because it did not improve performance in Task 3.
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr_pipeline = lr_pipeline.predict(X_test)

print("Classification Report for Logistic Regression Pipeline:")
print(classification_report(y_test, y_pred_lr_pipeline))


# Conclusions:
# The two pipelines do not have the same structure.
# The Random Forest pipeline only contains the classifier because tree-based models
# do not require scaling or PCA.
# The Logistic Regression pipeline includes scaling because it is sensitive to feature magnitudes.
# Packaging models as pipelines is useful because it keeps preprocessing and modeling together,
# reduces mistakes, avoids data leakage, and makes the model easier to reuse, share, or deploy.
