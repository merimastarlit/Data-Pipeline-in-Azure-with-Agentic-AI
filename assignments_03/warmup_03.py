

import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Create output folder for saved figures.
os.makedirs("outputs", exist_ok=True)

# --- Preprocessing ---


# Preprocessing Question 1: Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42. Print the shapes of all four arrays.


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Preprocessing Q1")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print()

# Preprocessing Question 2: Fit a StandardScaler on X_train and use it to transform both X_train and X_test. Print the mean of each column in X_train_scaled -- they should all be very close to 0. Add a comment explaining in one sentence why you fit the scaler on X_train only.


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing Q2")
print("Column means of X_train_scaled:", X_train_scaled.mean(axis=0))
# Fit the scaler on X_train only so information from the test set does not leak into training.
print()

# ---KNN----

# KNN Question 1: Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), and predict on the test set. Print the accuracy score and the full classification report.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Q1")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification report:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))
print()

#  KNN Qestion 2: Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). Print the accuracy score. Add a comment: does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
accuracy_knn_scaled = accuracy_score(y_test, y_pred_knn_scaled)
print("KNN Accuracy with Scaling:", accuracy_knn_scaled)
# Scaling improves performance because KNN relies on distance calculations, and scaling ensures that all features contribute equally to the distance metric. In this dataset, the features may be on different scales, so scaling helps KNN make more accurate predictions.


# KNN Question 3: Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. Print each fold score, the mean, and the standard deviation. Add a comment: is this result more or less trustworthy than a single train/test split, and why?

knn_cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print("KNN Cross-Validation Scores:", knn_cv_scores)
print("Mean CV Score:", knn_cv_scores.mean())
print("Standard Deviation of CV Scores:", knn_cv_scores.std())
# This result is more trustful than a single train/test split because it evaluates the model's performance across multiple subsets of the data. Cross-validation provides a more robust estimate of the model's generalization performance.


# KNN Question 4: Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. For each, compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score. Add a comment identifying which k you would choose and why.

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
mean_scores = {}

print("KNN Q4")
for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(
        n_neighbors=k), X_train, y_train, cv=5)
    mean_scores[k] = scores.mean()
    print(f"k={k}: mean CV score={scores.mean():.4f}")

best_k = max(mean_scores, key=mean_scores.get)
print("Chosen k:", best_k)
# I would choose k=5 because it ties for the best mean CV score and is a moderate value that is less noisy than k=1 while still performing at the top.
print()


# Classifier Evaluation Question 1: Using your predictions from KNN Question 1, create a confusion matrix and display it with ConfusionMatrixDisplay, passing display_labels=iris.target_names. Save the figure to outputs/knn_confusion_matrix.png. Add a comment: which pair of species does the model most often confuse (if any)?


cm = confusion_matrix(y_test, y_pred_knn)
print("Classifier Evaluation Q1")
print("Confusion matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png", bbox_inches="tight")
plt.close()

# The model most often confuses versicolor and virginica, with one versicolor predicted as virginica.
print("Saved confusion matrix figure to outputs/knn_confusion_matrix.png")


# ---The sklearn API: Decision Trees----
# Decision Trees Question 1
# Create a DecisionTreeClassifier(max_depth=3, random_state=42), fit it on the unscaled training data, and predict on the test set. Print the accuracy score and classification report. Add a comment comparing the Decision Tree accuracy to KNN. Then add a second comment: given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?


dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Trees Q1")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification report:")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))
# The Decision Tree accuracy is slightly lower than the unscaled KNN accuracy here, since KNN reached 1.0 on this test split.
# Scaled vs. unscaled data should not meaningfully affect a Decision Tree because trees split on feature thresholds rather than distance calculations.


# --- Logistic Regression ---

# Question 1:


C_values = [0.01, 1.0, 100]

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    coef_magnitude = np.abs(model.coef_).sum()
    print(f"C={C}: total coefficient magnitude={coef_magnitude:.4f}")

# As C increases, the total coefficient magnitude increases.
# This shows that weaker regularization (larger C) allows the model to use larger coefficients,
# while stronger regularization (smaller C) shrinks coefficients toward zero to prevent overfitting.


# --- PCA ---

# Question 1:

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images = digits.images  # same data shaped as 8x8 images for plotting

# printing shapes to confirm data loading
print("PCA Q1")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

# subplot the first 10 digits in the dataset
fig, axes = plt.subplots(1, 10, figsize=(15, 2.5))
# Loop over digits 0-9, find the first index of each digit in y_digits, and plot the corresponding image from images.
for digit in range(10):
    # Find the index of the first occurrence of the current digit in y_digits
    idx = np.where(y_digits == digit)[0][0]
    # Plot the corresponding image in the appropriate subplot
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(str(digit))
    # Turn off axes for a cleaner look
    axes[digit].axis('off')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("outputs/sample_digits.png", bbox_inches="tight")
plt.close()

print("Saved figure to outputs/sample_digits.png")


# Question 2:

# Fit PCA on the digit data and transform it to get the scores. Then create a scatter plot of the first two principal components, coloring points by their digit label.
pca = PCA()
# Fit PCA on the digit data and transform it to get the scores.
pca.fit(X_digits)
# Create a scatter plot of the first two principal components, coloring points by their digit label.
scores = pca.transform(X_digits)

print("PCA Q2")
plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1],
                      c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection of Digits")
plt.savefig("outputs/pca_2d_projection.png", bbox_inches="tight")
plt.close()

# Yes, same-digit images tend to cluster together in this 2D space, although there is still some overlap between classes.

print("Saved figure to outputs/pca_2d_projection.png")


# Question 3:

# Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components using np.cumsum(pca.explained_variance_ratio_). Save to outputs/pca_variance_explained.png.

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by PCA Components")
plt.grid(True)
plt.savefig("outputs/pca_variance_explained.png", bbox_inches="tight")
plt.close()

components_80 = np.argmax(cumulative_variance >= 0.80) + 1
print("Components needed for about 80% variance:", components_80)

# Approximately 13 components are needed to explain 80% of the variance.
print("Saved figure to outputs/pca_variance_explained.png")


# Question 4:

# To reconstruct a digit using the first n_components principal components, we can start with the mean image and then add back the contributions from the selected principal components. The contribution of each component is given by the score for that component multiplied by the corresponding eigenvector (principal component). We can loop over the first n_components and sum these contributions to get the reconstructed image.

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + \
            scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


n_values = [2, 5, 15, 40]
fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(10, 10))

for col in range(5):
    axes[0, col].imshow(images[col], cmap='gray_r')
    axes[0, col].set_title(f"Original\nLabel {y_digits[col]}")
    axes[0, col].axis('off')

for row, n in enumerate(n_values, start=1):
    for col in range(5):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap='gray_r')
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(f"n={n}")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png", bbox_inches="tight")
plt.close()

print("PCA Q4")
print("Saved figure to outputs/pca_reconstructions.png")

# The digits become clearly recognizable around n=15, and that roughly matches the point where the variance curve starts to level off.
