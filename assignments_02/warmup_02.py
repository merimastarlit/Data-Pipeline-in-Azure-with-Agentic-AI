# --- scikit-learn API ---

from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LinearRegression


# scikit-learn Question 1
# Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.


years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# Create and fit the model, then predict salaries for 4 and 8 years of experience
experience = np.array([4, 8]).reshape(-1, 1)

model = LinearRegression()                    # 1. create model
model.fit(years, salary)                 # 2. fit model to data (learn)
# 3. predict with new data
query_experience = model.predict(experience)

print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

print(f"Predicted salary for 4 years’ experience: {query_experience[0]:.2f}")
print(f"Predicted salary for 8 years’ experience: {query_experience[1]:.2f}")


# scikit-learn Question 2: Why to have 2d in X shape.

x = np.array([10, 20, 30, 40, 50])

print(x.shape)
# Reshape x to be 2D

reshaped_x = x.reshape(-1, 1)
print(reshaped_x.shape)

# Sckit-learn expects the input features (X) to be in a 2D array format, because it is a matrix of shape, so needs num_samples and num_features, otherwise it won't work. if we give only one array, it won't know is it a sample of features or a single feature, so it needs to be reshaped to 2D.


# scikit-learn Question 3


X_clusters, _ = make_blobs(n_samples=120, centers=3,
                           cluster_std=0.8, random_state=7)

print(X_clusters.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# Left plot: the raw, unlabeled data as the algorithm first sees it
ax1.scatter(X_clusters[:, 0], X_clusters[:, 1], color='gray', s=60, alpha=0.7)
ax1.set_title("Raw Data (No Labels)")
ax1.set_xlabel("Hours (synthetic scale)")
ax1.set_ylabel("Social Time (synthetic scale)")

# Right plot: what K-Means discovers
kmeans = KMeans(n_clusters=3, random_state=42)  # 1. Create the model
# 2. Fit -- find cluster centers
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
# 3. Predict a label for each point
print(kmeans.cluster_centers_)
points = np.bincount(labels)
print(f"Number of points in each cluster: {points}")

# Right plot: what K-Means discovers
ax2.scatter(X_clusters[:, 0], X_clusters[:, 1],
            c=labels, cmap='viridis', s=60, alpha=0.7)

# Adding cluster centers as black X's
ax2.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X', color='black', s=200
)

# Adding titles and labels to the right plot
ax2.set_title("Student Clusters Found by K-Means")
ax2.set_xlabel("Study Hours (synthetic scale)")
ax2.set_ylabel("Social Time (synthetic scale)")


# save the figure to outputs/kmeans_clusters.png and show it
plt.tight_layout()
plt.savefig("outputs/kmeans_clusters.png")
plt.show()


# Linear Regression Question 1:


np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis. Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.

plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm", edgecolor="k", alpha=0.7)
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.tight_layout()
plt.savefig("outputs/cost_vs_age.png")
plt.show()

# The non-smokers (c=0, cool colors) form a lower‐cost cluster,
# while smokers (c=1, warm colors) are distinctly higher cost—
# smoking status is clearly a strong predictor of medical cost.


# Linear Regression Question2: Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.


reshaped_age = age.reshape(-1, 1)


reshaped_age_train, reshaped_age_test, y_train, y_test = train_test_split(
    reshaped_age, cost, test_size=0.2, random_state=42
)
print(f"Shape of reshaped_age_train: {reshaped_age_train.shape}")
print(f"Shape of reshaped_age_test: {reshaped_age_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Linear Regression Question 3

model = LinearRegression()
model.fit(reshaped_age_train, y_train)
y_pred = model.predict(reshaped_age_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_age_only = model.score(reshaped_age_test, y_test)

print("RMSE:", rmse)
print("R² on the test set:", r2_age_only)

# The slope represents the relationship between age and medical cost.
# For each additional year of age, the model predicts that medical cost increases by about $196 dollars. This suggests that older patients tend to have higher medical costs on average.


# Linear Regression Question 4: Now add smoker as a second feature and fit a new model.
X_full = np.column_stack([age, smoker])

X_full_train, X_full_test, y_train, y_test = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)
print(f"Shape of X_full_train: {X_full_train.shape}")
print(f"Shape of X_full_test: {X_full_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

model_full = LinearRegression()
model_full.fit(X_full_train, y_train)
y_pred_2 = model_full.predict(X_full_test)

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
print("Intercept:", model_full.intercept_)

rmse = np.sqrt(np.mean((y_pred_2 - y_test) ** 2))
r2_with_smoker = model_full.score(X_full_test, y_test)

# The smoker coefficient represents the difference in medical cost between
# smokers and non-smokers. Since smoker is 0 or 1, being a smoker increases
# the predicted medical cost by about 15,000 dollars, holding age constant.

print("RMSE:", rmse)

print("R² without smoker:", r2_age_only)
print("R² with smoker:   ", r2_with_smoker)


# Interpretation:
#  - Each additional year of age increases expected cost by about $200.
#  - Being a smoker (smoker=1 versus 0) increases expected cost by about $15,000,
#    holding age constant.  This spike confirms smoking status is the dominant driver of cost.

# Linear Regression Question 5:

# Create a scatter plot of predicted vs actual costs for the model with both features. Add a diagonal line to indicate perfect predictions. Save the figure to outputs/predicted_vs_actual.png.
max_val = max(max(y_test), max(y_pred_2))
min_val = min(min(y_test), min(y_pred_2))

plt.figure(figsize=(8, 8))

# Scatter plot of predicted vs actual costs
plt.scatter(y_pred_2, y_test, color="blue", alpha=0.7)

plt.plot([min_val, max_val], [min_val, max_val], color="red")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")

# Points above the diagonal: actual cost > predicted cost (underestimation)
# Points below the diagonal: actual cost < predicted cost (overestimation)
plt.show()
