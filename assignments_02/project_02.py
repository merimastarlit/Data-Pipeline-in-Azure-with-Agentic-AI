import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Task 1

base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/student_performance_math.csv"

# Load the dataset into a pandas DataFrame. The csv file is separated by semicolons (;), so we need to make sure to specify the correct separator when reading the file.
df = pd.read_csv(base_url, sep=';')

print(df.shape)
print(df.head(5))
print(df.dtypes)

plt.hist(df['G3'], bins=21, color='blue', alpha=0.7)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.savefig("assignments_02/outputs/g3_distribution.png")

# We see a distinct bar at G3 = 0 representing students who did not take the final exam.
plt.show()

# Task 2: Preprocess the data

filtered_df = df[df["G3"] != 0]
print(df.shape)
print(filtered_df.shape)

# Keeping G3 = 0 rows would make the model think that more absences lead to low grades, even though those students didn’t actually take the exam, so their grades don’t reflect their true performance.

# Convert categorical variables to numeric using one-hot encoding or label encoding as appropriate.
cols = ["schoolsup", "internet", "higher", "activities"]

for col in cols:
    filtered_df[col] = filtered_df[col].replace({"yes": 1, "no": 0})


filtered_df["sex"] = filtered_df["sex"].replace({"M": 1, "F": 0})

# compare absences with G3
corr_original = df["absences"].corr(df["G3"])
corr_filtered = filtered_df["absences"].corr(filtered_df["G3"])
print(
    f"Correlation between absences and G3 in original dataset: {corr_original:.4f}")
print(
    f"Correlation between absences and G3 in filtered dataset: {corr_filtered:.4f}")

# corr_filtered is more trustworthy because it only includes students who took the exam, so the relationship between absences and grades reflects actual performance instead of missing or misleading data.


# scatter plot of absences vs G3
plt.scatter(filtered_df["absences"],
            filtered_df["G3"], color="blue", alpha=0.7)
plt.title("Absences vs Final Grade (G3)")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.grid()
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.show()


# Task 3: Exploratory Data Analysis

g3_corrs = filtered_df.corr()["G3"].sort_values()
print(g3_corrs)

# The feature with the strongest positive correlation with G3 is G2 (second period grade),
# which makes sense as prior performance is the best predictor of final grade.
# The feature with the strongest negative correlation is failures.
# A surprising finding is that studytime has a weaker correlation with G3 than expected,
# suggesting that time spent studying alone doesn't strongly predict final grade.

# a) first plot: studytime vs G3
plt.scatter(filtered_df["studytime"], filtered_df["G3"])
plt.title("Study Time vs Final Grade (G3)")
plt.xlabel(
    "Study Time (1=less than 2 hours, 2=2 to 5 hours, 3=5 to 10 hours, 4=more than 10 hours)")
plt.ylabel("Final Grade (G3)")
plt.grid()
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.show()

# The scatter plot shows almost no clear relationship between study time and G3.
# Grades are widely spread across all study time categories (1–4),
# suggesting study time alone is a poor predictor of final grade.


# b) second plot: failures vs G3

plt.scatter(filtered_df["failures"], filtered_df["G3"])
plt.title("Failures vs Final Grade (G3)")
plt.xlabel("Number of Failures")
plt.ylabel("Final Grade (G3)")
plt.grid()
plt.savefig("assignments_02/outputs/failures_vs_g3.png")
plt.show()

# The scatter plot shows a negative relationship because students with more failures tend to have lower final grades, and as failures increase, grades tend to decrease.


# Task 4: Baseline Model

x = filtered_df[["failures"]]
y = filtered_df["G3"]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Linear Regression Question 3

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(x_test, y_test)

print("RMSE:", rmse)
print("R² on the test set:", r2)

# For each additional failure, the final grade decreases by about 1.43 points
# An RMSE of about 3 means the model’s predictions are somewhat off, since it can miss the actual grade by around 3 points on a 0–20 scale.
# An R² of about 0.20 means that failures explain only about 20% of the variation in final grades, so there are likely many other factors influencing student performance that are not captured by this model.
# The model is weak because using only one feature (failures) is too limited, and it does not capture other important factors that affect student performance.


# Task 5: Final Model

# With no G1

# Feature selection: failures, Medu, Fedu, studytime, higher, schoolsup, internet
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]

X = filtered_df[feature_cols].values
y = filtered_df["G3"].values

# Split the data into training and test sets using an 80/20 split and random_state=42.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a linear regression model using the training set and evaluate it on the test set. Print the R² score for both the training and test sets, as well as the RMSE for the test set. Also, print the coefficients of each feature in the model.
model_final = LinearRegression()
model_final.fit(X_train, y_train)
y_pred_final = model_final.predict(X_test)

rmse_final = np.sqrt(np.mean((y_pred_final - y_test) ** 2))
r2_train = model_final.score(X_train, y_train)
r2_final = model_final.score(X_test, y_test)

print("Train R²:", r2_train)
print("Test R²:", r2_final)
print("RMSE:", rmse_final)

for name, coef in zip(feature_cols, model_final.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Interpretation:
# Adding more features slightly improved the model compared to the baseline from Task 4.
# The negative coefficient for schoolsup likely reflects that students receiving extra support
# are already struggling academically. Internet access has a positive coefficient, possibly
# because it provides better access to learning resources, though this does not imply causation.
# Train and test R² are close, suggesting no significant overfitting, but the overall R²
# remains low — most variation in grades is driven by factors not captured in this model.
# For production, I would keep failures, Medu, and internet as they show the clearest
# signal. Features like activities and freetime contribute little and could be dropped.



# Task 6: Evaluate and Summarize

plt.scatter(y_pred_final, y_test, color="blue", alpha=0.7)
plt.plot([0, 20], [0, 20], color="red")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Final Grade (G3)")
plt.ylabel("Actual Final Grade (G3)")
plt.grid()
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.show()

# Interpretation:

# The points are not tightly clustered around the diagonal, showing that the model is not very precise. The model struggles more at the high end, where it tends to underestimate student performance. A point above the diagonal means the actual grade is higher than the predicted grade (underestimation), while a point below the diagonal means the model overestimates the grade.

# The filtered dataset contains 357 rows, and the test set contains 72 rows after the train-test split.

# The model has an RMSE of about 2.86, which means predictions are typically off by around 3 points on a 0–20 scale. This is a moderate error and shows the model is not very precise. The R² value of about 0.15 means the model explains only about 15% of the variation in student grades, so most of the variation is due to other factors not included in the model.

# The feature with the largest negative coefficient is schoolsup, meaning students receiving extra school support tend to have lower predicted grades. This likely reflects that these students are already struggling academically. The feature with the largest positive coefficient is internet, meaning students with internet access tend to have higher predicted grades, possibly due to better access to learning resources.

# One surprising result is that school support has a negative coefficient. This is unexpected because support should help students, but it likely reflects that students receiving support are those who are already performing poorly.


# Task 7: Neglected Feature with the Power of G1

# feature selection: failures, Medu, Fedu, studytime, higher, schoolsup, internet, G1
feature_cols_g1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                   "internet", "sex", "freetime", "activities", "traveltime", "G1"]

# Split the data into training and test sets using an 80/20 split and random_state=42.
X_g1 = filtered_df[feature_cols_g1].values

# Split the data into training and test sets using an 80/20 split and random_state=42.
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)
y_pred_g1 = model_g1.predict(X_test_g1)

# Calculate R² for the test set with G1 included
r2_with_g1 = model_g1.score(X_test_g1, y_test_g1)

# Print the R² score for the test set with G1 included
print("Test R² with G1:", r2_with_g1)

# Adding G1 dramatically increases R², jumping from roughly 0.15 to ~0.80.
# This does not mean G1 causes G3 — G1 is simply an earlier measurement of the same
# underlying academic ability, so they are naturally highly correlated.
# This model is not useful for early intervention because G1 is only available
# after the first grading period, by which point students may already be struggling.
# Educators wanting to intervene earlier would need a model built solely from
# background and behavioral features available before the school year begins,
# such as failures, parental education, and study time.