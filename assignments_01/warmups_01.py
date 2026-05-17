# --- Pandas ---
#  # Pandas Q1


from scipy.stats import ttest_ind

import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"Num Rows: {len(df)}")

# Pandas Q2

print(df[(df['passed'] == True) & (df['grade'] > 80)])


# Pandas Q3: Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).

df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Q4: Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.

df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])


# Pandas Q5: Group the DataFrame by "city" and compute the mean grade for each city. Print the result.

print(df.groupby("city")["grade"].mean())

# Pandas Q6: Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.

df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Q7. Sort the DataFrame by "grade" in descending order and print the top 3 rows.

sorted_df = df.sort_values("grade", ascending=False)
print(sorted_df.head(3))

# NumPy Review

# np Q1: Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.

list = [10, 20, 30, 40, 50]

arr1 = np.array(list)

print("Array:", arr1)
print("Shape:", arr1.shape)
print("Data type:", arr1.dtype)
print("Dimensions:", arr1.ndim)

# np Q2: Create the following 2D array and print its shape and size (total number of elements).

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Array:", arr)
print("Shape:", arr.shape)
print("Size:", arr.size)


# np Q3: Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].

print(arr[0:2, 0:2])

# np Q 4: Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.

zeros = np.zeros((3, 4))
print(zeros)

ones = np.ones((2, 5))

print(ones)

# np Q5: Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.

aranged = np.arange(0, 50, 5)

print("Array:", aranged)
print("Shape:", aranged.shape)
print("Mean:", aranged.mean())
print("Sum:", aranged.sum())
print("Standard Deviation:", aranged.std())

# np Q6: Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.

random_data = np.random.normal(0, 1, 200)
print("Mean:", random_data.mean())
print("Standard Deviation:", random_data.std())

# Matplotlib Review

# Matplotlib Q. 1: Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y"

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]


plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Matplotlib Question 2: Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.

subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.bar(subjects, scores, color='blue')
plt.title("Subject Scores")
plt.xlabel("Subjets")
plt.ylabel("Scores")
plt.show()


# Matplotlib Question 3: Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="orange", label="Dataset 1")
plt.scatter(x2, y2, color="blue", label="Dataset 2")
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Matplotlib Question 4:Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.


# Example data (replace with your actual Q1 and Q2 data)
# Q1 data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Q2 data
subjects = ['Math', 'Science', 'English', 'History']
scores = [85, 90, 78, 88]

# Create figure with 1 row and 2 subplots
fig, axs = plt.subplots(1, 2)

# Left subplot: line plot (x vs y)
axs[0].plot(x, y)
axs[0].set_title("Line Plot (Q1)")

# Right subplot: bar plot (subjects vs scores)
axs[1].bar(subjects, scores)
axs[1].set_title("Bar Plot (Q2)")

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# Descriptive Statistics Review
# Descriptive Stats Question 1: Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
data = np.array(data)

print("Mean:", data.mean())
print("Median:", np.median(data))
print("Variance:", data.var())
print("Standard Deviation:", data.std())

# Descriptive Stats Question 2: Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.

random_data = np.random.normal(65, 10, 500)
print("Mean:", random_data.mean())
print("Standard Deviation:", random_data.std())


plt.hist(random_data, bins=20, color="purple", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Question 3:Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison"

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()


# Descptive Stats Q4:

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data, skewed_data], labels=[
            "Normal Distribution", "Skewed Distribution"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()
# The exponential distribution is more skewed.
# The median is better for skewed data, while the mean works well for normal data.

# Descriptive Stats Q5: Print the mean, median, and mode of the following:

data1 = [10, 12, 12, 16, 18]

data2 = [10, 12, 12, 16, 150]

print("Data1 Mean:", np.mean(data1))
print("Data1 Median:", np.median(data1))

# to get mede, we can use pandas Series mode function, which returns a Series of the most frequent value(s). We take the first one with [0] to get the mode as a single value.
print("Data1 Mode:", pd.Series(data1).mode()[0])
print("Data2 Mode:", pd.Series(data2).mode()[0])

print("Data2 Mean:", np.mean(data2))
print("Data2 Median:", np.median(data2))

# The mean is much higher in data2 because the outlier (150) pulls it up a lot.
# The median is less affected because it only depends on the middle value.


# Hypothesis Testing Review

# Hypothesis Question 1 and 2: Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.


group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)

print("t-statistic:", t_stat)
print("p-value:", p_val)

if p_val < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")


# Hypothesis Question 3: Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.

before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]


t_stat, p_val = stats.ttest_rel(before, after)
print("t-statistic:", t_stat)
print("p-value:", p_val)

if p_val < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")


# Hypothesis Question 4: Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.

scores = [72, 68, 75, 70, 69, 74, 71, 73]


t_stat, p_val = stats.ttest_1samp(scores, 70)
print("t-statistic:", t_stat)
print("p-value:", p_val)


# Hypothesis Question 5: Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.

t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative="less")
print("One-tailed p-value:", p_val)

# Hypothesis Question 6: Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.

print("The average score of Group A is significantly lower than that of Group B, with a p-value of indicating that this difference is unlikely to be due to random chance.")


# Correlation Review

# Correlation Question 1: Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print("Correlation Matrix:\n", corr_matrix)
print("Correlation Coefficient:", corr_matrix[0, 1])
# The correlation coefficient is 0.9, indicating a positive linear relationship between x and y.


# Correlation Question 2: Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p = pearsonr(x, y)
print("Correlation coefficient:", round(r, 2))
print("p-value:", round(p, 4))


# Correlation Question 3: Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

corr = df.corr()
print(corr)


# Correlation Question 4: Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y, color='teal')
plt.title("Negative Correlation")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()

# Correlation Question 5:Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Pipelines Question 1:


arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
               18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


# create_series(arr) : takes a NumPy array and returns a pandas Series with the name "values"

def create_series(nparray):
    series = pd.Series(nparray, name="values")
    return series


# clean_data(series) : takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.

def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series


# summarize_data(series) -- takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode"


def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary


def data_pipeline(array):
    series = create_series(array)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary


if __name__ == "__main__":
    result = data_pipeline(arr)
    for key, value in result.items():
        print(key, value)
