from prefect.logging import get_run_logger
from scipy.stats import ttest_ind
from prefect import task, flow
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Pipelines Question 2:

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan,
               18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


# create_series(arr) : takes a NumPy array and returns a pandas Series with the name "values"
@task
def create_series(nparray):
    series = pd.Series(nparray, name="values")
    return series


# clean_data(series) : takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
@task
def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series


# summarize_data(series) -- takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode"

@task
def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary


@flow
def pipeline_flow(array):
    series = create_series(array)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary


# Run the flow 
if __name__ == "__main__":
    pipeline_flow(arr)
