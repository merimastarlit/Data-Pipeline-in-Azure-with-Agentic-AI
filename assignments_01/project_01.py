from prefect.logging import get_run_logger
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from prefect import task, flow
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats
import scipy.stats as stats

# Task 1: Load Multiple Years of Data

base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_{year}.csv"


@task(retries=3, retry_delay_seconds=2)
def merge_happiness_data():
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    dfs = []

    # Loop through each year, load the corresponding CSV, and append it to the list of DataFrames
    for year in years:
        url = base_url.format(year=year)
        df = pd.read_csv(url, sep=";", decimal=",")
        df["year"] = year
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame and save it to a CSV file
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv("./outputs/merged_happiness.csv", index=False)
    return merged_df

# Run the flow
if __name__ == "__main__":
    output_df = merge_happiness_data()
    print(output_df.head())


# Task 2: Descriptive Statistics


@task
def test_stats(output_df):
    happiness_score = output_df["Happiness score"]

    """
    Compute and log  mean, median, and std.
    Compute and log the mean happiness score grouped by year and by region
    """
    # Initialize the logger
    logger = get_run_logger()

    mean = happiness_score.mean()
    median = happiness_score.median()
    std_dev = happiness_score.std()

    logger.info(f"Mean: {mean}")
    logger.info(f"Median: {median}")
    logger.info(f"Standard Deviation: {std_dev}")

    # Group by year and region, then compute mean happiness score for each group
    grouped_year = output_df.groupby("year")["Happiness score"].mean()
    grouped_region = output_df.groupby("Regional indicator")["Happiness score"].mean()

    logger.info(f"Mean Happiness Score by Year: {grouped_year}")
    logger.info(f"Mean Happiness Score by Region: {grouped_region}")


# Task 3: Visual Exploration

# a) histogram of happiness scores

@task
def histogram_happiness_scores(output_df):
    logger = get_run_logger()
    """
    - A histogram of all happiness scores across all years.
    - Save the histogram.
    """
    
    plt.hist(output_df["Happiness score"], bins=20, edgecolor="black")
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig("./outputs/happiness_histogram.png")
    logger.info('Histogram saved to "./outputs/happiness_histogram.png"')
    plt.close()


# b) boxplot of happiness scores

@task
def boxplot_happiness_scores(output_df):
    logger = get_run_logger()
    """
    - A boxplot comparing happiness score distributions across years (one box per year)..
    - Save the boxplot.
    """
    years = sorted(output_df["year"].unique())
    plt.boxplot([output_df[output_df["year"] == year]["Happiness score"] for year in years],
                labels=years)
    plt.title("Happiness Scores by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.savefig("./outputs/happiness_by_year.png")
    logger.info('Boxplot saved to "./outputs/happiness_by_year.png"')
    plt.close()


# c) scatter plot of happiness score vs. GDP per capita

@task
def scatter_plot_happiness_gdp(output_df):
    logger = get_run_logger()
    """
    - A scatter plot showing the relationship between GDP per capita and happiness score.
    - Save the scatter plot.
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(output_df["GDP per capita"], output_df["Happiness score"],
                          c=output_df["year"], cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Year")
    plt.title("Happiness Score vs. GDP per Capita")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.savefig("./outputs/gdp_vs_happiness.png")
    logger.info('Scatter plot saved to "./outputs/gdp_vs_happiness.png"')
    plt.close()


# Task 4: a Correlation heatmap

@task
def describe_correlations(output_df):
    logger = get_run_logger()
    """
    - A correlation heatmap (using sns.heatmap() with annot=True) showing the Pearson correlations between all numeric columns.
    - Save the heatmap.
    """
    # Select only numeric columns for correlation analysis
    numeric_df = output_df.select_dtypes(include="number")
    # Compute the correlation matrix
    corr = numeric_df.corr()

    # Create and save the heatmap
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("./outputs/correlation_heatmap.png")
    logger.info(
        'Correlation heatmap saved to "./outputs/correlation_heatmap.png"')
    plt.close()


# Task 4: Hypothesis Testing

@task
def hypothesis_testing(output_df):
    logger = get_run_logger()
    """
    - A t-test comparing the mean happiness scores between two different years (e.g., 2019 vs. 2020).
    - A t-test comparing the mean happiness scores between two different regions (e.g., Western Europe vs. East Asia).
    - Log the results of the t-tests, including the t-statistic and p-value, and whether the results are statistically significant.
    """
    # select 2019 and 2020 for year comparison
    group_a = output_df[output_df["year"] == 2019]["Happiness score"]
    group_b = output_df[output_df["year"] == 2020]["Happiness score"]

    # Compute means to determine direction of change
    group_a_mean = group_a.mean()
    group_b_mean = group_b.mean()

    # Determine if happiness increased or decreased from 2019 to 2020
    if group_b_mean < group_a_mean:
        direction_2019_2020 = "decreased"
    else:
        direction_2019_2020 = "increased"

    # Perform t-test with groups that may have unequal variances
    t_stat_2019_2020, p_val_2019_2020 = stats.ttest_ind(group_a, group_b, equal_var=False)

    # Log the results of the t-test for years
    logger.info(f"2019 mean: {group_a_mean}")
    logger.info(f"2020 mean: {group_b_mean}")
    logger.info(f"t-statistic: {t_stat_2019_2020}")
    logger.info(f"p-value: {p_val_2019_2020}")

    # select Western Europe and East Asia for region comparison and perform t-test
    region_a = output_df[output_df["Regional indicator"] == "Western Europe"]["Happiness score"]
    region_b = output_df[output_df["Regional indicator"] == "East Asia"]["Happiness score"]

    # Compute means to determine direction of difference between regions
    region_a_mean = region_a.mean()
    region_b_mean = region_b.mean()

    if region_a_mean > region_b_mean:
        region_direction = "higher"
    else:
        region_direction = "lower"

    # Perform t-test with groups that may have unequal variances
    t_stat_regions, p_val_regions = stats.ttest_ind(region_a, region_b, equal_var=False)

    # Log the results of the t-test for regions 
    logger.info(f"Western Europe mean: {region_a_mean}")
    logger.info(f"East Asia mean: {region_b_mean}")
    logger.info(f"t-statistic: {t_stat_regions}")
    logger.info(f"p-value: {p_val_regions}")

    # Return the results as a dictionary for use in the summary report
    return {
        "mean_2019": group_a_mean,
        "mean_2020": group_b_mean,
        "p_value_2019_2020": p_val_2019_2020,
        "direction_2019_2020": direction_2019_2020,
        "western_europe_mean": region_a_mean,
        "east_asia_mean": region_b_mean,
        "p_value_regions": p_val_regions,
        "region_direction": region_direction,
    }



# Task 5: Correlation and Multiple Comparisons

@task
def correlation_tests(output_df) -> dict:
    logger = get_run_logger()
    """
    Compute Pearson correlations between each numeric explanatory variable
    and happiness score, then apply Bonferroni correction.
    """
    # Select only numeric columns for correlation analysis
    numeric_df = output_df.select_dtypes(include="number")

    # Define the target variable for correlation
    target = "Happiness score"

    # Identify the columns to test (all numeric columns except the target)
    test_columns = [col for col in numeric_df.columns if col != target]

    # Calculate the number of tests and the adjusted alpha for Bonferroni correction
    number_of_tests = len(test_columns)
    adjusted_alpha = 0.05 / number_of_tests

    logger.info(f"Number of correlation tests: {number_of_tests}")
    logger.info(f"Adjusted alpha (Bonferroni): {adjusted_alpha}")

    # Initialize variables to track the strongest correlation that remains significant after correction
    strongest_variable = None
    strongest_correlation = None
    strongest_p_value = None

    # Loop through each test column, compute the Pearson correlation and p-value, and log the results
    for col in test_columns:
        corr_coef, p_val = stats.pearsonr(numeric_df[col], numeric_df[target])

        # Determine significance before and after Bonferroni correction
        significant_original = p_val < 0.05
        significant_bonferroni = p_val < adjusted_alpha

        logger.info(f"{col} vs {target}")
        logger.info(f"  Correlation coefficient: {corr_coef}")
        logger.info(f"  p-value: {p_val}")
        logger.info(f"  Significant at alpha=0.05: {significant_original}")
        logger.info(
            f"  Significant after Bonferroni correction (alpha={adjusted_alpha}): {significant_bonferroni}"
        )

        # Check if this variable has the strongest correlation that remains significant after Bonferroni correction
        if significant_bonferroni:
            # If this is the first significant variable or has a stronger correlation than the current strongest, update the tracking variables
            if strongest_correlation is None or abs(corr_coef) > abs(strongest_correlation):
                strongest_variable = col
                strongest_correlation = corr_coef
                strongest_p_value = p_val

    
    return {
        "strongest_variable": strongest_variable,
        "strongest_correlation": strongest_correlation,
        "strongest_p_value": strongest_p_value,
        "adjusted_alpha": adjusted_alpha,
    }


@task
def summary_report(output_df, hypothesis_results, correlation_results):
    logger = get_run_logger()

    logger.info(
        f"The merged dataset includes {output_df['Country'].nunique()} countries across {output_df['year'].nunique()} years."
    )

    region_means = output_df.groupby("Regional indicator")[
        "Happiness score"].mean()
    logger.info(
        f"Top 3 regions by mean happiness score:\n{region_means.nlargest(3)}")
    logger.info(
        f"Bottom 3 regions by mean happiness score:\n{region_means.nsmallest(3)}")

    if hypothesis_results["p_value_2019_2020"] < 0.05:
        logger.info(
            f"Average happiness {hypothesis_results['direction_2019_2020']} from 2019 "
            f"({hypothesis_results['mean_2019']:.3f}) to 2020 ({hypothesis_results['mean_2020']:.3f}), "
            f"and the change is statistically significant."
        )
    else:
        logger.info(
            f"Average happiness {hypothesis_results['direction_2019_2020']} from 2019 "
            f"({hypothesis_results['mean_2019']:.3f}) to 2020 ({hypothesis_results['mean_2020']:.3f}), "
            f"but the change is not statistically significant."
        )

    if correlation_results["strongest_variable"] is not None:
        logger.info(
            f"After Bonferroni correction, the variable most strongly correlated with happiness score is "
            f"{correlation_results['strongest_variable']} "
            f"(r={correlation_results['strongest_correlation']:.3f}, "
            f"p={correlation_results['strongest_p_value']:.6f})."
        )
    else:
        logger.info(
            "After Bonferroni correction, no numeric variable remained significantly correlated with happiness score."
        )

# Main flow to run all tasks in sequence and generate the summary report
@flow
def happiness_pipeline():
    output_df = merge_happiness_data()
    test_stats(output_df)
    histogram_happiness_scores(output_df)
    boxplot_happiness_scores(output_df)
    scatter_plot_happiness_gdp(output_df)
    describe_correlations(output_df)

    hypothesis_results = hypothesis_testing(output_df)
    correlation_results = correlation_tests(output_df)

    summary_report(output_df, hypothesis_results, correlation_results)


if __name__ == "__main__":
    happiness_pipeline()
