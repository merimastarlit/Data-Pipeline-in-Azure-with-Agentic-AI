
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool


base_dir = Path(__file__).resolve().parent
repo_dir = base_dir.parent


if load_dotenv(dotenv_path=repo_dir / "ao.env"):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "assignments_07/resources/merged_happiness.csv"

# Task 1: Implement the Tools

df = None


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Returns:
        A dict with the dataset shape and column names.
    """
    global df
    df = pd.read_csv(DATA_PATH)

    return {
        "shape": df.shape,
        "columns": df.columns.to_list()
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The column name to summarize.

    Returns:
        A dict of descriptive statistics, or an error dict.
    """
    if df is None:
        return {"error": "No data loaded. Run load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in the dataset."}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The first numeric column.
        col2: The second numeric column.

    Returns:
        A dict with col1, col2, pearson_r, and p_value, or an error dict.
    """
    if df is None:
        return {"error": "No data loaded. Run load_happiness_data first."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found in the dataset."}

    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": "Both columns must be numeric."}

    clean_data = df[[col1, col2]].dropna()

    if len(clean_data) < 2:
        return {"error": "Not enough valid data to compute correlation."}

    corr, p_value = pearsonr(clean_data[col1], clean_data[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(corr, 4),
        "p_value": round(p_value, 4)
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The column to rank by.
        year: The year to filter by.
        n: The number of countries to return.

    Returns:
        A dict with year, column, and top_countries, or an error dict.
    """
    if df is None:
        return {"error": "No data loaded. Run load_happiness_data first."}

    if "year" not in df.columns:
        return {"error": "Year column not found."}

    if "country" not in df.columns:
        return {"error": "Country column not found."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    year_data = df[df["year"] == year]

    if year_data.empty:
        return {"error": f"No data found for year {year}."}

    top_countries = year_data.sort_values(by=column, ascending=False).head(n)

    result = (
        top_countries[["country", column]]
        .to_dict(orient="records")
    )

    return {
        "year": year,
        "column": column,
        "top_countries": result
    }

# Task 2: Build the Agent


model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).

Important:
- load_happiness_data returns only metadata ("shape" and "columns"), not the full dataset.
- Do not convert the output of load_happiness_data into a DataFrame.
- For custom analysis or plots, read the CSV directly using pandas from:
  assignments_07/resources/merged_happiness.csv

"""
agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib.pyplot",
        "scipy.stats"
    ],
    max_steps=8,
)

# Task 3: Run Guided Queries

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    (
        "Plot happiness_score over the years as a line chart, with one line per region. "
        "Do not use load_happiness_data for the plot. "
        "Read the CSV directly with pandas from "
        "assignments_07/resources/merged_happiness.csv. "
        "Save the plot to assignments_07/outputs/happiness_by_region.png."
    ),
]

if __name__ == "__main__":
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

# Task 4: Your Own Questions

    # My query 1
    my_query_1 = (
        "Which 3 countries had the largest increase in happiness_score "
        "from 2015 to 2020?"
    )

    response_1 = agent.run(my_query_1, reset=False)
    print(f"\n--- Query: {my_query_1} ---")
    print(response_1)

    # Comment: This should trigger code generation because no tool directly
    # computes changes in happiness_score across years.

    # My query 2
    my_query_2 = (

        "Create a scatter plot of gdp_per_capita vs happiness_score. "
        "Do not use load_happiness_data for the plot. "
        "Instead, read the CSV directly with pandas from "
        "assignments_07/resources/merged_happiness.csv. "
        "Save it to assignments_07/outputs/gdp_vs_happiness.png."
    )

    response_2 = agent.run(my_query_2, reset=False)
    print(f"\n--- Query: {my_query_2} ---")
    print(response_2)

    # Comment: This should trigger code generation because the agent must
    # write matplotlib plotting code.

# Task 5: Reflection

# --- Reflection ---
#
# 1. In Query 3, the agent communicated statistical significance by reporting
#    the p-value along with the Pearson correlation coefficient. It correctly
#    treated the correlation as statistically significant when the p-value was
#    less than 0.05, which is a common threshold in statistics.
#
# 2. One surprising response was when the agent generated matplotlib code on
#    its own to create the regional happiness line chart. I expected it to
#    rely mostly on tools, but it was able to write custom plotting code and
#    save the figure correctly without a dedicated plotting tool.
#
# 3. One additional useful tool would be a "compare_countries" tool that lets
#    users compare multiple countries across several years. It could calculate
#    trends, averages, and ranking changes over time. This would help answer
#    questions like: "How did the happiness scores of the United States and
#    Canada change between 2015 and 2020?"
