from prefect import task, get_run_logger

# ---Prefect Orchestration---

# Prefect Question 1

# What is the difference between a @task and a @flow in Prefect? You have a helper function that converts a temperature from Celsius to Fahrenheit -- a pure, in-memory calculation with no I/O. Would you decorate it with @task? Why or why not?
# A: a @task is used for one step in a pipeline, like extract, transform, or load.
# Prefect can track it, log it, retry it, and show if it passed or failed.

# A @flow is used to organize and run tasks together in the order the pipeline needs.

# I would not decorate a small Celsius-to-Fahrenheit helper with @task because
# it is just a quick pure calculation in memory. It has no I/O and does not
# need Prefect tracking, retries, or separate logging.

# Prefect Question 2

# Write the decorator (just the decorator line, not the full function) for a task named call_api that retries up to 3 times with a 30-second delay between attempts.

# A: @task(retries=3, retry_delay_seconds=30)

# Prefect Question 3

# You run your pipeline and the Prefect UI shows: extract is Completed, transform is Failed, load never ran. In a comment block, describe: where in the UI do you look to understand what went wrong, and what specific information would you expect to find there?

# A: I would look at the failed transform task run in the Prefect UI. I would expect to find the error message, the task logs, and possibly a stack trace showing where the failure happened. I would also check the timing and state details for the failed task run.

# ---Production Patterns---

# Production Question 1

# In a comment block, explain what raise_for_status() does and why it is better than writing if response.status_code != 200: print("error") in a pipeline task. What happens to downstream tasks in each case when the API returns a 500 error?

# A: raise_for_status() checks the HTTP response. If the response has an error status code like 400 or 500, it raises a requests.exceptions.HTTPError.
# This is better than only writing:
#     if response.status_code != 200:
#         print("error")
# because print("error") does not stop the task. Prefect may think the task completed successfully, so downstream tasks might still run with bad or missing data.
# If the API returns a 500 error and we use raise_for_status(), the task fails. Then Prefect marks that task as Failed and downstream tasks do not run.
# If we only print("error"), the task may still look successful, and downstream tasks may run even though the API call failed.

# Production Question 2

# Your pipeline uploads results to final/{today}/weather_etl.json with overwrite=True. The pipeline crashes halfway through the transform step. You fix the bug and re-run it from the beginning. In a comment block, explain: what does overwrite=True protect you from in this scenario, and what would happen without it?

# A: overwrite=True allows the pipeline to replace an existing file at the same blob path.
# In this scenario, if the pipeline crashed halfway through transform, I can fix the bug and rerun from the beginning. When the load step writes to:

#     final/{today}/weather_etl.json

# overwrite=True protects me from errors caused by a file already existing at that path from a previous run.
# Without overwrite=True, the upload might fail because the file already exists, or the pipeline might avoid replacing the old result, leaving stale data.


# Production Question 3

# Write a task stub -- just the function signature, decorator, and a single log line -- that uses get_run_logger() to log an INFO message saying how many records were loaded. The function should accept records (a list) and blob_path (a string) as arguments.

# A:
@task
def load_records(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")
