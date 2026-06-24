# Pipeline Run Reflection

- Did the pipeline run cleanly on the first try? If not, what failed and how did you fix it?

A: The pipeline ran cleanly after my Azure login issue was resolved. The pipeline code itself did not fail, but I had trouble with `az login` at first because it did not give access right away. After trying again and using the subscription ID, I was able to authenticate successfully.

- What did the Prefect UI show? Were there any retries?

A: In the Prefect UI, the extract, transform, and load tasks all showed Completed, and I did not see any retries. I found the logs in the task run details, and they showed the printed progress messages from the pipeline.

- What is one thing you would change or add if you were deploying this pipeline to run on a daily schedule?

A: If I were deploying this pipeline on a daily schedule, I would add a Prefect schedule and alerts for authentication or cloud upload failures.