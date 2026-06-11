# Lesson 10
## ---LLMs as Transform---

# LLMs as Transform Question 1

# Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05"
# A: I would use deterministic code that uses from built in date libraries do this job easily. 

# Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
# A: I would use LLM for the above as the fine-tuned classifier that does the exact job.

# Calculate the average of a list of numbers.
# A: I would use deterministic code that would calculate the average. We don't need LLM here.

# Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)".
# A: Simple regex or an NLP entity extractor will be more reliable and debuggable than an LLM for this task.
# Determine whether a product review is more than 100 words long.
# I would use a deterministic code here to determine if it is above 100 words as LLM would do this process poorly.

# LLMs as Transform Question 2

system = "Summarize this product review in a few sentences"
# this original prompt needs more details and specificity to ensure consistent output formats that won't break automated pipelines

system = """You are a product-review summarizer. Input: raw review text. Output: JSON only, with exactly two keys: "summary": a concise 2–3 sentence summary, "word_count": integer word count of that summary. If the review is empty, output: {"error": "no content"} Do not output any additional keys, markdown, or commentary."""
# This system definition is incomplete. In order to confirm the system went through successfully, we can add more details as shown on the second system.

# LLMs as Transform Question 3

# Your dataset has 50,000 records and you need to run a classification call for each one using gpt-4o-mini. In a comment block, answer:
# If each call takes 1 second on average, how long would sequential processing take?
# What is one practical strategy to handle this more efficiently at scale, without changing models?

# Sequential processing time:
# 50,000 records × 1 second per call = 50,000 seconds total
# 50,000 seconds ÷ 3,600 seconds/hour ≈ 13.9 hours
# 
# Practical strategy for scale (without changing models):
# Introduce concurrency or parallelism. For example, spin up multiple workers (threads, async tasks, or processes) to issue API calls in parallel—up to your rate‐limit ceiling—so that if you run 10 concurrent requests you cut overall time roughly by a factor of 10 (down to ~1.4 hours).

# ---Azure OpenAI---

# Azure OpenAI Question 1
# In a comment block, name two reasons an organization might use Azure OpenAI instead of calling the OpenAI API directly. Be specific -- "it's better" is not an answer.

# A: Companies use Azure OpenAI because of main three reasons. I will write two of them. 1) Data Residency and compliance. When we use Azure OpenAI, all data will be kept in company's infrastructure, versus when we use just OpenAI, our data leaves and travels to OpenAI's servers. 2) When we use Azure OpenAI, our data will not be used for training. It explicitly written in the enterprise contract.

# Azure OpenAI Question 2
# When you switch from OpenAI to AzureOpenAI, the client initialization takes three Azure-specific parameters. In a comment block, name them and describe what each one is. (Do not include the standard api_key -- describe the Azure-specific ones.)

# A: azure_endpoint="https://<resource-name>.openai.azure.com", (It needs our resource name on Azure cloud to connect.)
    
    # api_version="2024-02-01" (The string that selects which Azure OpenAI API version to call (e.g. “2024-02-01”))
    # api_type (Set this to “azure” so the client knows you’re targeting Azure’s hosted models.)
    # api_base (the full URL of your Azure OpenAI resource)

# Azure OpenAI Question 3

# In a comment block, answer: when using AzureOpenAI, the model parameter in chat.completions.create() does not take a value like "gpt-4o-mini". What does it take instead, and where do you find the right value to use?

# A: Model takes a deployment name, not a model name. In Azure OpenAI, we do not call a model directly, we call a named deployment that our organization's admin created and configured. The deployment name might be something like "gpt4o-mini-prod" or "my-gpt4o-deployment" rather than "gpt-4o-mini". In the Azure Portal (OpenAI resource → Deployments blade) or in the Azure OpenAI Studio UI you’ll see each deployment’s exact identifier (e.g. “gpt4o-mini-prod”).