# --- Completions API ---
# API Q1

import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(f"Response text: {response.choices[0].message.content}")
print(f"Model: {response.model}")
print(f"Total tokens: {response.usage.total_tokens}")


# API Q2:

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print("Temperature:", temp)
    print("Response:", response.choices[0].message.content)

# As temperature increases, the responses become more creative and less deterministic. At temperature 0, the model gives a very straightforward and common name. At 0.7, the name is more unique and imaginative. At 1.5, the name is quite unconventional and may be less relevant to a data engineering consultancy, demonstrating increased creativity but decreased coherence. For personal choice, middle ground (0.7) often provides a good balance between creativity and relevance. But in some cases, a higher temperature might be preferred for brainstorming sessions where out-of-the-box ideas are desired. Especially for creative branding names, a higher temperature can yield more unique and memorable suggestions. If middle temperature doesn't satisfy me, I would analyze more creative version. Using below temperature has its own use cases. So it depends on the context and the desired outcome.
# For consistent and reproducable output, I would use temperature=0, as it doesn't need to be creative.


# API Q3:


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices):
    print(f"Fun Fact {i+1}: {choice.message.content}")


# API Q4:

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is neural network in machine learning?"}],
    max_tokens=15,
    temperature=0.5
)

print(f"Response text: {response.choices[0].message.content}")

# The response was cut off because max_tokens limits how many tokens the model can generate. if we put lower number of max_tokens, the model will generate shorter responses. This is useful to control cost and keep responses short and readable. If we want a more detailed answer, we can increase the max_tokens limit.


# --- System Messages and Personas ---

# System  Q1:

# system role 1:
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# system role 2:
messages = [
    {"role": "system", "content": "You are a funny pirate who explains Python with silly examples."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# The tone and style changed based on the system message (tutor vs pirate personality), even though the question stayed the same.

# System Q2:

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# Why does the model know Jordan's name, even though it's stateless?

# The model knows Jordan's name because we included it in the message history. Even though the API is stateless, we send the full conversation each time, so the model can read earlier messages and use that information.

# --- Prompt Engineering ---

# Prompt - Zero Shot Q1:

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = """
Classify the sentiment of each review as positive, negative, or mixed.

Review 1: The onboarding process was smooth and the team was welcoming.
Review 2: The software crashes constantly and support never responds.
Review 3: Great price, but the documentation is nearly impossible to follow.
Respond in this format:
Review 1: positive
Review 2: negative
Review 3: mixed

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

# Prompt - One Shot Q2:
prompt = """
Classify the sentiment of each review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review 1: The onboarding process was smooth and the team was welcoming.
Review 2: The software crashes constantly and support never responds.
Review 3: Great price, but the documentation is nearly impossible to follow.

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

# Adding one example made the output more consistent, because the model followed the pattern shown (Review → Sentiment).

# Prompt - Few Shot Q3:

prompt = """
Classify the sentiment of each review as positive, negative, or mixed.

Example 1:
Review: "I love the product because it is easy to use."
Sentiment: positive

Example 2:
Review: "I hate the product because it is not convenient to use."
Sentiment: negative

Example 3:
Review: "The product is good, but I don’t see myself using it."
Sentiment: mixed

Review 1: The onboarding process was smooth and the team was welcoming.
Review 2: The software crashes constantly and support never responds.
Review 3: Great price, but the documentation is nearly impossible to follow.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

# Zero-shot works for simple tasks but may be less consistent.
# One-shot improves format and gives the model a pattern to follow.
# Few-shot gives the best accuracy and consistency because multiple examples reinforce the pattern. It works best because it gives the model multiple patterns to learn from, so it can handle complex tasks more accurately and consistently.

# Prompt - Chain-of-Thought Q4:

prompt = """
Solve the following problem. Show your reasoning step by step.
Then give the final answer clearly labeled as 'Final Answer:'.

A data engineer earns $85,000 per year. She gets a 12 percent raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

# Why does asking the model to reason step by step tend to improve accuracy on problems like this?

# Asking the model to reason step by step helps it break down the problem into smaller parts, which can reduce errors and improve accuracy. It allows the model to follow a logical process, making it easier to identify mistakes and arrive at the correct final answer.


# Prompt - Strctured Output Q5:


review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the following review and return the result as valid JSON.
Include the keys: sentiment, confidence, and reason.
- sentiment should be positive, negative, or mixed
- confidence should be a number between 0 and 1
- reason should be one sentence
Review: "{review}"

Return only valid JSON. Do not include any extra text.

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

response_text = response.choices[0].message.content

try:
    data = json.loads(response_text)
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])
except json.JSONDecodeError:
    print("Invalid JSON. Raw response:")
    print(response_text)

# If the response is not valid JSON, json.loads() will throw an error. We use try/except to handle the error gracefully and debug the raw response.


# Prompt - Delimeters Q6:

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

# Now with the second version:

user_text2 = "I went to the park yesterday and enjoyed the sunshine."

prompt2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text2}```
"""

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt2}]
)

print(response2.choices[0].message.content)

# The delimeter helps prevent model confusing the instructions.



# --- Ollama ---

# Runnning with OpenAI's GPT-4o-mini model:
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": "Explain what a large language model is in two sentences."}]
)

print(response.choices[0].message.content)

# The output from running Ollama locally:

"""
A large language model is a powerful artificial intelligence system 
trained on massive amounts of text data to understand and generate 
human-like conversations, knowledge, and tasks. These models excel in 
processing complex information, understanding context, and generating 
coherent text, making them essential in fields like healthcare, finance, 
and education.
"""

# The Ollama response was simpler and less detailed compared to the OpenAI response, which was more polished and informative.
# One advantage of running a model locally is that it can be free to use after setup and does not require internet access.
# One disadvantage is that smaller local models are less powerful and may produce less accurate or less detailed responses compared to larger cloud-based models.
