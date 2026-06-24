from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import os
import platform
from pathlib import Path
from dotenv import load_dotenv

# if platform.machine() != "arm64":
#     raise RuntimeError(
#         "This virtual environment has arm64 packages installed, but Python is "
#         f"running as {platform.machine()}. Run with: "
#         "arch -arm64 ./venv/bin/python -u assignments_06/project_06.py"
#     )

base_dir = Path(__file__).resolve().parent
repo_dir = base_dir.parent


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(f"OPENAI_API_KEY not found. Expected it in: {env_path}")


# Step 1: Setup
docs_dir = base_dir / "resources" / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents
docs = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Documents loaded: {len(docs)}")

for doc in docs:
    print(f"File name: {doc.metadata['file_name']}")

# # Build a vector index automatically (handles chunking + embeddings)
# index = VectorStoreIndex.from_documents(docs)

# print(type(index._vector_store).__name__)


# Step 3: Build the Index and Query Engine

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

print(type(index._vector_store).__name__)


query_engine = index.as_query_engine(similarity_top_k=3)

print(f"Index built successfully. Ready to answer questions.")

# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(
            f"Document name: {node_with_score.node.metadata.get('file_name', 'Unknown')}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)


# After running all five queries, add a comment reflecting on the responses: did the assistant sound confident and accurate? Did any of the answers surprise you?

# The assistant provided responses that were generally accurate and relevant to the questions asked. The confidence in the answers seemed appropriate, as they were based on the information available in the documents. However, some answers may have been more detailed than others, depending on the content of the source documents. Overall, the assistant performed well in retrieving and presenting information from the provided documents.


# Step 5: Find a Failure

question = "How much does the manager get paid?"

print(f"\nQ: {question}")
response2 = query_engine.query(question)
answer_text = getattr(response2, "response", str(response2))
print("A:", answer_text)

if not response2.source_nodes:
    print("No source nodes returned; the model had nothing to retrieve.")
else:
    for node_with_score in response2.source_nodes:
        print(
            f"Document name: {node_with_score.node.metadata.get('file_name', 'Unknown')}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)


# Output:

# Q: How much the manager gets paid?
# A: I'm unable to provide an answer to that question as there is no information provided in the context about how much the manager gets paid at Groundwork Coffee Co.
# Document name: seasonal_specials.txt
# Similarity Score: 0.7369
# Text Snippet: Seasonal Specials — Current Menu

# These drinks are available for a limited time only.

# Iced Lavender Lemonade — $5.00

# What you asked and why you expected it to be hard?
# I asked about the manager's salary, which is typically considered private information and is unlikely to be included in public documents or menus. I expected it to be hard because such information is not commonly disclosed and may not be relevant to the general operations of the coffee shop.

# What went wrong — wrong retrieval, missing information, the model guessed anyway?
# The model correctly identified that there was no information available about the manager's salary in the provided documents. It did not attempt to guess or provide an answer based on unrelated information, which is a positive aspect of its performance.

# When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?

# The model's tone did not change significantly when it failed to retrieve relevant information. It provided a clear and direct response indicating that it could not answer the question due to the lack of information. This suggests that while the model can recognize when it does not have enough information to provide an answer, it may still sound confident in its response. This highlights the importance of critically evaluating AI-generated responses and being cautious about trusting them, especially when they pertain to sensitive or private information.


# What you would change about the system to improve it

# To improve the system, I would implement a more robust mechanism for handling questions that fall outside the scope of the provided documents. This could include a feature that allows the model to explicitly state when it does not have enough information to answer a question, rather than simply providing a generic response. But for now, it works greatly even with generic responses, ti is better than nothing or failure.


# Step 6: Reflection

# 1. The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?

# The equivalent LlamaIndex implementation took significantly fewer lines of code compared to the manual implementation of semantic RAG. This highlights the value of using a framework like LlamaIndex, as it abstracts away much of the complexity involved in chunking, embedding, and indexing. It allows developers to focus on higher-level tasks and reduces the likelihood of errors, ultimately speeding up development time and improving efficiency.

# 2. You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.

# A different use case for this approach could be in the legal industry. Law firms often have vast amounts of legal documents, case files, and research materials. Implementing a RAG system using LlamaIndex could allow lawyers and legal researchers to quickly retrieve relevant information from these documents when preparing for cases or conducting legal research. This would save time and improve the efficiency of legal work, allowing professionals to focus on analysis and strategy rather than sifting through large volumes of text.

# 3.What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

# One failure mode that RAG cannot fully prevent is the presence of outdated or incorrect information in the source documents. Even if the retrieval process is working correctly and the most relevant documents are being retrieved, if those documents contain inaccurate or outdated information, the model may provide responses based on that flawed data. This highlights the importance of ensuring that the source documents are regularly updated and verified for accuracy to maintain the reliability of the RAG system.
