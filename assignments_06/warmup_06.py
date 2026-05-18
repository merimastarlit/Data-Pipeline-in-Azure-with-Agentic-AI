# --- RAG Concepts ---

# Concepts Q1:

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import string
from dotenv import load_dotenv
import os
import platform
from pathlib import Path

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


# Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
# Best approach: RAG
# Explanation: RAG is ideal for scenarios where the model needs to access and retrieve information from a large collection of documents, such as a policy library.

# Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.
# Best approach: Fine-tuning
# Explanation: Fine-tuning is suitable when you have a specific task and a dataset of examples to train the model on, like generating content in a particular style.

# Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.
# Best approach: Prompt engineering
# Explanation: Prompt engineering is effective for tasks that require specific instructions or modifications to the input prompt, especially when dealing with unique or specialized content.


# Concepts Q2:

# AI hallucinations (responses that sound confident but are wrong) can be particularly difficult to detect. Add a comment to your code answering this:

# Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.

# A confidently wrong answer can be more harmful than one that admits uncertainty because it can lead to misguided decisions based on false information. For example, if a medical AI confidently provides an incorrect diagnosis, a doctor might rely on that information and prescribe the wrong treatment, potentially endangering the patient's health. In contrast, if the AI had said "I am not sure," the doctor would likely seek additional opinions or tests before making a decision, reducing the risk of harm.


# Think about the tone of the response as well as its content — why does the way the model expresses an answer affect how much we trust it?

# The tone of the response can significantly influence our perception of its credibility. A confident and authoritative tone may lead us to trust the information more, even if it's incorrect, while a hesitant or uncertain tone might make us question the validity of the response. This is because we often associate confidence with expertise, so a model that expresses answers in a way that sounds knowledgeable can create a false sense of reliability, making it harder for users to critically evaluate the information provided.

# Concept Q3:


# The steps below make up a complete RAG pipeline, but they are out of order. Copy the list into your code as a comment, arrange them in the correct order, and add a one-sentence description of what happens at each step.

# steps = [
#     "Generate a response from the LLM",
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query",
# ]


# steps = [

# "Extract text from source documents",
# (At this step, the raw text content is extracted from the source documents, such as PDFs or web pages, to prepare it for further processing.)
# "Split text into chunks",
# (This step breaks down the extracted text into smaller, manageable pieces or chunks, which can be more easily processed and embedded.)
# "Convert text chunks into embeddings",
# (This step transforms the text chunks into vector representations (embeddings) that capture their semantic meaning, allowing for efficient similarity comparisons.After this it is saved in a vector database for later retrieval.)
# "Receive the user's query",
# (At this step, the system receives a query from the user, which is the question or request they want the model to address.)
# "Embed the user's query",
# (This step converts the user's query into an embedding, similar to how the text chunks were embedded, enabling the system to compare the query with the document chunks.)
# "Retrieve the most relevant chunks",
# (In this step, the system compares the query embedding with the embeddings of the text chunks to identify and retrieve the most relevant pieces of information that can help answer the user's query.)
# "Inject retrieved chunks into the prompt",
# (This step involves incorporating the retrieved relevant chunks of information into the prompt that will be sent to the LLM, providing it with the necessary context to generate an informed response.)
# "Generate a response from the LLM"
# (Finally, the LLM processes the prompt, which now includes the relevant information from the source documents, and generates a response to the user's query based on that context.)

# ]


# Keyword RAG Q1:


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content)
                for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)
for name, content in results:
    print(f"\n{name}:\n{content}")

# Output:

# Query tokens (filtered): ['hours', 'weekend', 'what', 'your']
# [menu.txt] overlap=0 -> []
# [hours.txt] overlap=0 -> []
# [hiring.txt] overlap=1 -> ['your']
# [loyalty.txt] overlap=1 -> ['your']

# Selected best match: loyalty.txt

# loyalty.txt:
# Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.

# Interppretation:
# The keyword retrieval function identified "loyalty.txt" as the best match for the query "What are your hours on the weekend?" because it contained the word "your," which is a common word in both the query and the document. However, this is a false positive since "loyalty.txt" does not actually contain relevant information about the hours of operation. This shows a key limitation of simple keyword retrieval methods: they can be misled by common words that do not contribute to the actual meaning or relevance of the content, leading to incorrect results.


# Q2:

query = "Do you have anything without caffeine?"

results2 = simple_keyword_retrieval(query, documents, verbose=True)
for name, content in results2:
    print(f"\n{name}:\n{content}")


# Output:
# Query tokens (filtered): ['anything', 'caffeine', 'do', 'have', 'without']
# [menu.txt] overlap=0 -> []
# [hours.txt] overlap=0 -> []
# [hiring.txt] overlap=0 -> []
# [loyalty.txt] overlap=0 -> []

# No overlapping keywords found.

# None found:
# No relevant content.

# Interpretation:
# It didn't find any relevant documents because the query contains specific keywords like "caffeine" and "without" that do not appear in any of the documents. This highlights another limitation of keyword retrieval: if the query uses different terminology or phrasing than the documents, it may fail to identify relevant content even if it exists.
# Keyword RAG didn't get anything right because it relies solely on the presence of overlapping keywords, and in this case, there were no shared keywords between the query and the documents.
# A semantic retrieval approach, such as using embeddings to capture the meaning of the query and documents, would likely perform better in this scenario. Semantic retrieval can understand that "without caffeine" is related to "decaf" or "non-caffeinated" options, even if those specific words are not present in the documents, allowing it to retrieve relevant information about decaf beverages from the menu.

# Keyword RAG Q3:

query = "How do I sign up for rewards?"
results3 = simple_keyword_retrieval(query, documents, verbose=True)
for name, content in results3:
    print(f"\n{name}:\n{content}")

# Output:
# Query tokens (filtered): ['do', 'how', 'i', 'rewards', 'sign', 'up']
# [menu.txt] overlap=0 -> []
# [hours.txt] overlap=0 -> []
# [hiring.txt] overlap=0 -> []
# [loyalty.txt] overlap=0 -> []

# No overlapping keywords found.

# None found:
# No relevant content.

# Interpretation:
# I first thought that it would give similar output liek the firt one, but it didn't because the keyword retrieval function failed to find relevant information about signing up for rewards because the query's keywords ("sign," "up," "rewards") do not directly match any of the keywords in the documents. Although "loyalty.txt" contains information about a loyalty program, it does not explicitly mention "sign up" or "rewards," leading to a failure in retrieval. This again demonstrates the limitations of keyword-based retrieval methods, which can miss relevant content if the query and documents use different terminology or phrasing.


# Semantic RAG Q1:

# 1. What is a vector embedding? (1-2 sentences)
# Vector embedding is a numerical repsesentation of data such as text, audio, or image in the form of lists of numbers(vectors). Each list of numbers live in a multidimentional space(vector database) and algorithms can measure how related items are by calculating the distance between them, enabling AI to understand context and similarity between different pieces of data.

# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
# The chunk with a cosine similarity score of 0.85 is more relevant to the given query. A cosine similarity score ranges from -1 to 1, where 1 indicates that the two vectors (text chunks) are identical in direction, 0 indicates that they are orthogonal (unrelated), and -1 indicates that they are diametrically opposed. Therefore, a score of 0.85 suggests a strong semantic relationship between the chunk and the query, while a score of 0.30 indicates a weaker relationship.

# 3. Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
# Because it relies on the underlying meaning and context of the text rather than just keyword matching. By using vector embeddings, semantic search captures the relationships between words and concepts, allowing it to identify relevant information based on similarity in meaning, even if different words are used. This enables it to retrieve relevant content that may be phrased differently but still conveys the same idea or information as the query.

# Semantic RAG Q2:

# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Vector embeddings |
# | What is retrieved?         | Full document                     | Relevant chunk |
# | Can it handle synonyms?    | No                                | Yes |
# | Storage format             | Plain text dictionary             | Vector database |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity score |


# LlamaIndex:


# Load documents directly from PDFs in the folder
docs_dir = base_dir / "resources" / "brightleaf_pdfs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
docs = SimpleDirectoryReader(str(docs_dir)).load_data()

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

print(type(index._vector_store).__name__)

# LlamaIndex Q1:

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Ouput:

# Q: What employee benefits does BrightLeaf offer?
# A: The employee benefits that BrightLeaf offers are not explicitly mentioned in the provided context information.
# Node ID: 3fd0edc6-db9a-453d-a885-bb6f97f9836b
# Similarity Score: 0.7995
# Text Snippet: ;FT"h:K.ai,f?S69JlS[McJNtaQhM.s?&rm$mDD+ilWgkGE0]8^rW\>(FlZVVWc3?sr6I)UHKiAY`KsHJ:kaW+-F#gg;,^9-Z@l2...
# ------------------------------
# Node ID: d4d11962-e985-4b79-9d65-6cda6a8670d4
# Similarity Score: 0.7896
# Text Snippet: %PDF-1.4
# % ReportLab Generated PDF document http://www.reportlab.com
# 1 0 obj
# <<
# /F1 2 0 R /F2 3 0 R
# ...
# ------------------------------
# Node ID: b3c50c99-1bf9-4a14-b240-c696c4747a58
# Similarity Score: 0.7881
# Text Snippet: "s1OPhL@r0Ac!_%QbH83/j:H2FE[pXA2<C)+F4r[9UB?gb:7V.3-'O;dN,l,jo1Y&kla7Cc\I9'd'KOZ:o<X<N*ZrB8,ceQQVdM3...
# ------------------------------

# Q: What are BrightLeaf's security policies?
# A: BrightLeaf's security policies are outlined in the PDF document located at the specified file path.
# Node ID: be501f3e-be0c-45da-954b-b78211d45a27
# Similarity Score: 0.8152
# Text Snippet: @Ob!+aR2Xf(o;M^;Y)VkY_c3MS(h;ON,d6Y!W)\V_%mTe^B2hh=d`d,O<Nco@rG@>3aY$?kuRFOl4EWB</s7e7Df61a]oBCi/SZ=...
# ------------------------------
# Node ID: 0a486b33-cc3c-4507-9094-fc3ecf625955
# Similarity Score: 0.8097
# Text Snippet: @,,bD<:t;XK\SnFVa-N.'+]]mG][knfN:Nf+n^5pD]VbeI1<@0t=:Ek/89j3FeVPn!";#N1`GY8!CAdR/2]D93hd&ohPt+qggPXk...
# ------------------------------
# Node ID: bb87dd23-ce81-4056-867d-6542630d1c8f
# Similarity Score: 0.8064
# Text Snippet: %PDF-1.4
# % ReportLab Generated PDF document http://www.reportlab.com
# 1 0 obj
# <<
# /F1 2 0 R /F2 3 0 R ...
# ------------------------------


# Interpretation:
# Do the retrieved chunks look relevant to the question?
# The retrieved chunks do not appear to be relevant to the questions asked. The responses indicate that the employee benefits and security policies are not explicitly mentioned in the provided context, and the text snippets retrieved do not seem to contain information related to those topics. This suggests that the retrieval process may have returned chunks that are not directly related to the queries, possibly due to limitations in the embedding or similarity scoring process.


# Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
# The model's response does not sound confident and specific; instead, it hedges with phrases like "are not explicitly mentioned in the provided context information." This indicates that the model is uncertain about the information it retrieved and is cautious in its response, acknowledging that it may not have found relevant content to answer the questions directly. The tone suggests a lack of confidence in the retrieved information, which may be due to the fact that the retrieved chunks did not contain clear answers to the queries.


# Did anything unexpected get retrieved?
# Yes, the retrieved chunks contained text snippets that do not seem to be relevant to the questions about employee benefits and security policies. The snippets appear to be random segments of text, possibly from the PDF documents, that do not provide specific information related to the queries. This unexpected retrieval indicates that the similarity scoring may have identified chunks that are not actually pertinent to the questions, which could be a result of the embedding process or the nature of the documents being indexed.


# LlamaIndex Q2:

query_engine = index.as_query_engine(similarity_top_k=1)

question = "Which partner joined most recently?"

print(f"\nQ: {question}")
response = query_engine.query(question)
print("A:", response)

# for node_with_score in response.source_nodes:
#     print(f"Node ID: {node_with_score.node.node_id}")
#     print(f"Similarity Score: {node_with_score.score:.4f}")
#     print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
#     print("-" * 30)


query_engine = index.as_query_engine(similarity_top_k=5)

question = "Which partner joined most recently?"


print(f"\nQ: {question}")
response = query_engine.query(question)
print("A:", response)

# for node_with_score in response.source_nodes:
#     print(f"Node ID: {node_with_score.node.node_id}")
#     print(f"Similarity Score: {node_with_score.score:.4f}")
#     print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
#     print("-" * 30)

# Ouput:
# Q: Which partner joined most recently?
# A: The partner that joined most recently is the one whose name is at the end of the alphabetically sorted list of partners.

# Q: Which partner joined most recently?
# A: The partner who joined most recently is the one mentioned in the "employee_benefits.pdf" file.


# When the similarity_top_k was set to 1, the response was more specific and directly addressed the question about which partner joined most recently. However, when the similarity_top_k was increased to 5, the response became less specific and more vague, as it likely included information from multiple chunks that may not have been directly relevant to the question. This illustrates that more retrieved context is not always better; it can sometimes introduce noise and reduce the clarity of the response if the additional chunks are not closely related to the query.


# LlamaIndex Q3:

query_engine = index.as_query_engine(similarity_top_k=3)

question = "Do employees drink coffee in the morning?"


print(f"\nQ: {question}")
response = query_engine.query(question)
print("A:", response)

# Ouput:

# Q: Do employees drink coffee in the morning?
# A: There is no information provided in the context about whether employees drink coffee in the morning.

# Because the question was vague and out of the topic or context, the model responded that way. The retrieved chunks likely did not contain relevant information about employees drinking coffee in the morning, leading the model to conclude that there was no information available on that topic.


# LlamaIndex Q4:


# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# Evaluate faithfulness and relevancy for a query


def evaluate_query(q):
    response = query_engine.query(q)
    answer_text = getattr(response, "response", str(response))

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=q, response=response)
    relevancy_result = relevancy_evaluator.evaluate_response(
        query=q, response=response)

    print(f"\nEvaluation query: {q}")
    print("A:", answer_text)
    print("Faithfulness Evaluation: " + str(faithfulness_result.score))
    print("Relevancy Result: " + str(relevancy_result.score))

    return faithfulness_result, relevancy_result


q = "What employee benefits does BrightLeaf offer?"
faithfulness_result, relevancy_result = evaluate_query(q)

low_quality_q = "What is BrightLeaf's CEO's favorite pizza topping?"
low_faithfulness_result, low_relevancy_result = evaluate_query(low_quality_q)

# Ouput:
# Faithfulness Evaluation: 0.0
# Relevancy Result: 1.0

# Q&A:

# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# A faithfulness score of 1.0 means the answer is fully supported by the retrieved context. A score of 0.0 means the answer is not supported by the context and may be hallucinated.

# What does a relevancy score measure, and how is it different from faithfulness?
# A relevancy score measures how well the answer addresses the user's question. It is different from faithfulness because an answer can be relevant to the question but still not be supported by the retrieved documents.

# Did the scores change between your two queries? If so, why do you think that happened?
# Yes, the scores can change between the employee benefits query and the pizza topping query because the first question is at least related to BrightLeaf documents, while the second question is clearly outside the available context. If the retrieved chunks do not support the answer, faithfulness may drop. If the answer does not directly address the user's question, relevancy may drop.

# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
# "LLM-as-a-judge" means using another LLM to evaluate the answer based on criteria like faithfulness and relevancy. It is used for RAG because many questions do not have one exact answer, so a simple accuracy metric is too rigid.
