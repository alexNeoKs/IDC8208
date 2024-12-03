import os
import json
import matplotlib.pyplot as plt
import numpy as np
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Ensure OpenAI API key is set
os.environ["OPENAI_API_KEY"] = "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"

# Initialize the LangChain pipeline
def langchain_rag_pipeline():
    # Initialize retriever
    retriever = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True).as_retriever()
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3-turbo", temperature=0)
    
    # Create QA pipeline
    qa_pipeline = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_pipeline

# Load the JSON dataset
with open("ai_rag_ground_truth.json", "r") as file:
    dataset = json.load(file)

# Set up LangChain pipeline
langchain_pipeline = langchain_rag_pipeline()

# Evaluation loop for LangChain model
results = []

for entry in dataset:
    query = entry["query"]
    ground_truth_answer = entry["answer"]
    ground_truth_context = entry.get("context", "")

    # Run the query through LangChain RAG pipeline
    try:
        langchain_answer = langchain_pipeline.run(query)
        # Store the result
        results.append({
            "query": query,
            "langchain_answer": langchain_answer,
            "ground_truth_answer": ground_truth_answer,
        })
    except Exception as e:
        print(f"Error processing query '{query}' with LangChain: {e}")
        results.append({
            "query": query,
            "langchain_answer": "Error",
            "ground_truth_answer": ground_truth_answer,
        })

# Placeholder dummy scores for testing visualization (replace with real scores)
metrics = ["helpfulness", "correctness", "coherence", "relevance"]
text_based_results = results[:10]  # First 10 are text-based
image_based_results = results[10:]  # Next 10 are image-based

# Dummy scores for visualization
text_scores = {metric: np.random.uniform(3, 5, 10) for metric in metrics}
image_scores = {metric: np.random.uniform(2, 4, 10) for metric in metrics}

# Visualization
categories = list(text_scores.keys())
text_values = [np.mean(text_scores[metric]) for metric in metrics]
image_values = [np.mean(image_scores[metric]) for metric in metrics]

x = np.arange(len(categories))  # Metric indices
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, text_values, width, label="Text Queries")
ax.bar(x + width / 2, image_values, width, label="Image Queries")

# Formatting
ax.set_title("LangChain RAG Model: Text vs. Image Query Performance")
ax.set_ylabel("Average Score (0-5)")
ax.set_xlabel("Metrics")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()
