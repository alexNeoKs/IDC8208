from transformers import pipeline
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
with open("ai_rag_ground_truth.json", "r") as file:
    dataset = json.load(file)

# Initialize HuggingFace QA pipeline
hf_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

# Evaluation loop
results = []
for entry in dataset:
    query = entry["query"]
    ground_truth_answer = entry["answer"]
    ground_truth_context = entry.get("context", "")

    # Check if the query is image-based or text-based
    if "image_reference" in entry:
        # Skip images as HuggingFace QA pipelines process text-based queries only
        results.append({
            "query": query,
            "hf_answer": "Not applicable for image queries",
            "type": "image",
            "gpt_scores": {
                "helpfulness": 0,
                "correctness": 0,
                "coherence": 0,
                "relevance": 0
            }
        })
    else:
        # Use HuggingFace pipeline for text-based queries
        try:
            hf_result = hf_pipeline(question=query, context=ground_truth_context)
            hf_answer = hf_result["answer"]
            # Add dummy scores for now
            gpt_scores = {
                "helpfulness": np.random.uniform(3, 5),
                "correctness": np.random.uniform(3, 5),
                "coherence": np.random.uniform(3, 5),
                "relevance": np.random.uniform(3, 5)
            }
            results.append({
                "query": query,
                "hf_answer": hf_answer,
                "type": "text",
                "gpt_scores": gpt_scores
            })
        except Exception as e:
            print(f"Error processing query '{query}' with HuggingFace: {e}")
            results.append({
                "query": query,
                "hf_answer": "Error",
                "type": "text",
                "gpt_scores": {
                    "helpfulness": 0,
                    "correctness": 0,
                    "coherence": 0,
                    "relevance": 0
                }
            })

# Separate text-based and image-based results
text_results = [r for r in results if r["type"] == "text"]
image_results = [r for r in results if r["type"] == "image"]

# Extract scores for plotting
metrics = ["helpfulness", "correctness", "coherence", "relevance"]
text_scores = {metric: [r["gpt_scores"][metric] for r in text_results] for metric in metrics}
image_scores = {metric: [r["gpt_scores"][metric] for r in image_results] for metric in metrics}

# Visualization
x = np.arange(len(text_results))  # Query indices for text-based queries
width = 0.2  # Bar width
colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33"]  # Define custom colors for metrics

fig, ax = plt.subplots(figsize=(14, 8))

# Plot bars for each metric for text queries
for i, metric in enumerate(metrics):
    ax.bar(
        x + i * width,  # Offset positions for text metrics
        text_scores[metric],
        width,
        label=f"Text - {metric.capitalize()}",
        color=colors[i],
        alpha=0.8,
        edgecolor="black",
    )

# Since HuggingFace doesn't evaluate images, image results are placeholders
for i, metric in enumerate(metrics):
    ax.bar(
        x + len(text_results) + 1 + i * width,  # Offset positions for image metrics
        image_scores[metric],
        width,
        label=f"Image - {metric.capitalize()}",
        color=colors[i],
        alpha=0.4,  # Lighter alpha for image placeholders
        edgecolor="black",
    )

# Formatting
ax.set_title("Comparison of Performance Metrics Between Text and Image Queries (HuggingFace)", fontsize=16, fontweight="bold")
ax.set_ylabel("Score (0-5)", fontsize=12)
ax.set_xlabel("Query Index", fontsize=12)

# Create combined x-ticks for text and image queries
all_x = np.concatenate([x, x + len(text_results) + 1])
all_labels = [f"T{i+1}" for i in range(len(text_results))] + [f"I{i+1}" for i in range(len(image_results))]
ax.set_xticks(all_x + width)  # Center x-ticks between grouped bars
ax.set_xticklabels(all_labels, rotation=45, fontsize=10)

# Add gridlines and legend
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=10, frameon=False)

# Tight layout
plt.tight_layout()
plt.show()
