'''
Import statements
'''

import os
import time
import glob
import json
import torch
import openai
import ollama
import base64
import requests
import chardet
import argparse
import wikipediaapi
from PIL import Image       
from pdfplumber import PDF  
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain_huggingface import HuggingFaceEmbeddings 
from sklearn.metrics import precision_score, recall_score
from nltk.translate.bleu_score import sentence_bleu
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

'''
    Sets up the persistent cache file.
    This is to improve retrival times for repeated queries.
'''

cache_file = "query_cache.json"
if os.path.exists(cache_file): # Load cache from file or initialize an empty cache
    with open(cache_file, "r") as f:
        query_cache = json.load(f)
else:
    query_cache = {}

'''
    To utilise GPT-4 as LLM-as-a-Judge. An openai account with credits is required btw but if you do not plan to use the evaluation then just leave it blank.
    Please utilise this sparingly. I have limited credits prof. :(
'''
#OpenAi secret Key
openai.api_key = "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"

'''
    Checking to see if GPU is available 
    Ensures that PyTorch can detect the GPU and provides GPU details like the name and count.
'''
print("Is CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

'''
    load_dotenv(): Loads configuration variables from a .env file into the environment for flexibility.
    
        source_directory: Directory where source documents are located.
        embeddings_model_name: Name of the embeddings model.
        chunk_size and chunk_overlap: Parameters for splitting text into chunks for processing.
        model_type, model_path, etc.: Configuration for the LLM backend
'''

load_dotenv()
source_directory = os.environ.get("SOURCE_DIRECTORY")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
chunk_size = 250 #was 500
chunk_overlap = 50
model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
model_n_batch = int(os.environ.get("MODEL_N_BATCH", 8))
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 16)) # was 4

'''
    This function sends an image to ollama (presumably a multimodal AI API) and retrieves a description.
'''
def img_parse(img_path):
    # Generate a description of the image using ollama
    res = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": "Describe this image:",
                "images": [img_path],
            }
        ],
    )
    # Modify the file name to remove the original extension and use .txt
    base_name = os.path.splitext(os.path.basename(img_path))[0]  # Extract file name without extension
    txt_filename = f"{base_name}.txt"  # Append .txt extension

    # Write the description to the .txt file
    txt_file_path = os.path.join(source_directory, txt_filename)
    with open(txt_file_path, "w", encoding="utf-8") as write_file:
        write_file.write("---" * 10 + "\n\n")
        write_file.write(f"Image Name: {base_name}\n\n")
        write_file.write(res["message"]["content"])
        write_file.flush()

    print(f"Description saved for {img_path} as {txt_file_path}")
    
'''
    A wrapper around UnstructuredEmailLoader that handles emails without HTML content by falling back to plain text.
'''
# Custom document loaders for emails
class MyElmLoader(UnstructuredEmailLoader):
    def load(self) -> List[Document]:
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            raise Exception(f"{self.file_path}: {e}") from e
        return doc
"""
    Custom loader for processing images. Extracts data (e.g., descriptions) from the given file path.
"""
class ImgLoader:
    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs  # Store any additional arguments if needed

    def load(self) -> List[Document]:
        """
            Load the image, process it, and return it as a Document.
        """
        try:
            # Process the image and generate a description
            img_parse(self.file_path)
            
            # Clean up the file path to locate the generated .txt file
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]  # Extract file name without extension
            txt_file = os.path.join(source_directory, f"{base_name}.txt")  # Corresponding .txt file

            # Use TextLoader to load the .txt file into a Document
            loader = TextLoader(txt_file, **self.kwargs)
            doc = loader.load()
        except Exception as e:
            # Add the file path to the exception message for better debugging
            raise Exception(f"Error processing {self.file_path}: {e}") from e
        
        return doc

"""
    Custom TextLoader to handle encoding issues gracefully.
"""
class RobustTextLoader(TextLoader):
    def load(self) -> List[Document]:
        try:
            # Attempt to load using the base TextLoader
            return super().load()
        except UnicodeDecodeError as e:
            # Detect encoding and attempt to reload
            with open(self.file_path, 'rb') as f:
                raw_data = f.read()
                detected_encoding = chardet.detect(raw_data)['encoding']
                if not detected_encoding:
                    raise RuntimeError(f"Could not detect encoding for {self.file_path}") from e
                
                # Decode and create a Document object
                decoded_text = raw_data.decode(detected_encoding)
                return [Document(page_content=decoded_text, metadata={"source": self.file_path})]

"""
    Extract text content from a PDF.
"""
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    print(f"Extracting text from PDF: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Extracted {len(documents)} text chunks from PDF: {pdf_path}")
    return [doc.page_content for doc in documents]  # List of text chunks

"""
    Extract images from a PDF and save them.
"""
def extract_images_from_pdf(pdf_path: str) -> List[str]:
    print(f"Extracting images from PDF: {pdf_path}")
    images = []
    with PDF.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for img in page.images:
                img_bytes = img["image"]
                img_file = os.path.join(source_directory, f"{os.path.basename(pdf_path)}_page_{i}.png")
                with open(img_file, "wb") as f:
                    f.write(img_bytes)
                images.append(img_file)
                print(f"Saved image: {img_file}")
    if not images:
        print(f"No images found in PDF: {pdf_path}")
    return images

"""
    Extract tables from a PDF and return them as CSV-like documents.
"""
def extract_tables_from_pdf(pdf_path: str) -> List[Document]:
    print(f"Extracting tables from PDF: {pdf_path}")
    documents = []
    with PDF.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if tables:
                print(f"Found {len(tables)} tables on page {page_num} of PDF: {pdf_path}")
            for table_num, table in enumerate(tables, start=1):
                content = "\n".join([",".join(row) for row in table])  # Convert to CSV-like text
                doc = Document(page_content=content, metadata={"source": pdf_path, "type": "table", "page": page_num, "table_num": table_num})
                documents.append(doc)
                print(f"Extracted table {table_num} from page {page_num} of PDF: {pdf_path}")
    if not documents:
        print(f"No tables found in PDF: {pdf_path}")
    return documents

"""
    Extract images from a PDF and save them.
"""
def extract_images_from_pdf(pdf_path: str) -> List[str]:
    print(f"Extracting images from PDF: {pdf_path}")
    images = []
    with PDF.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for img in page.images:
                img_bytes = img["image"]
                img_file = os.path.join(source_directory, f"{os.path.basename(pdf_path)}_page_{i}.png")
                with open(img_file, "wb") as f:
                    f.write(img_bytes)
                images.append(img_file)
                print(f"Saved image: {img_file}")
    if not images:
        print(f"No images found in PDF: {pdf_path}")
    return images

"""
    Encodes an image to a Base64 string to include it in the prompt.
"""
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

"""
    Process user input. If it's an image file, describe it and return the description.
    Otherwise, return the text input.
"""
def handle_input(input_text):
    if os.path.isfile(input_text) and input_text.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        # Process the image using img_parse to generate a description
        print(f"Processing image: {input_text}")
        try:
            res = ollama.chat(
                model="llava",
                messages=[
                    {
                        "role": "user",
                        "content": "Describe this image:",
                        "images": [input_text],
                    }
                ],
            )
            image_description = res["message"]["content"]
            # print(f"Image description: {image_description}")
            return f"Image description: {image_description}"  # Use this as the query
        except Exception as e:
            print(f"Error processing image: {e}")
            return "Error processing image."
    else:
        return input_text

"""
    Trim the prompt to fit within the maximum token size.
"""
def trim_prompt(prompt, max_tokens=2048):
    if len(prompt.split()) > max_tokens:
        print(f"Prompt too large ({len(prompt.split())} tokens). Trimming...")
        prompt = " ".join(prompt.split()[:max_tokens])
    return prompt


"""
    Fetches the summary of a Wikipedia page for a given query using the REST API.

    Args:
        query (str): The topic to search for on Wikipedia.
        lang (str): The language for the Wikipedia API (default is 'en').

    Returns:
        str: The summary of the Wikipedia page or an error message if not found.
"""
def fetch_wikipedia_summary(query: str, lang: str = "en") -> str:
    user_agent = "MyRAGSystem/1.0 (contact: neoalexanderks@gmail.com)"  # Please do not spam me
    headers = {"User-Agent": user_agent}
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if "extract" in data:
            return data["extract"]
        else:
            return f"Wikipedia page for '{query}' does not have a summary."
    elif response.status_code == 404:
        return f"Wikipedia page for '{query}' does not exist."
    else:
        return f"Error: Unable to fetch Wikipedia data (HTTP {response.status_code})."

'''
    Functions for evaluation
'''
# Calculate relevance between retrieved contexts and ground truth context
def calculate_relevance(retrieved_contexts, ground_truth_context):
    return len(set(retrieved_contexts) & set(ground_truth_context)) / len(set(ground_truth_context))

# Evaluate generated answers against ground truth
def evaluate_generation(generated_answer, ground_truth_answer):
    return sentence_bleu([ground_truth_answer.split()], generated_answer.split())

'''
    Map file extensions to document loaders and their arguments
'''
LOADER_MAPPING = {
    # Text-based file formats
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".txt": (RobustTextLoader, {"encoding": "utf8"}),

    # PDF handling with multi-modal capabilities
    ".pdf": (PyMuPDFLoader, {}),  # Default PDF loader
    ".pdf_text": (PyMuPDFLoader, {}),  # Extract text content
    ".pdf_images": (ImgLoader, {}),  # Extract images embedded in PDFs
    ".pdf_tables": (CSVLoader, {}),  # Extract tables as CSV files

    # Presentation formats
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),

    # Image-based file formats
    ".jpg": (ImgLoader, {}),
    ".png": (ImgLoader, {}),
    ".tiff": (ImgLoader, {}),
    ".svg": (UnstructuredHTMLLoader, {}),  # Optionally load vector images

    # Additional vector formats for advanced processing
    ".xml": (UnstructuredHTMLLoader, {}),  # For flowcharts or structured data
}

'''
    Loads a single document using the appropriate loader based on its file extension.
'''
def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    # Multi-modal PDF processing
    if ext == ".pdf":
        print(f"Processing multi-modal PDF: {file_path}")
        documents = []

        # Extract text
        text_chunks = extract_text_from_pdf(file_path)
        for i, chunk in enumerate(text_chunks, start=1):
            doc = Document(page_content=chunk, metadata={"source": file_path, "type": "text", "chunk_num": i})
            documents.append(doc)

        # Extract images
        image_files = extract_images_from_pdf(file_path)
        for img_path in image_files:
            print(f"Processing image: {img_path}")
            img_doc = Document(page_content=f"Image file: {img_path}", metadata={"source": file_path, "type": "image"})
            documents.append(img_doc)

        # Extract tables (if applicable)
        table_docs = extract_tables_from_pdf(file_path)
        documents.extend(table_docs)

        print(f"Completed processing PDF: {file_path} ({len(documents)} items created)")
        return documents

    raise ValueError(f"Unsupported file extension '{ext}'")

'''
    Scans the source directory for supported files and loads them using multiprocessing for efficiency.
'''
def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files.
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True))
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    
    documents = []  # Store successfully loaded documents
    with Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=len(filtered_files), desc="Loading new documents", ncols=80) as pbar:
            for file_path in filtered_files:
                try:
                    # Process each file and append the result
                    doc = pool.apply(load_single_document, (file_path,))
                    if doc:  # Ensure doc is not None or empty
                        documents.extend(doc)
                except Exception as e:
                    # Log the error and skip the file
                    print(f"Error processing file {file_path}: {e}")
                finally:
                    pbar.update()
    
    return documents

"""
    Load documents and split them into chunks, including multi-modal PDF processing and optional Wikipedia enrichment.

    Args:
        ignored_files (List[str]): List of files to ignore during processing.
        enrich_with_wikipedia (bool): Whether to fetch additional data from Wikipedia.

    Returns:
        List[Document]: List of processed documents split into chunks.
"""
def process_documents(ignored_files: List[str] = [], enrich_with_wikipedia: bool = True) -> List[Document]:
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)

    print(f"Loaded {len(documents)} new documents from {source_directory}")

    # Log image-related documents
    image_docs = [doc for doc in documents if doc.metadata.get("type") == "image"]
    print(f"Found {len(image_docs)} image-related documents.")

    # Enrich corpus with Wikipedia data
    if enrich_with_wikipedia:
        topics_to_enrich = ["Artificial Intelligence", "Machine Learning", "Deep Learning"]  # Example topics
        for topic in topics_to_enrich:
            print(f"Fetching Wikipedia summary for: {topic}")
            wiki_summary = fetch_wikipedia_summary(topic)
            if wiki_summary and "does not exist" not in wiki_summary:
                # Add Wikipedia summary as a new document
                documents.append(
                    Document(
                        page_content=wiki_summary,
                        metadata={"source": "Wikipedia", "topic": topic, "type": "wiki_summary"},
                    )
                )
                print(f"Added Wikipedia summary for topic: {topic}")

    # Split text documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

# Function to use GPT as a judge
def gpt_as_judge(query, generated_answer, ground_truth_answer):
    prompt = f"""
    Evaluate the following generated answer based on the user's query and the ground truth answer:
    
    - **Query**: {query}
    - **Generated Answer**: {generated_answer}
    - **Ground Truth Answer**: {ground_truth_answer}
    
    Provide scores (0-5) for the following metrics:
    1. **Helpfulness**: How well does the answer address the query?
    2. **Correctness**: Is the answer factually accurate?
    3. **Coherence**: Is the answer well-structured and logically sound?
    4. **Relevance**: Is the answer closely related to the query and context?

    Respond only with JSON format:
    {{
        "helpfulness": <score>,
        "correctness": <score>,
        "coherence": <score>,
        "relevance": <score>
    }}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant evaluating generated answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        raw_response = response["choices"][0]["message"]["content"].strip()
        print(f"Raw GPT Response: {raw_response}")  # Debugging line
        return json.loads(raw_response)
    except Exception as e:
        print(f"Error using GPT as judge: {e}")
        return None
    
    
'''
    Main loop to call all the functions.
        It sets up a retriever, loads a pre-trained large language model (LLM), and processes a dataset of queries 
        to evaluate the system's performance. The results include metrics such as helpfulness, correctness, coherence, 
        and relevance, which are evaluated using GPT-based scoring. The function also generates a visualization comparing 
        the performance of text-based and image-based queries.
    Key Steps:
        - Initialize embeddings using HuggingFace.
        - Load or create a FAISS vector store.
        - Set up the LLM based on the specified model type.
        - Evaluate queries from the dataset, including both text and image-based queries.
        - Generate and store evaluation metrics.
        - Visualize and compare performance metrics for text and image queries.

    Dependencies:
        - HuggingFaceEmbeddings for embeddings generation.
        - FAISS for vector store creation and retrieval.
        - LlamaCpp or GPT4All for LLM functionality.
        - Matplotlib for visualization.
    @note Ensure required resources (e.g., FAISS index, LLM model files, dataset) are accessible before execution.
'''
def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)  # Initializes embeddings with GPU support.
    try:
        os.listdir("faiss_index")
        print("FAISS index already exists. Loading...")
    except FileNotFoundError:
        print("FAISS index not found. Creating new vectorstore...")
        # Load and process documents
        texts = process_documents()
        if not texts:
            print("No documents were processed. Check the source directory.")
            exit(1)
        print(f"Creating embeddings for {len(texts)} text chunks...")
        # Create FAISS index
        db = FAISS.from_documents(texts, embeddings)
        db.save_local("faiss_index")
        print("FAISS index created successfully.")
    
    print("Loading FAISS index...")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    print("FAISS index loaded successfully.")

    # Initialize LLM
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=12,
                n_batch=512,
                f16_kv=True,
                callback_manager=callback_manager,
                verbose=True,
            )
        case "GPT4All":
            llm = GPT4All(
                model=model_path,
                n_ctx=model_n_ctx,
                backend="gptj",
                n_batch=model_n_batch,
                callback_manager=callback_manager,
                verbose=True,
                device="cuda",
            )
        case _default:
            raise Exception(f"Model type {model_type} is not supported.")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Load the JSON dataset
    with open("ai_rag_ground_truth.json", "r") as file:
        dataset = json.load(file)

    results = []  # Store evaluation results

    for entry in dataset:
        query = entry["query"]
        ground_truth_answer = entry["answer"]
        ground_truth_context = entry.get("context", "")

        # Retrieve context
        if "image_reference" in entry:
            # For image-based queries
            image_path = entry["image_reference"]
            processed_query = handle_input(image_path)
        else:
            # For text-based queries
            processed_query = handle_input(query)

        processed_query = trim_prompt(processed_query, max_tokens=2048)
        start = time.time()
        res = qa(processed_query)
        end = time.time()

        # Extract retrieved contexts and generated answer
        retrieved_contexts = [doc.page_content for doc in res["source_documents"]] if "source_documents" in res else []
        generated_answer = res["result"]

        # Evaluate with GPT as a Judge
        gpt_scores = gpt_as_judge(query, generated_answer, ground_truth_answer)
        if not gpt_scores:
            gpt_scores = {"helpfulness": 0, "correctness": 0, "coherence": 0, "relevance": 0}  # Default if GPT fails

        print(f"\nGPT Scores for Query: {query}")
        print(f"Helpfulness: {gpt_scores['helpfulness']}, Correctness: {gpt_scores['correctness']}, "
              f"Coherence: {gpt_scores['coherence']}, Relevance: {gpt_scores['relevance']}")

        # Store results
        results.append({
            "query": query,
            "retrieved_contexts": retrieved_contexts,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "type": "image" if "image_reference" in entry else "text",
            "gpt_scores": gpt_scores,
            "latency": round(end - start, 2),
        })

    # Separate text and image queries
    text_results = [r for r in results if r["type"] == "text"]
    image_results = [r for r in results if r["type"] == "image"]

    # Compute average scores for each type
    def average_scores(results):
        return {
            "helpfulness": sum(r["gpt_scores"]["helpfulness"] for r in results) / len(results),
            "correctness": sum(r["gpt_scores"]["correctness"] for r in results) / len(results),
            "coherence": sum(r["gpt_scores"]["coherence"] for r in results) / len(results),
            "relevance": sum(r["gpt_scores"]["relevance"] for r in results) / len(results),
        }

    text_avg_scores = average_scores(text_results)
    image_avg_scores = average_scores(image_results)

    print("\nAverage Scores for Text-Based Queries:", text_avg_scores)
    print("Average Scores for Image-Based Queries:", image_avg_scores)

    # Visualization: Grouped bar chart for text vs. image performance
    import matplotlib.pyplot as plt
    import numpy as np

    categories = list(text_avg_scores.keys())
    text_values = list(text_avg_scores.values())
    image_values = list(image_avg_scores.values())

    x = np.arange(len(categories))  # Category indices
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, text_values, width, label="Text Queries")
    ax.bar(x + width / 2, image_values, width, label="Image Queries")

    # Formatting
    ax.set_title("Comparison of Text-Based vs. Image-Based Query Performance")
    ax.set_ylabel("Average Score (0-5)")
    ax.set_xlabel("Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
