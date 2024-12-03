from flask import Flask, render_template, request, jsonify
import os
import json
import time
from PIL import Image
import torch
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings  # Correct import for embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Ensure a folder for uploaded files exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load configuration from environment variables
source_directory = os.environ.get("SOURCE_DIRECTORY")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
chunk_size = 250
chunk_overlap = 50
model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
model_n_batch = int(os.environ.get("MODEL_N_BATCH", 8))
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 16))

# Load the FAISS index and setup RetrievalQA
embeddings = None
retriever = None
qa = None

def initialize_rag():
    global embeddings, retriever, qa
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

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
            raise Exception(f"Unsupported model type: {model_type}")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

initialize_rag()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"reply": "Please provide a valid input."})

    start = time.time()
    try:
        res = qa(user_message)
        answer = res["result"]
        sources = [doc.metadata["source"] for doc in res.get("source_documents", [])]
        end = time.time()

        response = {
            "reply": f"{answer}",
            "sources": sources,
            "response_time": f"{round(end - start, 2)} seconds"
        }
    except Exception as e:
        response = {"reply": f"Error processing your request: {str(e)}"}

    return jsonify(response)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"reply": "No file provided."})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"reply": "No file selected."})

    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        
        # Process the image (e.g., extract text or features)
        # Example: Using a placeholder for image-to-text conversion
        processed_text = f"Image {file.filename} processed successfully."
        
        return jsonify({"reply": processed_text})
    except Exception as e:
        return jsonify({"reply": f"Error processing image: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
