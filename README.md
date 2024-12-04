
# <center> **Open-domain Text-Centric Multi-Modal RAG for Research Insights** </center>

<center>IDC 8208 – Implementation of Advance AI</center>
<center>
    Name: Neo Kah Siak <br/>
    Student ID: 2101960 <br/>
    Academic Supervisor: Prof Kar Peo Yar , Prof Malcolm Low <br/><br/>
</center>


* **Features**
    - Q&A with LLM + Wikipedia about domain specific knowledge
    - Run locally
    - Structured Corpus for different file type handling
    - Returns relevant resources with citations
    - Text-centric RAG design 

* **Process**
    - Parse images in the folder into text with LLava, which is then ran locally with ollama, and is capable of ingesting other text file types with LangChain.
    - Ingest the text into vectorDB (FAISS)
    - Query it with LLM of choice. (Currently support all llamacpp , gpt formats)
<br/><br/>
<h4>Installation</h4>

Its heavily recommended you create a virtual environment to run this project.

**Create and activate virtual environment**
        
```bash
    python -m venv project
    source project/bin/activate
```
Clone repo and install dependencies

```bash
    git clone https://github.com/alexNeoKs/IDC8208.git
    python -m pip install -r requirements.txt
    cp example.env .env #Create a copy of the environ file
```

Download model files

- Put local LLM weights into folder _models_, supporting any GGUF format, and change the MODEL_PATH in .env for your model path. You can download the weights by visiting this guy at [Huggingface/theBloke](https://huggingface.co/TheBloke). I use [mistral-7b-instruct-v0.1.Q4_K_S.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) as our LLM for query. But feel free to add your own.
- I currently employed the HuggingfaceEmbedding, but you can change it to local embedding like GPT4ALLEmbedding by changing the EMBEDDINGS_MODEL_NAME in .env.
- Run MLLM. We employ the latest llava 1.6 for image parsing.

Ensure you have ollama installed on your device. if it is not installed you can grab a copy
at (https://ollama.com/download)

- **Environment variables setting**
    
    - Change the environment variables according to your needs in .env. _SOURCE_DIRECTORY_ refers to the folder which contains all the images you want to retrieve.

- **Run**
    
    Put all the files you want to talk with into the folder _source_. Run following command:
    
    ```bash
    python RAG_Evaluation.py
    ```
    It will generate the folder _source_documents_ as the storage of parsed text and _faiss_index_ as the vectorDB. If the two folders already exist, it will start query directly.

    ```bash
    python app.py
    ```
    After the faiss index is generated, you can use the app.py to launch the flask interface.

*Do note that to do evaluation you need to have an openAI API key. I placed mine in the code as reference on where you should add yours. But if you added yours to environmental variables that would work too. The api key in the code is not functional by the way, it has already been disabled*

<h2>Introduction </h2>
The exponential growth in scientific literature across various fields poses a significant challenge for researchers trying to stay current with the latest advancements. [1] Researchers often need to review hundreds of papers to extract key insights relevant to their work. Interestingly, this challenge has led to the development of several innovative solutions. For instance, OpenResearcher, an AI-powered platform, has been introduced to accelerate the research process by answering diverse questions from researchers. [2] AI-driven systems for paper summarization and retrieval have made significant strides in recent years, leveraging advanced natural language processing techniques to extract key information from research papers. AI-driven systems are increasingly multimodal, enabling them to process complex research papers that combine text, figures, tables, and charts. However, many current implementations still face challenges in effectively integrating these diverse data types for comprehensive insights. [3,4] Additionally, the complexity of scientific language and domain-specific terminology poses significant challenges for general-purpose text summarization models. To bridge this gap, I propose developing an open-domain multimodal Retrieval-Augmented Generation (RAG) system designed to help researchers efficiently read, summarize, and extract critical information from academic papers.

<h3>Background</h3>
Retrieval-augmented generation (RAG) combines retrieval-based methods with deep learning to address the limitations of large language models (LLMs), such as hallucination, by dynamically incorporating external information [5]. While LLMs generate coherent text, they often rely on static training data, leading to outdated or incorrect outputs. RAG mitigates this by retrieving relevant, up-to-date data during generation, enhancing accuracy and reliability using real-world information. The core of RAG involves retrieving documents from sources like databases, research papers, or the web and grounding responses in factual data. This dynamic integration improves contextual appropriateness and reduces reliance on pre-trained, potentially outdated knowledge. RAG has proven effective in applications like Arabic text generation, where it combines semantic embedding models for retrieval with LLMs, addressing language-specific challenges and resource constraints [6]. Despite its potential, RAG faces challenges such as seamlessly integrating retrieved content into generated text, managing irrelevant or erroneous information, and mitigating bias from external sources. These issues highlight ongoing research needs to ensure RAG systems consistently deliver trustworthy, accurate responses [7].

<h3>Problem statement </h3>
Academic research papers often integrate multimodal elements like text, images, tables, and graphs, posing challenges for traditional RAG systems, which excel in processing textual data but lack robust multimodal capabilities. This limitation hinders their effectiveness in domains such as scientific research, where insights require interpreting and integrating multiple modalities. Non-textual elements, like images and tables, are often critical but treated as supplementary, leading to incomplete knowledge representation and reduced response reliability. To bridge this gap, this project explores multimodal RAGs by converting non-textual data into descriptive textual embeddings, enabling seamless integration within existing RAG frameworks.

<h2>Literature Review</h2>
<h4>Exponential Growth of Scientific Literature and Challenges for Researchers</h4>
The exponential growth of scientific literature presents significant challenges for researchers in managing and analysing vast volumes of publications. This rapid increase in research output has made it increasingly difficult for scientists to stay current with the latest field developments and identify relevant work efficiently. [8,9] The factors contributing to this growth are complex and multifaceted, including both external and internal factors affecting scientific literature development [10] This exponential growth has led to an unprecedented volume of scientific outputs, particularly evident in fields such as COVID-19 research, where significant scientific and financial efforts have been concentrated. The sheer volume of publications questions the capacity of scientists, policymakers, and citizens to maintain infrastructure, digest content, and make scientifically informed decisions. To address these challenges, various approaches and tools have been developed. LLAssist, for example, leverages Large Language Models (LLMs) and Natural Language Processing (NLP) techniques to automate key aspects of the literature review process, helping researchers focus more on analysing and synthesising information. [9] Other solutions include the development of knowledge models for unveiling meaningful relationships among articles based on topics and latent citation dependencies and using graphical abstracts and visual summaries to facilitate the exploratory process. [8]

<h4>Limitations of Current AI-driven Systems in Handling Multimodal Data</h4>
AI-driven systems have made remarkable progress in tasks such as text summarization, information retrieval, and language generation. However, effectively processing multimodal data—data combining diverse forms such as text, images, tables, and charts—remains a significant challenge, especially in academic research papers, where insights often rely on the interplay of textual and visual elements. The primary challenge in handling multimodal data lies in integrating diverse data types into a unified representation. Each modality—text, images, tables, and graphs—has unique structures and semantic meanings, complicating the process of understanding their relationships. Current systems struggle with "multimodal fusion," the process of combining inputs from different modalities in a manner that preserves their individual information while creating a cohesive representation of the document. While progress is being made, models capable of robust multimodal fusion remain in early development, and existing implementations frequently encounter inconsistencies in interpreting the relationships between textual and visual content.

<h2>Methodology</h2>
<h4>Overview</h4>
The solution is a multimodal RAG system designed to assist in extracting domain-specific knowledge from research papers. Research papers contain multimodal content such as images and tables on top of plain texts. Unlike traditional multimodal RAG systems, which use specialized encoders for different data types, this system uniquely leverages text embeddings to represent all input data, regardless of modality. By converting images and tables into descriptive text, the system maintains a uniform embedding space, enabling seamless integration into the retrieval pipeline. The core methodology involves preprocessing data into chunks, embedding them using a text-based encoder, and storing the embeddings in a vector database. When queries are made, the system retrieves relevant chunks based on their similarity to the query, processes the retrieved information using a Large Language Model (LLM), and generates a domain-specific response. This approach simplifies the architecture while achieving multimodal capabilities through text embeddings alone.

<h4>Data retrieval</h4>
We used Selenium to automate the retrieval of PDFs from research websites, enabling seamless data collection for our system. The process involves navigating to target sites, performing search queries, identifying PDF links, and automating downloads. To handle dynamic content, we employed explicit waits and error handling, ensuring resilience against loading delays or missing links. The downloaded PDFs are stored in a structured directory, ready for integration into the Retrieval-Augmented Generation pipeline.
<h4>Preprocessing and extraction </h4>
This module is responsible for processing diverse multimodal inputs (text, images, and tables) into a structured, uniform corpus of text-based chunks enriched with metadata. The structured corpus is foundational for the system's retrieval and generation processes, ensuring efficient indexing, retrieval, and seamless multimodal integration.
<h4>Embedding Module</h4>
The Embedding Module converts text-based chunks from the structured corpus into dense vector representations. These embeddings serve as the foundation for similarity-based retrieval, enabling the system to locate the most relevant information efficiently.
The embedding model is initialized using the HuggingFaceEmbeddings class, with the specific model dynamically selected via the EMBEDDINGS_MODEL_NAME environment variable, allowing flexibility in choosing pre trained transformer models. Each chunk from the structured corpus, regardless of modality, is passed through the embedding model to generate a semantic vector representation. By converting all input modalities, such as text, image descriptions, and tables, into text, the system ensures that the same embedding model uniformly processes all chunks, enabling seamless multimodal integration.
This process is implemented using FAISS.from_documents to create the index, followed by saving it locally for reuse. When a query is received, it is converted into an embedding using the same HuggingFace model, and FAISS performs a nearest-neighbour search to identify the most similar embeddings in the corpus. This process is highly efficient, even for large datasets, thanks to FAISS’s optimized algorithms for approximate nearest-neighbor (ANN) searches. Each embedding in FAISS is linked to the metadata of its corresponding chunk, such as file name, chunk type, and page number, ensuring that retrieval results include both the embedding and its associated context for accurate and contextual responses.
<h4>Retrieval Module</h4>
The Retrieval Module identifies and retrieves the most relevant information chunks from the structured corpus in response to user queries. By leveraging Flask, it also serves as an interface for real-time interaction, allowing users to input queries through a web application and receive contextual responses.
