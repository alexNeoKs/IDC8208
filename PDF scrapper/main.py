import os
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from PyPDF2 import PdfReader

def fetch_arxiv_data(query, start=0, max_results=200, start_date=None, end_date=None):
    """Fetch metadata from arXiv API with optional date filters."""
    base_url = "http://export.arxiv.org/api/query"
    search_query = f"all:{query}"
    if start_date and end_date:
        search_query += f" AND submittedDate:[{start_date} TO {end_date}]"
    
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch data from arXiv API (Status: {response.status_code})")
        return None


def parse_arxiv_data(xml_data):
    """Parse XML data from arXiv API to extract paper IDs and construct valid PDF links."""
    root = ET.fromstring(xml_data)
    pdf_links = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"
        pdf_links.append(pdf_url)
    return pdf_links


def download_pdf(link, save_folder):
    """Download a single PDF."""
    paper_id = link.split('/')[-1]
    filename = f"{paper_id}.pdf"
    file_path = os.path.join(save_folder, filename)
    
    if os.path.exists(file_path):
        return f"{filename} already exists"

    try:
        response = requests.get(link, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            return f"{filename} downloaded successfully"
        else:
            return f"Failed to download {filename} (Status: {response.status_code})"
    except requests.exceptions.RequestException as e:
        return f"Error downloading {filename}: {e}"


def download_pdfs_concurrently(pdf_links, save_folder="downloaded_papers"):
    os.makedirs(save_folder, exist_ok=True)
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_link = {executor.submit(download_pdf, link, save_folder): link for link in pdf_links}
        for future in tqdm(as_completed(future_to_link), total=len(pdf_links), desc="Downloading PDFs", unit="file"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")

    for res in results:
        print(res)


def search_arxiv_by_date(query, save_folder="downloaded_papers"):
    # Define date range from 1900 to now
    start_date = "1900-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    start = 0
    max_results = 10000  
    total_downloaded = 0

    while True:
        print(f"Fetching results starting from {start} for date range {start_date} to {end_date}...")
        xml_data = fetch_arxiv_data(
            query, start=start, max_results=max_results, start_date=start_date, end_date=end_date
        )
        if not xml_data:
            break

        pdf_links = parse_arxiv_data(xml_data)
        if not pdf_links:
            print("No more results found.")
            break

        download_pdfs_concurrently(pdf_links, save_folder=save_folder)
        total_downloaded += len(pdf_links)
        start += max_results

    print(f"Total PDFs downloaded: {total_downloaded}")


topics = ["AI", "RAG", "Deep learning", "Machine Learning", "Multi Modal", "neural network"]

for topic in topics:
    print(f"\nStarting search for topic: {topic}")
    try:
        # Call the search function for each topic
        search_arxiv_by_date(query=topic, save_folder=f"downloaded_papers/{topic.replace(' ', '_')}")
    except Exception as e:
        print(f"Error occurred while processing topic {topic}: {e}")
        

def is_valid_pdf(file_path):
    """Check if a PDF file can be opened."""
    try:
        # Attempt to open the PDF
        reader = PdfReader(file_path)
        # Check if the file has at least one page
        return len(reader.pages) > 0
    except Exception:
        # If any error occurs, the file is invalid
        return False

def clean_invalid_pdfs(folder_path):
    """Check all PDFs in the folder and delete invalid ones."""
    total_files = 0
    deleted_files = 0

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Skip non-PDF files
        if not file_name.endswith(".pdf"):
            continue

        total_files += 1

        # Check if the PDF is valid
        if not is_valid_pdf(file_path):
            os.remove(file_path)  # Delete invalid file
            deleted_files += 1

    # Calculate remaining files
    remaining_files = total_files - deleted_files

    print(f"Total PDFs checked: {total_files}")
    print(f"Invalid PDFs deleted: {deleted_files}")
    print(f"Remaining valid PDFs: {remaining_files}")

    return remaining_files

folder_path = "downloaded_papers"
remaining_pdfs = clean_invalid_pdfs(folder_path)