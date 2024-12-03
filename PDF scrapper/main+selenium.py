from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

def setup_driver():
    """Set up the Selenium WebDriver."""
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-usb-discovery")
    driver_path = "chromedriver-win64/chromedriver.exe"  # Update with your actual ChromeDriver path
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def solve_recaptcha(driver):
    """Open the main site and wait for manual reCAPTCHA solving."""
    arxiv_url = "https://arxiv.org"
    print(f"Opening arXiv homepage: {arxiv_url}")
    driver.get(arxiv_url)

    # Wait for manual reCAPTCHA resolution
    input("Solve the reCAPTCHA manually in the browser, then press Enter here to continue...")


def search_and_scrape(driver, query, save_folder="downloaded_papers"):
    """Search a query on arXiv and scrape PDF links."""
    base_url = "https://arxiv.org/search/"
    start = 0
    page_size = 200
    total_downloaded = 0

    while True:
        print(f"Searching for query: {query} starting from result {start}")
        search_url = f"{base_url}?query={query.replace(' ', '+')}&searchtype=all&size={page_size}&start={start}"
        retries = 3
        for attempt in range(retries):
            try:
                driver.get(search_url)
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "arxiv-result"))
                )
                break  # Exit loop if successful
            except Exception:
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(10)
        else:
            print("Failed to load page after retries.")
            return

        # Parse the search results
        soup = BeautifulSoup(driver.page_source, "html.parser")
        papers = soup.find_all("li", class_="arxiv-result")
        print(f"Number of papers found on this page: {len(papers)}")

        if not papers:
            print("No more results found.")
            break

        # # Extract and download PDF links immediately
        # for paper in papers:
        #     pdf_link = paper.find("a", href=True)  # Find the first anchor with href

        #     if pdf_link:  # If a link is found
        #         href = pdf_link["href"]
        #         link_text = pdf_link.get_text(strip=True).lower()  # Get the text and convert to lowercase
                
        #         # Check if the text contains "pdf" and the href exists
        #         if "pdf" in link_text:
        #             print(f"Found PDF link: {href}")

        #             # Download the PDF immediately
        #             download_result = download_pdf(href, save_folder)
        #             print(download_result)  # Print the result of the download
        #             total_downloaded += 1  # Increment the count
        #         else:
        #             print("Link found, but it doesn't indicate a PDF.")
        #     else:
        #         print("No links found in this paper.")
        # Extract and append PDF links to a text file
        output_file = "pdf_links.txt"  # File to store the URLs

        for paper in papers:
            pdf_link = paper.find("a", href=True)  # Find the first anchor with href

            if pdf_link:  # If a link is found
                href = pdf_link["href"]
                link_text = pdf_link.get_text(strip=True).lower()  # Get the text and convert to lowercase
                
            
                with open(output_file, "a") as file:
                    file.write(href + "\n")  # Write the URL to the file
             
            else:
                print("No links found in this page.")

        # Add delay to avoid being flagged
        time.sleep(random.randint(5, 10))

        # Move to the next page
        start += page_size

    print(f"Total PDFs downloaded for query '{query}': {total_downloaded}")




def download_pdf(link, save_folder):
    """Download a single PDF."""
    filename = link.split("/")[-1] + ".pdf"
    file_path = os.path.join(save_folder, filename)

    if os.path.exists(file_path):
        return f"{filename} already exists"

    try:
        response = requests.get(link, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            return f"{filename} downloaded successfully"
        else:
            return f"Failed to download {filename} (Status: {response.status_code})"
    except Exception as e:
        return f"Error downloading {filename}: {e}"


def download_pdfs_concurrently(pdf_links, save_folder="downloaded_papers"):
    """Download PDFs concurrently."""
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


# Main Script
if __name__ == "__main__":
    topics = ["AI", "RAG", "Deep learning", "Machine Learning", "Multi Modal", "neural network"]
    driver = setup_driver()

    try:
        # Step 1: Solve reCAPTCHA once
        solve_recaptcha(driver)

        # Step 2: Process each query
        for topic in topics:
            print(f"\nProcessing topic: {topic}")
            search_and_scrape(driver, query=topic, save_folder=f"downloaded_papers/{topic.replace(' ', '_')}")
    finally:
        driver.quit()
        print("Browser closed.")
