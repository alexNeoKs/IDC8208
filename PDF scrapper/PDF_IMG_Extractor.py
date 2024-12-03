import os
from PyPDF2 import PdfReader
from PIL import Image
import io
import re

# Define the source folder and output folder
source_folder = "source"  # Replace with your source folder path
output_folder = "source_images"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def sanitize_filename(filename):
    """Sanitize the filename to remove invalid characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def extract_images_from_pdf(pdf_path, output_dir):
    """Extracts and saves images from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            # Check if the page has '/XObject' in '/Resources'
            resources = page.get('/Resources', {})
            if resources and '/XObject' in resources:
                x_objects = resources['/XObject'].get_object()
                for obj_name, obj_ref in x_objects.items():
                    obj = obj_ref.get_object()  # Dereference the IndirectObject
                    if obj.get('/Subtype') == '/Image':
                        # Get image properties
                        size = (obj.get('/Width'), obj.get('/Height'))
                        data = obj.get_data()
                        color_space = obj.get('/ColorSpace')
                        mode = "RGB" if color_space == '/DeviceRGB' else "P"
                        
                        # Create an image from raw data
                        img = Image.frombytes(mode, size, data)
                        
                        # Construct the output file path
                        base_name = sanitize_filename(os.path.splitext(os.path.basename(pdf_path))[0])
                        img_filename = f"{base_name}_page{page_number}_{sanitize_filename(obj_name)}.png"
                        img_path = os.path.join(output_dir, img_filename)
                        
                        # Save the image
                        img.save(img_path)
                        print(f"Saved image: {img_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def main():
    # List all PDF files in the source folder
    pdf_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]

    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(source_folder, pdf_file)
        print(f"Processing: {pdf_path}")
        extract_images_from_pdf(pdf_path, output_folder)

if __name__ == "__main__":
    main()
