import os
from pypdf import PdfReader
from typing import List , Dict , Any


def load_pdf(file_path: str)-> List[Dict[str,Any]]:

    """
    Loads a PDF file and extracts text page by page.
    Returns a list of dictionaries containing text and metadata.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}") 

    file_name = os.path.basename(file_path)
    reader = PdfReader(file_path)
    Documents = []

    for page_num , page in enumerate(reader.pages, start = 1):

        text = page.extract_text() or ""
        text = text.strip()

        if text :
            Documents.append({
                "text": text,
                "metadata":{
                    "page_number":page_num,
                    "source": file_name
                }
            })



    return Documents 