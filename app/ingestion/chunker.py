from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List , Dict , Any



def chunk_documents(document :List[Dict[str,Any]], chunk_size : int = 500 , chunk_overlap : int = 50
                    )->List[Dict[str,Any]]:

    """
    Takes loaded page documents and splits them into smaller, 
    overlapping chunks while preserving metadata and adding chunk IDs.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n", "\n"," " ,""]
    )


    all_chunks = []

    for doc in document:

        text = doc["text"]
        data = doc["metadata"]


        raw_chunk = splitter.split_text(text)
        source_clean = data["source"].replace(".","_").replace(" ","_")
        page_num = data["page_number"]

        for idx , chunk in enumerate(raw_chunk,start = 1):

            chunk_id = "{source_clean}_p{page_num}_c{idx}"

            all_chunks.append({
                "chunk_id":chunk_id,
                "chunk_text":chunk,
                "source":data["source"],
                "page_number": page_num
            })


    return all_chunks