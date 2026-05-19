from langchain_community.document_loaders import PyPDFLoader
import os
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs(pdf_folders):
    pypdf.filters.ZLIB_MAX_OUTPUT_LENGTH = 1024 * 1024 *512

    all_docs=[]
    for pdf_folder in pdf_folders:
        for path in os.listdir(pdf_folder):
            if path.endswith(".pdf"):
                document=PyPDFLoader(os.path.join(pdf_folder,path))
                print("Found Document")
                all_docs.extend(document.load())
                print("Added Document")
        print('Done')
    return all_docs

def chunker(all_docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_documents(all_docs)
    print(len(chunks))
    return chunks