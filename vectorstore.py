from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDINGS_MODEL,FAISS_INDEX_PATH,PDF_FOLDERS
from loaders import load_pdfs,chunker
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PKL = os.path.join(BASE_DIR, "all_docs.pkl")

def build_or_load():
    embeddings=HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL,model_kwargs={'device':'cpu'})
    try:
        vectorstore=FAISS.load_local(FAISS_INDEX_PATH,embeddings=embeddings,allow_dangerous_deserialization=True)
        print('Loaded Existing Faiss Index')
        with open(DOCS_PKL,"rb") as f:
            all_docs=pickle.load(f)
    except (FileNotFoundError,RuntimeError):
        try:
            with open(DOCS_PKL, "rb") as f:
                all_docs=pickle.load(f)
                print('loaded stored documents')
        except (FileNotFoundError,RuntimeError):
            print("Building from scratch")
            all_docs=load_pdfs(PDF_FOLDERS)
            with open(DOCS_PKL,"wb") as f:
                pickle.dump(all_docs,f)
        chunks=chunker(all_docs)
        vectorstore=FAISS.from_documents(chunks,embeddings)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        vectorstore.save_local(FAISS_INDEX_PATH)

    return embeddings,vectorstore,all_docs

