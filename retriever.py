from langchain_cerebras import ChatCerebras
from config import LLM_MODEL,CEREBRAS_API_KEY,LLM_TEMPERATURE,CROSS_ENCODER_MODEL,FAISS_K,BM25_K,ENSEMBLE_WEIGHTS,LLM_MAX_TOKENS
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def llm_load():
    llm=ChatCerebras(
        model=LLM_MODEL,
        api_key=CEREBRAS_API_KEY,temperature=LLM_TEMPERATURE,max_tokens=LLM_MAX_TOKENS
    )
    return llm

def retrievers_load(all_docs,vectorstore):
    bm25=BM25Retriever.from_documents(all_docs)
    bm25.k=BM25_K

    ce=CrossEncoder(CROSS_ENCODER_MODEL,device='cpu')
    faiss=vectorstore.as_retriever(search_kwargs={'k':FAISS_K})
    retriever=EnsembleRetriever(retrievers=[bm25,faiss],weights=ENSEMBLE_WEIGHTS)
    return ce,retriever