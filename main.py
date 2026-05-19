from vectorstore import build_or_load
from retriever import llm_load, retrievers_load
from chains import ask_ncert,misconception_detection,answer_with_cite
from app import launch

embeddings,vectorstore,all_docs=build_or_load()
llm=llm_load()
ce,retriever=retrievers_load(all_docs, vectorstore)
launch(answer_with_cite, misconception_detection)

