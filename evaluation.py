from ragas import evaluate
from ragas.metrics import faithfulness,answer_relevancy,context_precision
from datasets import Dataset
from langchain_groq import ChatGroq
from chains import ask_ncert
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EVALUATOR_MODEL,EVAL_EMBEDDING,GROQ_API_KEY

QUESTIONS=[
    "What is the defining characteristic that separates Archaebacteria from Eubacteria?",
    "What is amniocentesis? Why has it been banned for sex determination in India?",
    "At which stage of meiosis does crossing over occur, and what is its significance?",
    "What is the C4 pathway and how does it differ from C3 photosynthesis?",
    "What is the oxygen dissociation curve and what factors cause a right shift?",
    "What is incomplete dominance? Give an example and explain the F2 phenotypic ratio."
]
GROUND_TRUTH=[
    "Archaebacteria have a different cell wall structure...",
    "Amniocentesis is a technique where amniotic fluid...",
    "Crossing over occurs during pachytene stage...",
    "C4 plants have Kranz anatomy...",
    "The oxygen dissociation curve is sigmoid...",
    "Incomplete dominance is when neither allele..."
]
def evaluate_results(QUESTIONS,GROUND_TRUTH,ce,retriever,llm):
    evaluator_llm=ChatGroq(
        model=EVALUATOR_MODEL,temperature=0,
        groq_api_key=GROQ_API_KEY)
    
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name=EVAL_EMBEDDING,
        model_kwargs={"device": "cpu"})
    
    answers=[]
    context=[]
    for q in QUESTIONS:
        ans,cont=ask_ncert(q,retriever,ce,llm)
        answers.append(ans)
        context.append([c.page_content for c in cont])

    dataset=Dataset.from_dict({
        "question":list(QUESTIONS),
        "answer":answers,
        "contexts":context,
        "ground_truth": GROUND_TRUTH
    })
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    return results