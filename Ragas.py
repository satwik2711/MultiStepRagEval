from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall,
)

# Prepare your dataset
questions = [
    "Who is leading the 2024 Formula 1 championship?",
    "Which team has the most constructor points in 2024?",
    "How many podium finishes does Lewis Hamilton have in 2024?",
    "What is Max Verstappen's win count for the 2024 season?",
    "Which driver has shown the most improvement from the 2023 season?"
]

ground_truths = [
    "Max Verstappen is leading the 2024 Formula 1 championship.",
    "Red Bull Racing has the most constructor points in 2024.",
    "Lewis Hamilton has 8 podium finishes in 2024.",
    "Max Verstappen has 10 wins in the 2024 season.",
    "Lewis Hamilton has shown significant improvement in his new team Ferrari."
]

# Create a dataset
dataset = Dataset.from_dict({
    "question": questions,
    "ground_truth": ground_truths
})

# Function to get responses from your basic RAG system
def get_basic_rag_response(question):
    return basic_rag_query_engine.query(question).response

# Function to get responses from your self-querying RAG system
def get_self_querying_rag_response(question):
    return process_query(question)[0].page_content

# Evaluate basic RAG
basic_rag_results = evaluate(
    dataset,
    get_basic_rag_response,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]
)

print("Basic RAG Results:")
print(basic_rag_results)

# Evaluate self-querying RAG
self_querying_rag_results = evaluate(
    dataset,
    get_self_querying_rag_response,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]
)

print("\nSelf-Querying RAG Results:")
print(self_querying_rag_results)