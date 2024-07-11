from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def calculate_scores(reference, candidate):
    # BLEU score
    bleu_score = sentence_bleu([reference.split()], candidate.split())
    
    # ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference)[0]
    
    return {
        'BLEU': bleu_score,
        'ROUGE-1': rouge_scores['rouge-1']['f'],
        'ROUGE-2': rouge_scores['rouge-2']['f'],
        'ROUGE-L': rouge_scores['rouge-l']['f']
    }

# Questions and ground truths (same as in RAGAS evaluation)
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

# Function to get responses from your basic RAG system
def get_basic_rag_response(question):
    return basic_rag_query_engine.query(question).response

# Function to get responses from your self-querying RAG system
def get_self_querying_rag_response(question):
    return process_query(question)[0].page_content

# Evaluate both systems
for i, question in enumerate(questions):
    print(f"\nQuestion: {question}")
    print(f"Ground Truth: {ground_truths[i]}")
    
    basic_rag_output = get_basic_rag_response(question)
    self_querying_rag_output = get_self_querying_rag_response(question)
    
    basic_scores = calculate_scores(ground_truths[i], basic_rag_output)
    self_querying_scores = calculate_scores(ground_truths[i], self_querying_rag_output)
    
    print("\nBasic RAG:")
    print(f"Output: {basic_rag_output}")
    print("Scores:", basic_scores)
    
    print("\nSelf-Querying RAG:")
    print(f"Output: {self_querying_rag_output}")
    print("Scores:", self_querying_scores)