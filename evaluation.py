"""
RAG Evaluation Suite
Retrieval : Precision@K, Recall@K, MRR, Hit Rate@K, NDCG@K
Generation: Faithfulness, Answer Relevancy (LLM-as-judge)
"""

from typing import List, Callable, Dict
import math

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return round(hits/k, 4)

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return round(hits/len(relevant), 4)

def mrr(retrieved: List[str], relevant: List[str]) -> float:
    for i, retrieve in enumerate(retrieved):
        if retrieve in relevant:
            return 1/(i+1)
    return 0.0

def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top_k = retrieved[:k]
    hit_rate = sum(1 for doc in top_k if doc in relevant)
    return 1.0 if hit_rate > 0 else 0.0

def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top_k = retrieved[:k]
    dcg = sum(1/math.log2(i+2) for i, doc in enumerate(top_k) if doc in relevant)
    ideal = min(k, len(relevant))
    idcg = sum(1/math.log2(i+2) for i in range(ideal))
    return round(dcg/idcg, 4) if idcg > 0.0 else 0.0

def evaluate_retrieval(retrieved: List[str], relevant: List[str], k: int = 5) -> dict:
    """Run all retrieval metrics at once"""
    return {
        f"precision@{k}" : precision_at_k(retrieved, relevant, k),
        f"recall@{k}"    : recall_at_k(retrieved, relevant, k),
        f"hit_rate@{k}"  : hit_rate_at_k(retrieved, relevant, k),
        f"ndcg@{k}"      : ndcg_at_k(retrieved, relevant, k),
        "mrr"            : mrr(retrieved, relevant),
    }
 
  
# ── Generation Metrics (LLM-as-judge) ────────────────────────


def faithfulness_score(
    answer: str,
    context: List[str],
    llm_fn  
) -> float:
    claims_prompt = f"""Extract all factual claims from this answer as a numbered list.
Answer: {answer}
Claims:"""
    claims_response = llm_fn(claims_prompt)
    claims = [line.strip() for line in claims_response.strip().split('\n') if line.strip()]
    
    if not claims:
        return 0.0

    supported = 0
    context_str = "\n".join(context)
    for claim in claims:
        verification_prompt = f"""Is this claim supported by the context below? Answer only 'YES' or 'NO'.
        Claim: {claim}
        Context: {context_str}
        """
        verification_response = llm_fn(verification_prompt)
        if 'YES' in verification_response.upper():
            supported += 1

    return round(supported/len(claims), 4)

def answer_relevancy_score(
    question: str,
    answer: str,
    llm_fn: Callable[[str], str]
) -> float:
    """
    Measures if answer actually addresses the question.
    Algorithm:
      1. Generate N questions from the answer
      2. Compute cosine similarity between generated and original question
      3. High similarity = answer is relevant to the question
    Simplified: ask LLM to rate relevancy 0-10
    """
    prompt = f"""Rate how well this answer addresses the question on a scale of 0-10.
    Question: {question}
    Answer: {answer}
    Rating (0-10):"""
    response = llm_fn(prompt)
    
    for token in response.split():
        try:
            score = float(token.strip('.,'))
            return round(min(score / 10, 1.0), 4)
        except ValueError:
            continue
    return 0.0
 
def evaluate_generation(
    question: str,
    answer: str,
    context: List[str],
    llm_fn: Callable[[str], str]
) -> Dict:
    return {
        "faithfulness"     : faithfulness_score(answer, context, llm_fn),
        "answer_relevancy" : answer_relevancy_score(question, answer, llm_fn),
    }


# ── Full RAG Evaluation ───────────────────────────────────────
 
def evaluate_rag(
    question: str,
    answer: str,
    retrieved: List[str],
    relevant: List[str],
    llm_fn: Callable[[str], str],
    k: int = 5
) -> Dict:
    """Run complete RAG evaluation — retrieval + generation"""
    return {
        **evaluate_retrieval(retrieved, relevant, k),
        **evaluate_generation(question, answer, retrieved, llm_fn),
    }
 
 
# ── Test ──────────────────────────────────────────────────────
 
if __name__ == "__main__":
    def mock_llm(prompt):
        if "Extract all factual claims" in prompt:
            return "1. attention uses Q K V projections\n2. scaling by sqrt d_k stabilizes training\n3. invented in 1990"
        if "invented in 1990" in prompt:
            return "NO"
        if "Rate how well" in prompt:
            return "8"
        return "YES"
 
    question  = "how does attention mechanism work"
    answer    = "Attention uses Q K V projections. Scaling by sqrt d_k stabilizes training. Invented in 1990."
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    relevant  = ["doc_A", "doc_C", "doc_E"]
    context   = [
        "transformer attention uses query key value projections",
        "scaling by sqrt d_k keeps gradients stable",
    ]
 
    results = evaluate_rag(question, answer, retrieved, relevant, mock_llm, k=5)
    print("RAG Evaluation Results:")
    print("-" * 40)
    for metric, score in results.items():
        print(f"  {metric:<20} {score}")


# ── Production: DeepEval integration ─────────────────────────
# from deepeval import evaluate
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
#
# test_case = LLMTestCase(
#     input=question,
#     actual_output=answer,
#     retrieval_context=context
# )
# metrics = [FaithfulnessMetric(threshold=0.7), AnswerRelevancyMetric(threshold=0.7)]
# evaluate([test_case], metrics)