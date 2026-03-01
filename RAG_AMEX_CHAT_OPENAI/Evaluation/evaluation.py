import math

from jsonschema.exceptions import relevance
from pydantic import BaseModel, Field

from RAG_AMEX_CHAT_OPENAI.Evaluation.test import load_tests
from RAG_AMEX_CHAT_OPENAI.answer import fetch_context

class RetrivalEvaluation(BaseModel):
    mrr : float=Field(description="How many times my keyword appeared in page content with ranking based on retrieval ranking")
    ndcg : float=Field(description="How many times my keyword appeared in page content in relation to ideal keyword sorting as 1s and 0s")
    keyword_found: int=Field(description="how many keywords found")
    total_keywords: int=Field(description="Expected keywords")
    keywords_coverage: float=Field(description="how many words were found?")


def calculate_mrr(keyword, retrieved_docs):

    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword.lower() in doc.page_content.lower():
            return 1.0/rank
    return 0.0
def calculate_dcg(relevance, k):

    dcg = 0.0
    for i in range(min(len(relevance), k)):
        dcg += relevance[i] / math.log2(i + 2)
    return dcg

def calculate_ndcg(retrieved_docs, keyword, k):

    relevance = [1 if keyword.lower() in doc.page_content.lower() else 0 for doc in retrieved_docs]
    dcg = calculate_dcg(relevance, k)
    ideal_relevance = sorted(relevance, reverse= True)
    idcg = calculate_dcg(ideal_relevance, k)
    return dcg/idcg if idcg > 0 else 0.0

def evaluate_retrieval(test, k = 10):
    retrieved_docs = fetch_context(test.question)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    average_mrr = sum(mrr_scores)/len(mrr_scores) if mrr_scores else 0.0
    ndcg_scores = [calculate_ndcg(retrieved_docs, keyword,k) for keyword in test.keywords]
    average_ndcg = sum(ndcg_scores)/len(ndcg_scores)
    keyword_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keywords_coverage = (keyword_found/total_keywords*100) if total_keywords else 0.0
    return RetrivalEvaluation(mrr = average_mrr, ndcg=average_ndcg, keyword_found= keyword_found, total_keywords= total_keywords, keywords_coverage=keywords_coverage)



def eval_all_test():
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress