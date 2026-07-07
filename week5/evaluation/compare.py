from evaluation.test import load_tests
from evaluation.eval import evaluate_retrieval, evaluate_answer, judge_answer
from implementation.answer import answer_question
from implementation.agentic_answer import agentic_answer_question


def run_comparison(num_tests: int = 10):
    """Run both RAG modes on the same tests and collect average scores."""
    tests = load_tests()[:num_tests]

    simple_scores = {"accuracy": [], "completeness": [], "relevance": []}
    agentic_scores = {"accuracy": [], "completeness": [], "relevance": []}

    for test in tests:
        simple_eval, _, _ = evaluate_answer(test)
        simple_scores["accuracy"].append(simple_eval.accuracy)
        simple_scores["completeness"].append(simple_eval.completeness)
        simple_scores["relevance"].append(simple_eval.relevance)

        # Temporarily swap the answer function for agentic evaluation
        agentic_answer, agentic_docs = agentic_answer_question(test.question)
        agentic_eval = judge_answer(test, agentic_answer)
        agentic_scores["accuracy"].append(agentic_eval.accuracy)
        agentic_scores["completeness"].append(agentic_eval.completeness)
        agentic_scores["relevance"].append(agentic_eval.relevance)

    def avg(scores):
        return {k: sum(v) / len(v) for k, v in scores.items()}

    return avg(simple_scores), avg(agentic_scores)
