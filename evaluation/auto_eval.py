import os
import json
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('punkt_tab')


try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import pipeline
except ImportError as e:
    raise ImportError(f"missing dependency: {e.name}")


smooth_fn = SmoothingFunction().method1
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


def compute_self_bleu(response: str) -> float:
    sentences = sent_tokenize(response)
    if len(sentences) < 2:
        return 0.0
    scores = []
    for i, hypo in enumerate(sentences):
        refs = [word_tokenize(s) for j, s in enumerate(sentences) if j != i]
        hypo_tokens = word_tokenize(hypo)
        score = sentence_bleu(refs, hypo_tokens, weights=(0.2,) * 5, smoothing_function=smooth_fn)
        scores.append(score)
    return round(sum(scores) / len(scores), 4)


def compute_relevance(query: str, response: str) -> float:
    embeddings = embed_model.encode([query, response])
    return round(float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]), 4)


def generate_questions(text: str):
    try:
        out = qg_pipeline(f"generate questions: {text}", max_length=64, do_sample=False)
        return [o['generated_text'] for o in out]
    except Exception:
        return []


def answer_question(question: str, context: str):
    try:
        result = qa_pipeline(question=question, context=context)
        answer = result.get('answer', '')
        score = result.get('score', 0.0)
        return answer, round(score, 4), (answer == '' or score < 0.2)
    except Exception:
        return '', 0.0, True


def groundedness_check(response: str, rag_excerpt: str):
    sentences = sent_tokenize(response)
    if not sentences or not rag_excerpt.strip():
        return False, 0.0
    results = []
    for sent in sentences:
        questions = generate_questions(sent)
        if not questions:
            continue
        max_score = 0.0
        verifiable = False
        for q in questions:
            ans, score, _ = answer_question(q, rag_excerpt)
            if score >= 0.7:
                verifiable = True
            max_score = max(max_score, score)
        results.append((verifiable, max_score))
    if not results:
        return False, 0.0
    overall_verifiable = all(r[0] for r in results)
    avg_score = round(sum(r[1] for r in results) / len(results), 4)
    return overall_verifiable, avg_score


def evaluate_json_file(json_path: str, output_csv: str, rag_excerpt_default: str = ""):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return

    output_rows = []
    for i, item in enumerate(data):
        prompt = item.get("Prompt", "")
        response = item.get("Response", "")
        rag_excerpt = item.get("rag_excerpt", rag_excerpt_default)

        try:
            self_bleu = compute_self_bleu(response)
            relevance = compute_relevance(prompt, response)
            verifiable, grounded_score = groundedness_check(response, rag_excerpt)

            output_rows.append({
                "Prompt": prompt,
                "Response": response,
                "Self-BLEU": self_bleu,
                "Relevance Score": relevance,
                "Groundedness Score": grounded_score,
                "Verifiable?": verifiable
            })

            print(f"[{i+1}/{len(data)}] Done: BLEU={self_bleu}, Rel={relevance}, Gnd={grounded_score}")

        except Exception as e:
            print(f"[{i+1}] Error evaluating entry: {e}")
            continue

    pd.DataFrame(output_rows).to_csv(output_csv, index=False)
    print(f"\n Results saved to: {output_csv}")


if __name__ == "__main__":
    evaluate_json_file(
        json_path="test_cases.json",
        output_csv="evaluation_results.csv",
        rag_excerpt_default="Kale stems, squash skin, and citrus peels can be reused in broths, pestos, or fermented condiments. Avoid soy if allergic. Cooking seasonally helps hydration and nourishment."
    )
