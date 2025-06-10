import json
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from logic import initialize_rag, retrieve_relevant_context, nvidia_embed_model

# One-time download
nltk.download('punkt')

# Initialize pipelines
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# ---------------------------
# Metric 1: Self-BLEU
# ---------------------------
def compute_self_bleu(response):
    sentences = sent_tokenize(response)
    if len(sentences) < 2:
        return 0.0
    scores = []
    chencherry = SmoothingFunction().method1
    for i, hypo in enumerate(sentences):
        refs = [word_tokenize(s) for j, s in enumerate(sentences) if j != i]
        hypo_tokens = word_tokenize(hypo)
        score = sentence_bleu(refs, hypo_tokens, weights=(0.2, 0.2, 0.2, 0.2, 0.2), smoothing_function=chencherry)
        scores.append(score)
    return sum(scores) / len(scores)

# ---------------------------
# Metric 2: Relevance Score
# ---------------------------
def relevance_score(query, response):
    embeddings = nvidia_embed_model._get_query_embedding_batch([query, response])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

# ---------------------------
# Metric 3: Groundedness via QA
# ---------------------------
def generate_questions(text):
    try:
        outputs = qg_pipeline(f"generate questions: {text}", max_length=64, do_sample=False)
        return [q['generated_text'] for q in outputs]
    except Exception:
        return []

def check_groundedness(response, context):
    questions = generate_questions(response)
    results = []
    for q in questions:
        try:
            ans = qa_pipeline(question=q, context=context)
            verifiable = ans['score'] >= 0.7 and ans['answer'].strip().lower() not in ['unanswerable', '']
            results.append((q, ans['answer'], ans['score'], verifiable))
        except:
            results.append((q, "", 0.0, False))
    return results

# ---------------------------
# Evaluation Pipeline
# ---------------------------
def evaluate_responses(data):
    results = []
    for entry in data:
        prompt = entry["Prompt"]
        response = entry["Response"]
        context = retrieve_relevant_context(prompt)

        # Metric calculations
        sbleu = compute_self_bleu(response)
        rel = relevance_score(prompt, response)
        grounding_results = check_groundedness(response, context)
        grounding_scores = [r[2] for r in grounding_results]
        verifiable_flags = [r[3] for r in grounding_results]
        grounded_score = max(grounding_scores) if grounding_scores else 0.0
        verifiable = any(verifiable_flags)

        results.append({
            "Prompt": prompt,
            "Response": response,
            "Self-BLEU": round(sbleu, 4),
            "Relevance Score": round(rel, 4),
            "Groundedness Score": round(grounded_score, 4),
            "Verifiable?": verifiable
        })
    return pd.DataFrame(results)

# ---------------------------
# Main Entry
# ---------------------------
if __name__ == "__main__":
    # Load input
    with open("test_cases.json", "r") as f:
        test_data = json.load(f)

    # Build query engine from system_data/
    print("Initializing RAG database...")
    status = initialize_rag("./system_data")
    print(status)

    # Evaluate
    print("Evaluating responses...")
    df = evaluate_responses(test_data)
    df.to_csv("rootwise_evaluation_results.csv", index=False)
    print("✅ Saved results to rootwise_evaluation_results.csv")
