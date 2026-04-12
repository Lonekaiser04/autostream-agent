"""
rag.py – Retrieval-Augmented Generation pipeline for AutoStream Agent.

Loads knowledge.json and performs simple keyword-based retrieval.
No external vector DB needed — fully local and dependency-light.
"""

import json
import os
from typing import List

# Load knowledge base from local JSON file
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")

with open(KB_PATH, "r") as f:
    KNOWLEDGE_BASE = json.load(f)


# Flatten KB into retrievable chunks

def _build_chunks() -> List[dict]:
    """Convert the JSON knowledge base into flat text chunks with tags."""
    chunks = []

    # Company overview
    chunks.append({
        "tags": ["company", "about", "autostream", "what"],
        "text": f"AutoStream: {KNOWLEDGE_BASE['description']}"
    })

    # Plans
    for plan_key, plan in KNOWLEDGE_BASE["plans"].items():
        feature_list = ", ".join(plan["features"])
        chunks.append({
            "tags": ["plan", "pricing", "price", plan_key, "cost", "money", "how much"],
            "text": (
                f"{plan['name']}: {plan['price']}. "
                f"Features: {feature_list}."
            )
        })

    # Policies
    for policy_key, policy_text in KNOWLEDGE_BASE["policies"].items():
        chunks.append({
            "tags": ["policy", "policies", policy_key, "refund", "support",
                     "cancel", "trial", "guarantee"],
            "text": f"Policy – {policy_key}: {policy_text}"
        })

    # FAQs
    for faq in KNOWLEDGE_BASE["faq"]:
        chunks.append({
            "tags": ["faq", "question"],
            "text": f"Q: {faq['question']}\nA: {faq['answer']}"
        })

    return chunks


CHUNKS = _build_chunks()


# Retrieval function
def retrieve_knowledge(query: str, top_k: int = 4) -> str:
    """
    Score each chunk by keyword overlap with the query.
    Returns a concatenated string of the top-k most relevant chunks.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored = []
    for chunk in CHUNKS:
        score = 0
        # Tag match (high weight)
        for tag in chunk["tags"]:
            if tag in query_lower:
                score += 3
        # Word overlap in chunk text (low weight)
        chunk_words = set(chunk["text"].lower().split())
        score += len(query_words & chunk_words)
        scored.append((score, chunk["text"]))

    # Sort descending, take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [text for score, text in scored[:top_k] if score > 0]

    if not top_chunks:
        # Return full KB summary as fallback
        top_chunks = [c["text"] for c in CHUNKS[:top_k]]

    return "\n\n".join(top_chunks)


# Debug helper
if __name__ == "__main__":
    test_queries = [
        "What is the price of the Pro plan?",
        "Do you offer refunds?",
        "What platforms does AutoStream support?",
        "Tell me about the basic plan features",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Retrieved context:")
        print(retrieve_knowledge(q))
        print("-" * 40)