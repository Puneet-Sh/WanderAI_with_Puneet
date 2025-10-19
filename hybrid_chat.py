# hybrid_chat.py

import json
import hashlib
import os
import asyncio
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# -----------------------------
# Puneet's changes 
# -----------------------------
# Embedding cache for faster repeated queries
cache_file = "embedding_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        try:
            embedding_cache = json.load(f)
        except json.JSONDecodeError:
            embedding_cache = {}
else:
    embedding_cache = {}

def get_cached_embedding(text, client):
    """Return cached embedding if available, else compute and store it."""
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in embedding_cache:
        return embedding_cache[key]
    else:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        emb = resp.data[0].embedding
        embedding_cache[key] = emb
        with open(cache_file, "w") as f:
            json.dump(embedding_cache, f)
        return emb


# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string (with caching)."""
    return get_cached_embedding(text, client)


def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print("DEBUG: Pinecone top 5 results:")
    print(len(res["matches"]))
    return res["matches"]


def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
    print("DEBUG: Graph facts:")
    print(len(facts))
    return facts


# -----------------------------
# Puneet's changes 
# -----------------------------
# Added summarizer for Pinecone + Neo4j results
def search_summary(pinecone_matches, graph_facts):
    """Summarize top Pinecone and Neo4j results for prompt clarity."""
    summary = ["Context Summary:"]
    summary.append("\nTop Pinecone Matches:")
    for m in pinecone_matches[:3]:
        summary.append(f"- {m['metadata'].get('name', '')}: {m['metadata'].get('description', '')[:100]}...")

    summary.append("\nConnected Graph Nodes:")
    for f in graph_facts[:3]:
        summary.append(f"- {f['target_name']} ({f['rel']}) → {f['target_desc'][:80]}...")

    return "\n".join(summary)


# -----------------------------
# Puneet's changes 
# -----------------------------
# Improved prompt engineering with reasoning steps
def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a structured prompt combining semantic + graph context with reasoning."""
    context_summary = search_summary(pinecone_matches, graph_facts)

    system_prompt = (
        "You are an expert travel planner AI specializing in Vietnam and other Asian destinations. "
        "You have access to a semantic vector database and a Neo4j travel knowledge graph. "
        "Use both to craft clear, day-wise, or step-wise itineraries or answers. "
        "Cite node ids when useful and avoid repetition."
    )

    user_prompt = f"""
User query: {user_query}

{context_summary}

Think step by step:
1️⃣ Understand the intent behind the query.
2️⃣ Select relevant places and activities from the context.
3️⃣ Build a creative yet realistic travel plan or answer.
4️⃣ Add 1–2 personalized suggestions at the end.

Now write your final answer below:
"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return prompt


def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.4
    )
    return resp.choices[0].message.content


# -----------------------------
# Puneet's changes 
# -----------------------------
# async structure for parallel tasks (future improvement)
async def async_fetch(query):
    """Async placeholder for future parallel Pinecone + Neo4j calls."""
    loop = asyncio.get_event_loop()
    pinecone_task = loop.run_in_executor(None, pinecone_query, query)
    pinecone_res = await pinecone_task
    return pinecone_res


# -----------------------------
# Interactive chat
# -----------------------------
def interactive_chat():
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break

        matches = pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")


if __name__ == "__main__":
    interactive_chat()
