# Chapter 6 — Modern AI Systems: RAG, Agents, and Glue Code

## The Crux
Models alone are useless. Real AI systems are models + data pipelines + retrieval + guardrails + monitoring + glue code. This chapter is about engineering AI into production, not just training models.

## Why Models Alone Are Useless

You've trained a great model. Congratulations. Now what?

**Reality**:
- The model needs to integrate with existing systems (databases, APIs, user interfaces)
- Users don't send perfectly formatted inputs
- The model drifts as the world changes
- You need to monitor failures, log predictions, retrain periodically
- You need to handle errors gracefully (what if the API is down?)

**The model is 10% of the system.** The other 90% is infrastructure.

## RAG: Retrieval-Augmented Generation

LLMs hallucinate because they rely on memorized training data. What if we give them access to external knowledge?

### The Idea

Instead of asking the LLM to answer directly:
1. **Retrieve** relevant documents from a database
2. **Augment** the prompt with retrieved information
3. **Generate** the answer based on retrieved context

**Example**:
- User: "What's the return policy?"
- System retrieves: Company policy doc mentioning "30-day returns"
- Prompt: "Based on this policy: [retrieved text], answer: What's the return policy?"
- LLM: "We offer 30-day returns."

### Why It Works

The LLM doesn't need to memorize every fact. It just needs to read context and extract answers—something LLMs are good at.

### Architecture

1. **Document store**: Database of knowledge (vector database, Elasticsearch, etc.)
2. **Embedding model**: Convert queries and documents to vectors
3. **Retrieval**: Find top-k most similar documents to the query (cosine similarity)
4. **LLM**: Generate answer given query + retrieved docs

### When to Use RAG vs Fine-Tuning

**RAG**:
- Knowledge changes frequently (e.g., product docs updated weekly)
- You need to cite sources
- You have limited GPU resources

**Fine-tuning**:
- Knowledge is stable
- You want the model to internalize a style or domain-specific reasoning
- You have labeled data and compute

Often, you use both: fine-tune for style/domain, RAG for up-to-date facts.

## Agents: When LLMs Take Actions

An agent is an LLM that can:
1. Use tools (search, calculator, APIs)
2. Plan multi-step tasks
3. Reflect on its actions

### The Basic Loop

```
while not done:
    observation = get_current_state()
    thought = llm("Given [observation], what should I do?")
    action = parse_action(thought)
    result = execute_action(action)
    if is_goal_achieved(result):
        done = True
```

### Example: Research Agent

**Task**: "Find the GDP of France in 2022."

**Agent steps**:
1. Thought: "I need to search for France GDP 2022."
2. Action: `search("France GDP 2022")`
3. Observation: Search results mention $2.78 trillion.
4. Thought: "I found the answer."
5. Action: `return_answer("$2.78 trillion")`

### Why Agents Are Hard

**Problem #1: LLMs make mistakes**
Agents amplify errors. If the LLM calls the wrong API, takes the wrong action, or misinterprets results, the whole plan fails.

**Problem #2: Infinite loops**
Without careful design, agents can loop: search → no result → search again → repeat forever.

**Problem #3: Cost**
Each step requires an LLM call. Complex tasks can cost dollars in API fees.

**Problem #4: Evaluation**
How do you test an agent? Unit tests don't cover emergent multi-step behavior. You need integration tests, but tasks are open-ended.

### When Agents Work

- **Narrow domains**: Customer support, data analysis scripts, code generation.
- **Human-in-the-loop**: Agent suggests, human approves.
- **Guardrails**: Constrain action space. Don't let the agent run arbitrary shell commands.

## War Story: An Agent That Took the Wrong Action

**The Setup**: A company built an agent to automate customer refunds. It had access to:
- Customer database
- Transaction history
- Refund API

**The Task**: "Process refunds for customers who received damaged items."

**The Incident**: The agent ran. Thousands of refunds were issued. Then accounting noticed: refunds were issued to customers who *hadn't* requested them.

**The Investigation**: The agent's logic:
1. Search for "damaged items" in customer messages.
2. For each match, call refund API.

**The Bug**: Some messages said "I didn't receive damaged items, everything was fine." The agent searched for the keyword "damaged" and issued refunds.

**The Lesson**: LLMs don't reason perfectly. They pattern-match. Agents need:
- Robust parsing and validation
- Confirmation steps before irreversible actions
- Human oversight for high-stakes decisions

## Evaluation Is Harder Than Training

You can train a model overnight. Evaluating it properly takes weeks.

### Why Evaluation Is Hard

**Problem #1: Metrics lie**
Accuracy, F1, AUC—all are proxies. They don't capture user satisfaction, edge cases, or silent failures.

**Problem #2: Test sets drift**
Your test set is from last year. User behavior changed. Your metrics don't reflect production reality.

**Problem #3: Open-ended tasks**
How do you evaluate "write a creative story"? No single correct answer. Human evaluation is expensive and subjective.

**Problem #4: Adversarial robustness**
Your model works on random test examples. What about adversarial ones? Users will try to break it.

### How to Evaluate Properly

**1. Holdout sets that match production distribution**
Don't just split randomly. Split by time, geography, user type—whatever matches how you'll deploy.

**2. A/B testing**
Deploy to a small percentage of users. Measure real metrics (engagement, revenue, errors).

**3. Human evaluation**
Sample predictions, have humans rate quality. Expensive but necessary for subjective tasks.

**4. Monitoring in production**
Track model predictions, user feedback, error rates. Set up alerts for anomalies.

**5. Adversarial testing**
Red-team your model. Try to make it fail. Fix failure modes.

## Things That Will Confuse You

### "My model has 95% accuracy, it's production-ready"
Accuracy on what distribution? Did you test edge cases? Can users adversarially break it?

### "RAG fixes hallucinations"
It reduces them, but if retrieval fails (no relevant docs), the LLM still hallucinates. You need fallback logic.

### "Agents are autonomous"
In production, agents are semi-autonomous. You constrain actions, log everything, and often require human confirmation.

### "Fine-tuning is better than prompting"
Depends. Prompting is faster and cheaper. Fine-tuning is better if you have lots of task-specific data and need consistent behavior.

## Common Traps

**Trap #1: Over-relying on LLMs**
Use rule-based systems for deterministic tasks. LLMs for ambiguous, creative, or language-heavy tasks. Don't use an LLM where a regex suffices.

**Trap #2: Not versioning prompts**
Prompts are code. Version them. Track which prompt version produced which outputs.

**Trap #3: Ignoring latency**
Retrieval + LLM generation can take seconds. Users expect milliseconds. Cache aggressively.

**Trap #4: No fallback logic**
What if the API times out? The LLM returns garbage? The database is down? Always have a fallback.

## Production Reality Check

Real AI systems:

- **Are mostly glue code**: 70% data pipelines, API integrations, error handling. 20% monitoring and retraining. 10% model training.
- **Require monitoring**: Model drift, data drift, latency, errors—all need dashboards and alerts.
- **Degrade gracefully**: If the model fails, fall back to rules or human escalation.
- **Cost real money**: LLM API calls, GPU inference, storage, bandwidth. Optimize aggressively.

## Build This Mini Project

**Goal**: Build a simple RAG system.

**Task**: Create a question-answering system over your own documents.

Here's a complete, runnable RAG implementation:

```python
import numpy as np
from typing import List, Tuple
import os

# For embeddings, we'll use sentence-transformers (free, runs locally)
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

print("="*70)
print("BUILDING A RAG SYSTEM FROM SCRATCH")
print("="*70)

# =============================================================================
# Step 1: Create Sample Documents (Knowledge Base)
# =============================================================================
print("\n📚 Step 1: Creating knowledge base...")

# Simulate a company's documentation
documents = {
    "return_policy": """
    Return Policy:
    - All items can be returned within 30 days of purchase.
    - Items must be in original packaging and unused condition.
    - Refunds are processed within 5-7 business days.
    - Digital products cannot be returned once downloaded.
    - Shipping costs for returns are the customer's responsibility.
    """,

    "shipping_info": """
    Shipping Information:
    - Standard shipping takes 5-7 business days.
    - Express shipping takes 2-3 business days.
    - Free shipping on orders over $50.
    - We ship to all 50 US states and Canada.
    - International shipping is not currently available.
    - Track your order using the tracking number in your confirmation email.
    """,

    "product_warranty": """
    Product Warranty:
    - All electronics come with a 1-year manufacturer warranty.
    - Warranty covers defects in materials and workmanship.
    - Warranty does not cover accidental damage or misuse.
    - To claim warranty, contact support with your order number.
    - Extended warranty available for purchase at checkout.
    """,

    "account_help": """
    Account Help:
    - Reset your password using the "Forgot Password" link.
    - Update billing information in Account Settings.
    - View order history under "My Orders".
    - Contact support at support@example.com.
    - Business hours: Monday-Friday 9am-5pm EST.
    """,

    "payment_methods": """
    Payment Methods:
    - We accept Visa, Mastercard, American Express, and Discover.
    - PayPal and Apple Pay are also accepted.
    - Gift cards can be purchased and redeemed online.
    - Payment is processed securely using SSL encryption.
    - Subscriptions can be managed in Account Settings.
    """
}

# =============================================================================
# Step 2: Chunk Documents
# =============================================================================
print("📄 Step 2: Chunking documents...")

def chunk_document(doc_name: str, text: str, chunk_size: int = 200) -> List[dict]:
    """Split document into overlapping chunks"""
    sentences = text.strip().split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                'source': doc_name,
                'text': ' '.join(current_chunk),
                'chunk_id': len(chunks)
            })
            # Keep last sentence for overlap
            current_chunk = current_chunk[-1:] if current_chunk else []
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    # Add remaining chunk
    if current_chunk:
        chunks.append({
            'source': doc_name,
            'text': ' '.join(current_chunk),
            'chunk_id': len(chunks)
        })

    return chunks

# Chunk all documents
all_chunks = []
for doc_name, doc_text in documents.items():
    chunks = chunk_document(doc_name, doc_text)
    all_chunks.extend(chunks)

print(f"   Created {len(all_chunks)} chunks from {len(documents)} documents")

# =============================================================================
# Step 3: Create Embeddings
# =============================================================================
print("🔢 Step 3: Creating embeddings...")

# Load a small, efficient embedding model
# This runs locally and is free!
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all chunks
chunk_texts = [chunk['text'] for chunk in all_chunks]
chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)

print(f"   Embedding shape: {chunk_embeddings.shape}")
print(f"   (Each chunk is a {chunk_embeddings.shape[1]}-dimensional vector)")

# =============================================================================
# Step 4: Build Vector Search
# =============================================================================
print("🔍 Step 4: Building vector search...")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_documents(query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
    """Find most relevant chunks for a query"""
    # Embed the query
    query_embedding = embedding_model.encode([query])[0]

    # Calculate similarity to all chunks
    similarities = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, chunk_emb)
        similarities.append((all_chunks[i], sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

print("   Vector search ready!")

# =============================================================================
# Step 5: RAG Pipeline
# =============================================================================
print("🤖 Step 5: Building RAG pipeline...")

def rag_answer(query: str, top_k: int = 3, verbose: bool = True) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build prompt with context
    3. Generate answer (simulated here - in production, use an LLM API)
    """

    # Step 1: Retrieve
    results = search_documents(query, top_k=top_k)

    if verbose:
        print(f"\n📋 Query: '{query}'")
        print(f"\n📚 Retrieved {len(results)} relevant chunks:")
        for chunk, score in results:
            print(f"   [{score:.3f}] {chunk['source']}: {chunk['text'][:80]}...")

    # Step 2: Build context
    context_parts = []
    for chunk, score in results:
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    # Step 3: Build prompt
    prompt = f"""Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {query}

Answer:"""

    if verbose:
        print(f"\n📝 Generated prompt ({len(prompt)} chars)")

    # In production, you would call an LLM API here:
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content

    # For this demo, we'll simulate based on context
    answer = simulate_llm_response(query, results)

    return answer

def simulate_llm_response(query: str, results: List[Tuple[dict, float]]) -> str:
    """Simulate an LLM response based on retrieved context"""
    query_lower = query.lower()

    # Check if we have relevant results (similarity > 0.3)
    if not results or results[0][1] < 0.3:
        return "I don't have information about that in my knowledge base."

    # Extract key information from top result
    top_chunk = results[0][0]
    source = top_chunk['source']
    text = top_chunk['text']

    # Generate response based on query type
    if 'return' in query_lower:
        return "Based on the return policy: Items can be returned within 30 days of purchase. They must be in original packaging and unused condition. Refunds are processed within 5-7 business days. Note that digital products cannot be returned once downloaded."

    elif 'ship' in query_lower:
        return "Based on shipping information: Standard shipping takes 5-7 business days, and express shipping takes 2-3 business days. Free shipping is available on orders over $50. We ship to all 50 US states and Canada."

    elif 'warranty' in query_lower:
        return "Based on the warranty policy: All electronics come with a 1-year manufacturer warranty covering defects in materials and workmanship. Accidental damage is not covered. Contact support with your order number to claim warranty."

    elif 'password' in query_lower or 'account' in query_lower:
        return "To reset your password, use the 'Forgot Password' link on the login page. For other account issues, you can update settings in Account Settings or contact support at support@example.com."

    elif 'payment' in query_lower or 'pay' in query_lower:
        return "We accept Visa, Mastercard, American Express, Discover, PayPal, and Apple Pay. All payments are processed securely using SSL encryption."

    else:
        return f"Based on {source}: {text[:200]}..."

print("   RAG pipeline ready!")

# =============================================================================
# Step 6: Test the System
# =============================================================================
print("\n" + "="*70)
print("TESTING THE RAG SYSTEM")
print("="*70)

# Test queries
test_queries = [
    "What is your return policy?",
    "How long does shipping take?",
    "Do you offer warranty on products?",
    "How do I reset my password?",
    "What payment methods do you accept?",
    "Do you ship internationally?",  # Answer is in docs
    "What's the weather like today?",  # Not in docs - should fail gracefully
]

print("\n" + "-"*70)
for query in test_queries:
    answer = rag_answer(query, top_k=2, verbose=False)
    print(f"\n❓ Q: {query}")
    print(f"💬 A: {answer}")
print("\n" + "-"*70)

# =============================================================================
# Step 7: Demonstrate Retrieval Quality
# =============================================================================
print("\n" + "="*70)
print("RETRIEVAL QUALITY ANALYSIS")
print("="*70)

query = "How do I return an item?"
results = search_documents(query, top_k=5)

print(f"\nQuery: '{query}'")
print("\nTop 5 results by similarity score:")
for i, (chunk, score) in enumerate(results, 1):
    print(f"\n{i}. Score: {score:.4f}")
    print(f"   Source: {chunk['source']}")
    print(f"   Text: {chunk['text'][:100]}...")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("RAG SYSTEM SUMMARY")
print("="*70)
print(f"""
COMPONENTS BUILT:
1. Document Store: {len(documents)} documents, {len(all_chunks)} chunks
2. Embedding Model: all-MiniLM-L6-v2 ({chunk_embeddings.shape[1]}D vectors)
3. Vector Search: Cosine similarity retrieval
4. RAG Pipeline: Retrieve → Context → Generate

KEY INSIGHTS:
- Retrieval quality determines answer quality
- Chunk size affects precision vs recall
- Embedding model choice matters
- Always have fallback for no-match queries

IN PRODUCTION, ADD:
- Persistent vector database (Pinecone, Weaviate, FAISS)
- Real LLM for generation (GPT-4, Claude)
- Caching for repeated queries
- Monitoring for retrieval quality
- Reranking for better precision
""")
print("="*70)
```

**Expected Output:**
```
======================================================================
BUILDING A RAG SYSTEM FROM SCRATCH
======================================================================

📚 Step 1: Creating knowledge base...
📄 Step 2: Chunking documents...
   Created 12 chunks from 5 documents
🔢 Step 3: Creating embeddings...
   Embedding shape: (12, 384)
   (Each chunk is a 384-dimensional vector)
🔍 Step 4: Building vector search...
   Vector search ready!
🤖 Step 5: Building RAG pipeline...
   RAG pipeline ready!

======================================================================
TESTING THE RAG SYSTEM
======================================================================

----------------------------------------------------------------------

❓ Q: What is your return policy?
💬 A: Based on the return policy: Items can be returned within 30 days
      of purchase. They must be in original packaging and unused
      condition. Refunds are processed within 5-7 business days.

❓ Q: How long does shipping take?
💬 A: Based on shipping information: Standard shipping takes 5-7
      business days, and express shipping takes 2-3 business days.

❓ Q: What's the weather like today?
💬 A: I don't have information about that in my knowledge base.

----------------------------------------------------------------------

======================================================================
RETRIEVAL QUALITY ANALYSIS
======================================================================

Query: 'How do I return an item?'

Top 5 results by similarity score:

1. Score: 0.7234
   Source: return_policy
   Text: Return Policy: - All items can be returned within 30 days...

2. Score: 0.4521
   Source: shipping_info
   Text: Shipping Information: - Standard shipping takes 5-7 business...
```

**Key Insights**:

1. **Retrieval is Everything**: The LLM can only use what you retrieve. Bad retrieval = bad answers.
2. **Graceful Failure**: When retrieval finds nothing relevant, say "I don't know" instead of hallucinating.
3. **Chunk Size Matters**: Too small = lose context. Too large = noise in retrieval.
4. **Embedding Choice**: Different models have different strengths (semantic vs lexical matching).

**Key Insight**: RAG grounds LLMs in external knowledge. Retrieval quality determines answer quality. If retrieval fails, the LLM has no signal.

---

