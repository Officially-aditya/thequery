# 9. ASSESSMENTS & QUIZZES

(Quizzes test retention, not understanding. If you can pass these without looking up answers, great - you remember the material. If not, that's fine too. What matters is whether you can build working systems, not whether you memorized which embedding model is cheaper. Use these to identify gaps, then go back and reread those sections.)

## Module 1 Quiz: Foundations

1. **What is the main advantage of embeddings over keyword search?**
   - a) Faster processing
   - b) Captures semantic similarity
   - c) Requires less storage
   - d) No API needed

2. **In a knowledge graph, what represents the relationship between two entities?**
   - a) Node
   - b) Edge
   - c) Property
   - d) Label

3. **Which embedding model is most cost-effective for general use?**
   - a) text-embedding-3-large
   - b) text-embedding-ada-002
   - c) text-embedding-3-small
   - d) GPT-4

**Answer Key**: 1-b, 2-b, 3-c

## Module 2 Quiz: RAG Engineering

1. **Why is chunk overlap important?**
   - a) Increases total chunks
   - b) Prevents context loss at boundaries
   - c) Improves embedding quality
   - d) Reduces API costs

2. **What does a reranker do?**
   - a) Re-sorts initial retrieval results for better precision
   - b) Generates new embeddings
   - c) Rewrites user queries
   - d) Ranks LLM responses

3. **Which retrieval method combines exact matching with semantic search?**
   - a) BM25 only
   - b) Dense retrieval
   - c) Hybrid retrieval
   - d) Keyword search

**Answer Key**: 1-b, 2-a, 3-c

## Module 3 Quiz: Knowledge Graphs

1. **What Cypher query finds all people working at "Acme"?**
   - a) `FIND (p)->(c) WHERE c.name = "Acme"`
   - b) `MATCH (p:Person)-[:WORKS_FOR]->(c:Company {name: "Acme"}) RETURN p`
   - c) `SELECT * FROM Person WHERE company = "Acme"`
   - d) `GET Person WITH Company = "Acme"`

2. **What is entity linking?**
   - a) Creating relationships between entities
   - b) Resolving different mentions to the same entity
   - c) Extracting entities from text
   - d) Storing entities in a database

3. **What does a 2-hop neighborhood query return?**
   - a) Entities exactly 2 steps away
   - b) Entities within 1-2 steps
   - c) 2 random neighbors
   - d) Second-degree connections only

**Answer Key**: 1-b, 2-b, 3-b

## Module 4 Quiz: Hybrid Systems

1. **When should you route a query to the knowledge graph?**
   - a) Always
   - b) For analytical questions requiring summarization
   - c) For factual and relationship queries
   - d) Never, always use RAG

2. **What is context fusion?**
   - a) Merging multiple documents
   - b) Combining KG facts with RAG documents into unified context
   - c) Fusing query and answer
   - d) Merging embeddings

3. **What advantage does hybrid RAG+KG have over plain RAG?**
   - a) Faster query speed
   - b) Multi-hop reasoning and structured relationships
   - c) Lower cost
   - d) Easier implementation

**Answer Key**: 1-c, 2-b, 3-b

## Coding Assignments

### Assignment 1: Build a Basic RAG System
**Task**: Implement a RAG system that answers questions about 3 PDF documents.

**Requirements**:
- Extract and chunk PDFs
- Store in vector DB
- Implement query function
- Add citations

**Submission**: GitHub repo with code + README

---

### Assignment 2: Extract Knowledge Graph from News Articles
**Task**: Extract entities and relationships from 20 news articles and build a Neo4j graph.

**Requirements**:
- Use LLM for extraction
- Entity deduplication
- Load into Neo4j
- Run 5 example Cypher queries

**Submission**: Code + Graph visualization screenshot + Query examples

---

### Assignment 3: Build a Hybrid Query System
**Task**: Implement query routing that decides between KG, RAG, or hybrid.

**Requirements**:
- Query classifier (LLM-based)
- Three retrieval strategies
- Test on 20 varied questions
- Compare performance

**Submission**: Code + evaluation results (accuracy, latency)

---

## Final Interview-Style Questions

### Technical Deep-Dive Questions

1. **"Explain your approach to chunking. Why did you choose that strategy?"**
   - Expected: Discussion of document structure, semantic boundaries, trade-offs

2. **"How would you handle entity disambiguation in a KG?"**
   - Expected: Entity linking, alias management, confidence scores

3. **"Walk me through your hybrid retrieval pipeline."**
   - Expected: Query understanding → routing → KG/RAG → fusion → generation

4. **"How do you ensure answers are grounded and not hallucinated?"**
   - Expected: Citations, hallucination detection, "I don't know" pattern

5. **"What metrics do you use to evaluate RAG system quality?"**
   - Expected: Faithfulness, relevancy, precision/recall, latency, cost

### System Design Questions

1. **"Design a RAG system for a company with 1M documents."**
   - Expected: Sharding, caching, batch processing, cost optimization

2. **"How would you scale a knowledge graph to billions of nodes?"**
   - Expected: Graph partitioning, distributed systems, query optimization

3. **"Design a monitoring system for a production RAG application."**
   - Expected: Logging, metrics (latency, accuracy), cost tracking, alerts

### Scenario-Based Questions

1. **"A user complains that the system keeps giving wrong answers. How do you debug?"**
   - Expected: Check retrieval quality, inspect prompts, evaluate chunks, test queries

2. **"Costs are too high. How do you optimize?"**
   - Expected: Smaller models, caching, batch processing, context compression

3. **"The system is slow. How do you improve latency?"**
   - Expected: Caching, async processing, smaller models, index optimization

### Decision Questions

These questions test your ability to make architectural tradeoffs. There's no single correct answer - what matters is your reasoning about constraints and priorities.

1. **You have a RAG system that performs well on factual queries but fails on multi-hop reasoning.**

   You can either:
   - Increase chunk overlap
   - Add reranking
   - Introduce a knowledge graph

   Which do you try first, and why?

   *Consider*: Implementation complexity, existing infrastructure, data characteristics, performance requirements.

2. **You need to choose a chunk size for a legal document corpus.**

   Option A: 512 tokens (faster retrieval, more granular)

   Option B: 2048 tokens (slower retrieval, more context)

   What factors determine your choice? What's the real tradeoff here?

   *Consider*: Document structure, query types, LLM context limits, retrieval precision vs recall.

3. **Your system currently uses text-embedding-ada-002. A new model offers 15% better accuracy but 3x higher latency.**

   Do you switch? What questions do you ask before deciding?

   *Consider*: User experience requirements, cost implications, accuracy vs speed tradeoff, production SLAs.

4. **Your retrieval returns mediocre results. You can either:**

   A) Retrieve top-20 chunks instead of top-5

   B) Keep top-5 but add a reranking step

   Which approach is better, and under what conditions would you choose the opposite?

   *Consider*: LLM context costs, reranking latency, quality vs cost, false positive handling.

5. **You're building a RAG system for medical records (highly sensitive data).**

   Option A: Use GPT-4 with careful prompt engineering and auditing

   Option B: Use a local Llama model with lower quality but complete data privacy

   How do you make this decision? What if accuracy matters for patient safety?

   *Consider*: Regulatory requirements, liability, performance requirements, deployment complexity.

6. **Your knowledge graph needs a database. Neo4j is mature but expensive at scale. A custom solution would be cheaper but requires engineering time.**

   What factors drive this decision? When is "build vs buy" the wrong framing?

   *Consider*: Team expertise, time to market, long-term maintenance, scale requirements, vendor lock-in.

7. **Users complain about stale data. You can either:**

   A) Disable caching entirely (fresh data, high costs)

   B) Implement smart cache invalidation (complex, might miss edge cases)

   C) Accept staleness with a clear TTL policy (simple, documented tradeoff)

   Which do you choose and why? What questions determine this?

   *Consider*: Data change frequency, user expectations, cost constraints, system complexity.

### Failure Diagnosis Questions

These test your ability to debug production systems. For each scenario, identify distinct failure modes and how you'd test for them.

1. **Your system returns fluent but factually incorrect answers with high confidence.**

   List three distinct failure modes that could cause this and how you would test for each.

   *Possible causes to consider*: Retrieval failures, prompt issues, hallucination, context contamination, outdated knowledge.

2. **After scaling from 10k to 1M documents, query performance degraded from 200ms to 8 seconds.**

   Diagnose three different bottlenecks that could cause this. How would you isolate which one is the culprit?

   *Possible causes to consider*: Index quality, memory issues, network latency, algorithmic complexity, database saturation.

3. **Retrieval quality is inconsistent: excellent for some queries, terrible for others.**

   What are three different root causes? How would you systematically identify which applies?

   *Possible causes to consider*: Query type mismatch, embedding model limitations, chunking boundary issues, sparse vs dense query characteristics.

4. **Your knowledge graph traversal sometimes returns 100,000+ nodes and times out.**

   Identify three distinct failure modes. How would you prevent this without losing legitimate broad queries?

   *Possible causes to consider*: Missing query limits, relationship explosion, cycle detection, query optimization, schema design.

5. **Your system costs have tripled but answer quality hasn't improved.**

   List three different resource waste patterns and how you'd detect each.

   *Possible causes to consider*: Over-retrieval, redundant API calls, inefficient caching, prompt bloat, unnecessary reranking.

### "Bad Idea" Questions (Anti-Pattern Recognition)

Mark each statement as True or False, then explain why. The explanation is what matters.

1. **True or False**: "Increasing the embedding dimension from 768 to 1536 always improves retrieval quality."

   *Why it matters*: Understand curse of dimensionality, overfitting, diminishing returns, computational cost.

2. **True or False**: "Retrieving more chunks (top-20 instead of top-5) always leads to better answers."

   *Why it matters*: Context pollution, cost implications, lost-in-the-middle problem, LLM attention limits.

3. **True or False**: "A larger LLM context window (128k tokens) eliminates the need for good retrieval."

   *Why it matters*: Cost scaling, attention degradation, retrieval as filtering, context vs reasoning.

4. **True or False**: "Your knowledge graph should store every possible relationship you can extract."

   *Why it matters*: Signal vs noise, query performance, maintenance burden, schema design principles.

5. **True or False**: "Fine-tuning your embedding model on domain data is always worth the effort."

   *Why it matters*: Cost-benefit analysis, data requirements, maintenance overhead, diminishing returns.

6. **True or False**: "Caching should be disabled in production to ensure users always get the freshest data."

   *Why it matters*: Cost optimization, latency requirements, staleness tolerance, cache invalidation strategies.

7. **True or False**: "Knowledge graphs are always faster than vector similarity search for multi-hop queries."

   *Why it matters*: Query complexity, graph size, indexing strategies, hybrid approaches.

8. **True or False**: "In a hybrid RAG+KG system, you should always weight both retrieval methods equally in fusion."

   *Why it matters*: Query-dependent routing, strengths/weaknesses of each approach, adaptive systems.

---
