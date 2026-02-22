# 5. HYBRID RAG + KG SYSTEMS (MAIN FOCUS)

(This is it. This is why you're here. RAG alone is useful but limited. Knowledge graphs alone are powerful but rigid. Combined correctly, you get a system that can answer questions neither approach could handle alone. Combined incorrectly, you get twice the complexity and worse results than either system alone. Pay attention to the failure modes in this section - they're drawn from real production systems that had to be rebuilt.)

## 5.1 Why Combine RAG and Knowledge Graphs?

### Limitations of Pure RAG

❌ **Struggles with multi-hop reasoning**
```
Question: "What technology does Alice's manager's company use?"
Pure RAG: Retrieves documents mentioning Alice, managers, companies, technology separately
→ No coherent answer
```

❌ **Misses structured relationships**
```
Question: "Who reports to the CTO?"
Pure RAG: Finds documents with "CTO" and "reports"
→ May miss implicit reporting structures
```

❌ **No entity disambiguation**
```
Question: "What does Apple produce?"
Pure RAG: Returns info about fruit OR company
→ No context to disambiguate
```

### Limitations of Pure KG

❌ **Can't handle unstructured knowledge**
```
Question: "What are best practices for API design?"
Pure KG: No nodes for "best practices" concept
→ Can't answer without structured triples
```

❌ **Limited by schema**
```
Question: "What did the CEO say in the Q4 earnings call?"
Pure KG: Doesn't store full transcript text
→ Only has structured metadata
```

### Power of Hybrid RAG + KG

✅ **Multi-hop reasoning** (from KG) + **Rich context** (from RAG)
✅ **Structured queries** (KG) + **Semantic search** (RAG)
✅ **Entity disambiguation** (KG) + **Document retrieval** (RAG)
✅ **Explainable paths** (KG) + **Cited answers** (RAG)

## 5.2 Graph-RAG Architecture

### Architecture Overview

```
User Query
    ↓
[1] Query Understanding
    ├─→ Extract entities
    ├─→ Classify intent
    └─→ Identify query type
    ↓
[2] Hybrid Retrieval
    ├─→ KG: Graph traversal for structured knowledge
    ├─→ RAG: Vector search for unstructured text
    └─→ Merge results
    ↓
[3] Context Enhancement
    ├─→ Expand KG neighborhoods
    ├─→ Retrieve related documents
    └─→ Rank and filter
    ↓
[4] LLM Generation
    ├─→ Combine graph paths + documents
    ├─→ Generate answer
    └─→ Add citations + reasoning traces
    ↓
Answer with explanation
```

### System Components

```python
class HybridRAGKGSystem:
    def __init__(self):
        self.vector_db = ChromaDB()        # For RAG
        self.graph_db = Neo4jConnection()  # For KG
        self.llm = OpenAI()               # For generation
        self.entity_linker = EntityLinker()

    def query(self, user_question):
        # Step 1: Understand query
        entities = self.extract_entities(user_question)
        query_type = self.classify_query(user_question)

        # Step 2: Retrieve from both sources
        if query_type == "structured":
            # KG-heavy retrieval
            graph_results = self.query_graph(entities)
            doc_results = self.query_documents(user_question, top_k=3)
        elif query_type == "unstructured":
            # RAG-heavy retrieval
            doc_results = self.query_documents(user_question, top_k=10)
            graph_results = self.query_graph(entities, max_depth=1)
        else:
            # Hybrid retrieval
            graph_results = self.query_graph(entities)
            doc_results = self.query_documents(user_question, top_k=5)

        # Step 3: Enhance context
        enhanced_context = self.enhance_context(
            graph_results, doc_results, entities
        )

        # Step 4: Generate answer
        answer = self.generate_answer(
            user_question, enhanced_context
        )

        return answer
```

## 5.3 KG-Augmented Retrieval

### Pattern 1: Entity-Centric Retrieval

**Use Case**: Query mentions specific entities

```python
def entity_centric_retrieval(query, entities):
    """
    1. Find entities in KG
    2. Get their neighborhoods
    3. Retrieve documents mentioning those neighbors
    """
    # Step 1: Find entity in KG
    kg_query = """
    MATCH (e:Entity {name: $entity})-[*1..2]-(neighbor)
    RETURN DISTINCT neighbor.name, labels(neighbor)[0] AS type
    """
    neighbors = graph_db.query(kg_query, {"entity": entities[0]})

    # Step 2: Build expanded query
    expanded_query = query + " " + " ".join([n['name'] for n in neighbors])

    # Step 3: Retrieve documents
    documents = vector_db.query(expanded_query, top_k=10)

    return {
        "graph_context": neighbors,
        "documents": documents
    }

# Example
query = "What projects is Alice working on?"
entities = ["Alice"]
results = entity_centric_retrieval(query, entities)

# Results include:
# - KG: Alice → WORKS_ON → Project X, Project Y
# - Docs: Project descriptions, meeting notes about those projects
```

### Pattern 2: Relationship-Aware Retrieval

```python
def relationship_aware_retrieval(subject, relation, object_=None):
    """
    Query like: "What does Alice manage?"
    → Find relationship in KG
    → Retrieve supporting documents
    """
    # Build Cypher query
    if object_:
        kg_query = """
        MATCH (s:Entity {name: $subject})-[r:RELATION {type: $relation}]->(o:Entity {name: $object})
        RETURN s, r, o
        """
        params = {"subject": subject, "relation": relation, "object": object_}
    else:
        kg_query = """
        MATCH (s:Entity {name: $subject})-[r:RELATION {type: $relation}]->(o)
        RETURN s, r, o
        LIMIT 10
        """
        params = {"subject": subject, "relation": relation}

    # Execute graph query
    graph_results = graph_db.query(kg_query, params)

    # For each result, find supporting documents
    all_docs = []
    for result in graph_results:
        doc_query = f"{result['s']['name']} {relation} {result['o']['name']}"
        docs = vector_db.query(doc_query, top_k=3)
        all_docs.extend(docs)

    return {
        "graph_facts": graph_results,
        "supporting_docs": all_docs
    }
```

### Pattern 3: Multi-Hop Graph → RAG

```python
def multi_hop_retrieval(start_entity, path_pattern, max_hops=3):
    """
    Follow graph paths, then retrieve documents for each node
    """
    # Find paths in graph
    kg_query = f"""
    MATCH path = (start:Entity {{name: $start}})-[*1..{max_hops}]-(end)
    WHERE {path_pattern}
    RETURN path, end
    LIMIT 20
    """

    paths = graph_db.query(kg_query, {"start": start_entity})

    # For each path, retrieve documents
    context = {
        "paths": [],
        "documents": {}
    }

    for path_result in paths:
        path = path_result['path']
        nodes = [node['name'] for node in path]

        context['paths'].append(nodes)

        # Retrieve docs for each node
        for node in nodes:
            if node not in context['documents']:
                docs = vector_db.query(node, top_k=2)
                context['documents'][node] = docs

    return context

# Example: "How is Alice connected to Machine Learning?"
context = multi_hop_retrieval(
    start_entity="Alice",
    path_pattern="end:Skill AND end.name = 'Machine Learning'"
)
# Returns:
# - Paths: [Alice → WORKS_ON → ML Project → REQUIRES → Machine Learning]
# - Documents for each: {Alice: [...], ML Project: [...], Machine Learning: [...]}
```

## 5.4 KG-Guided Query Routing

### Query Classification

```python
class QueryRouter:
    def __init__(self, llm):
        self.llm = llm

    def classify_query(self, query):
        """
        Classify query type to route to appropriate retrieval strategy
        """
        prompt = f"""
        Classify this query into one category:

        1. FACTUAL: Simple fact retrieval (who, what, when, where)
           Example: "Who is the CEO?"

        2. RELATIONAL: About relationships between entities
           Example: "Who reports to Alice?"

        3. MULTI_HOP: Requires following multiple relationships
           Example: "What skills do Alice's teammates have?"

        4. ANALYTICAL: Requires deep understanding or summarization
           Example: "What are the main challenges in our Q4 report?"

        5. HYBRID: Combines structured and unstructured knowledge
           Example: "How does our product compare to competitors based on customer feedback?"

        Query: {query}

        Classification (JSON):
        {{"type": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
        """

        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)
```

### Routing Logic

```python
def route_query(query, classification):
    """
    Route to appropriate retrieval strategy based on query type
    """
    query_type = classification['type']

    if query_type == "FACTUAL":
        # KG-first: Direct lookup
        return kg_factual_lookup(query)

    elif query_type == "RELATIONAL":
        # KG-only: Traverse relationships
        return kg_relationship_query(query)

    elif query_type == "MULTI_HOP":
        # KG traversal + RAG for context
        return multi_hop_retrieval(query)

    elif query_type == "ANALYTICAL":
        # RAG-heavy: Retrieve many docs, minimal KG
        return rag_heavy_retrieval(query)

    elif query_type == "HYBRID":
        # Full hybrid: Both systems equally
        return full_hybrid_retrieval(query)

def kg_factual_lookup(query):
    """Simple KG lookup for factual queries"""
    entities = extract_entities(query)
    if entities:
        result = graph_db.query("""
            MATCH (e:Entity {name: $name})
            RETURN e
        """, {"name": entities[0]})
        return {"source": "KG", "result": result}

def rag_heavy_retrieval(query):
    """RAG-focused for analytical queries"""
    docs = vector_db.query(query, top_k=20)
    reranked = rerank(query, docs, top_k=10)
    return {"source": "RAG", "documents": reranked}
```

## 5.5 Combining Structured + Unstructured Knowledge

### Context Fusion Strategy

(Context fusion is where most hybrid systems fail. You have graph facts ("Alice reports to Bob") and document snippets ("Bob recently announced new product priorities"). How do you combine them without confusing the LLM? The naive approach is to dump both into the prompt and hope for the best. That fails about 40% of the time - the LLM either ignores one source or hallucinates connections between them. The approach below actually works.)

```python
class ContextFusion:
    def fuse_contexts(self, graph_results, doc_results, query):
        """
        Combine graph paths and documents into unified context
        """
        # Build graph context
        graph_context = self.format_graph_context(graph_results)

        # Build document context
        doc_context = self.format_doc_context(doc_results)

        # Create fused prompt
        fused_context = f"""
        STRUCTURED KNOWLEDGE (from Knowledge Graph):
        {graph_context}

        UNSTRUCTURED KNOWLEDGE (from Documents):
        {doc_context}

        Instructions:
        - Use structured knowledge for facts, relationships, and entities
        - Use unstructured knowledge for details, explanations, and context
        - Cite sources: [Graph: ...] or [Doc: ...]
        - If structured and unstructured conflict, prefer structured for facts
        """

        return fused_context

    def format_graph_context(self, graph_results):
        """Format graph results as readable text"""
        formatted = []
        for result in graph_results:
            if 'path' in result:
                path_str = " → ".join([
                    f"{node['name']} ({node['type']})"
                    for node in result['path']
                ])
                formatted.append(f"- Path: {path_str}")
            elif 'entity' in result:
                entity = result['entity']
                formatted.append(
                    f"- Entity: {entity['name']} ({entity['type']}) "
                    f"Properties: {entity.get('properties', {})}"
                )
        return "\n".join(formatted)

    def format_doc_context(self, doc_results):
        """Format documents with source info"""
        formatted = []
        for i, doc in enumerate(doc_results):
            formatted.append(
                f"[Doc {i+1}] (Source: {doc.get('source', 'Unknown')})\n"
                f"{doc['content']}\n"
            )
        return "\n".join(formatted)
```

### Practical Example

```python
# Query: "What are Alice's manager's responsibilities?"

# Step 1: Extract entities
entities = ["Alice"]

# Step 2: Query KG
graph_results = graph_db.query("""
    MATCH (alice:Person {name: "Alice"})-[:REPORTS_TO]->(manager:Person)
    MATCH (manager)-[:RESPONSIBLE_FOR]->(responsibility)
    RETURN manager.name, collect(responsibility.name) AS responsibilities
""")
# Result: {"manager.name": "Bob", "responsibilities": ["Engineering", "Product"]}

# Step 3: Query RAG
doc_query = "Bob's responsibilities engineering product"
doc_results = vector_db.query(doc_query, top_k=5)
# Returns documents about Bob's role, engineering team, product roadmap

# Step 4: Fuse contexts
fusion = ContextFusion()
fused_context = fusion.fuse_contexts(graph_results, doc_results, query)

# Step 5: Generate answer
prompt = f"""
{fused_context}

Question: What are Alice's manager's responsibilities?

Answer with citations:
"""

answer = llm.generate(prompt)
# "Alice's manager is Bob [Graph]. His responsibilities include Engineering and Product [Graph].
#  According to the team charter, he oversees the development of core platform features [Doc 2]
#  and coordinates with the product team on roadmap priorities [Doc 3]."
```

## 5.6 Using LLMs to Generate Cypher Queries

### Text-to-Cypher

```python
class Text2Cypher:
    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema  # Graph schema description

    def generate_cypher(self, natural_language_query):
        """Convert natural language to Cypher query"""
        prompt = f"""
        Convert the natural language question to a Cypher query.

        GRAPH SCHEMA:
        {self.schema}

        RULES:
        - Use MATCH for patterns
        - Use WHERE for filters
        - Use RETURN for results
        - Limit results to 10 unless specified

        EXAMPLES:
        Q: "Who works at Acme?"
        A: MATCH (p:Person)-[:WORKS_FOR]->(c:Company {{name: "Acme"}})
           RETURN p.name

        Q: "Who does Alice report to?"
        A: MATCH (alice:Person {{name: "Alice"}})-[:REPORTS_TO]->(manager)
           RETURN manager.name

        QUESTION: {natural_language_query}

        CYPHER QUERY:
        """

        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        cypher_query = response.choices[0].message.content.strip()
        return cypher_query

# Usage
schema = """
Nodes:
- Person (properties: name, email, role)
- Company (properties: name, industry)
- Project (properties: name, status)

Relationships:
- (Person)-[:WORKS_FOR]->(Company)
- (Person)-[:REPORTS_TO]->(Person)
- (Person)-[:WORKS_ON]->(Project)
"""

text2cypher = Text2Cypher(llm, schema)
query = "What projects is Alice working on?"
cypher = text2cypher.generate_cypher(query)
# MATCH (p:Person {name: "Alice"})-[:WORKS_ON]->(proj:Project)
# RETURN proj.name, proj.status
```

### Query Validation

```python
def validate_and_execute_cypher(cypher_query, graph_db):
    """
    Validate Cypher query before execution
    """
    # Basic validation
    if "DELETE" in cypher_query.upper() or "REMOVE" in cypher_query.upper():
        raise ValueError("Destructive queries not allowed")

    # Dry run (explain query)
    try:
        explain_query = f"EXPLAIN {cypher_query}"
        graph_db.query(explain_query)
    except Exception as e:
        # Query has syntax error
        return {
            "success": False,
            "error": str(e),
            "suggestion": "Check Cypher syntax"
        }

    # Execute
    try:
        results = graph_db.query(cypher_query)
        return {
            "success": True,
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### Self-Correcting Cypher Generation

```python
def self_correcting_text2cypher(nl_query, max_attempts=3):
    """
    Generate Cypher with self-correction
    """
    for attempt in range(max_attempts):
        # Generate Cypher
        cypher = text2cypher.generate_cypher(nl_query)

        # Try to execute
        result = validate_and_execute_cypher(cypher, graph_db)

        if result['success']:
            return result['results']

        # If failed, try to fix
        if attempt < max_attempts - 1:
            fix_prompt = f"""
            This Cypher query failed:
            {cypher}

            Error: {result['error']}

            Generate a corrected version:
            """
            # Continue loop with correction
            nl_query = fix_prompt

    return {"error": "Failed to generate valid Cypher after retries"}
```

## 5.7 KG Reasoning + RAG Context for Perfect Answers

### The Perfect Answer Pattern

```python
class PerfectAnswerSystem:
    def answer_question(self, question):
        """
        Combine KG reasoning + RAG context for comprehensive answers
        """
        # Step 1: Extract structured components
        entities = self.extract_entities(question)
        intent = self.classify_intent(question)

        # Step 2: KG reasoning (find facts and paths)
        kg_facts = self.kg_reasoning(entities, intent)

        # Step 3: RAG context (find supporting details)
        rag_context = self.rag_retrieval(question, entities)

        # Step 4: Verify facts (cross-check KG with RAG)
        verified_facts = self.verify_facts(kg_facts, rag_context)

        # Step 5: Generate comprehensive answer
        answer = self.generate_with_reasoning(
            question,
            verified_facts,
            rag_context,
            kg_facts
        )

        return answer

    def kg_reasoning(self, entities, intent):
        """Extract facts and relationships from KG"""
        # Generate Cypher based on intent
        if intent == "relationship":
            cypher = f"""
            MATCH (e1)-[r]-(e2)
            WHERE e1.name IN {entities}
            RETURN e1, type(r) AS relationship, e2
            LIMIT 20
            """
        elif intent == "property":
            cypher = f"""
            MATCH (e)
            WHERE e.name IN {entities}
            RETURN e, properties(e) AS props
            """
        else:
            cypher = f"""
            MATCH path = (e1)-[*1..3]-(e2)
            WHERE e1.name IN {entities}
            RETURN path
            LIMIT 10
            """

        return graph_db.query(cypher, {"entities": entities})

    def verify_facts(self, kg_facts, rag_context):
        """Cross-verify KG facts with RAG documents"""
        verified = []

        for fact in kg_facts:
            # Check if any document supports this fact
            fact_str = self.format_fact(fact)
            supporting_docs = [
                doc for doc in rag_context
                if self.supports_fact(doc, fact_str)
            ]

            verified.append({
                "fact": fact,
                "confidence": "high" if supporting_docs else "medium",
                "supporting_docs": supporting_docs
            })

        return verified

    def generate_with_reasoning(self, question, facts, context, kg_facts):
        """Generate answer with reasoning trace"""
        prompt = f"""
        Answer the question using the provided information.

        QUESTION: {question}

        VERIFIED FACTS (from Knowledge Graph):
        {json.dumps(facts, indent=2)}

        SUPPORTING CONTEXT (from Documents):
        {self.format_docs(context)}

        Instructions:
        1. Answer the question directly
        2. Explain your reasoning
        3. Cite all sources
        4. Show the logical path from question to answer

        Format:
        ANSWER: [Direct answer]

        REASONING:
        - [Step-by-step logical reasoning]

        EVIDENCE:
        - [Graph facts cited]
        - [Documents cited]

        Response:
        """

        response = llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content
```

## 5.8 Trustworthiness and Explainability Patterns

### Pattern 1: Provenance Tracking

```python
class ProvenanceTracker:
    def track_answer_sources(self, answer, kg_results, rag_results):
        """
        Track where each claim in the answer comes from
        """
        # Parse answer into claims
        claims = self.extract_claims(answer)

        provenance = []
        for claim in claims:
            sources = {
                "claim": claim,
                "kg_support": self.find_kg_support(claim, kg_results),
                "rag_support": self.find_rag_support(claim, rag_results),
                "confidence": self.calculate_confidence(claim, kg_results, rag_results)
            }
            provenance.append(sources)

        return provenance

    def calculate_confidence(self, claim, kg_results, rag_results):
        """Calculate confidence based on source agreement"""
        kg_support = len(self.find_kg_support(claim, kg_results))
        rag_support = len(self.find_rag_support(claim, rag_results))

        if kg_support > 0 and rag_support > 0:
            return "HIGH"  # Both sources agree
        elif kg_support > 0 or rag_support > 0:
            return "MEDIUM"  # One source
        else:
            return "LOW"  # No clear source
```

### Pattern 2: Reasoning Chains

```python
def generate_with_reasoning_chain(question, kg_context, rag_context):
    """
    Generate answer with explicit reasoning chain
    """
    prompt = f"""
    Answer the question step-by-step, showing your reasoning.

    KG Context: {kg_context}
    RAG Context: {rag_context}

    Question: {question}

    Format your response as:

    THOUGHT 1: [What I need to find out first]
    ACTION 1: [Which knowledge source to use: KG or RAG]
    OBSERVATION 1: [What I found]

    THOUGHT 2: [Next step in reasoning]
    ACTION 2: [...]
    OBSERVATION 2: [...]

    FINAL ANSWER: [Complete answer with citations]
    """

    response = llm.generate(prompt)
    return response
```

### Pattern 3: Confidence Scores

```python
class ConfidenceScorer:
    def score_answer(self, answer, question, sources):
        """
        Score answer confidence based on multiple factors
        """
        scores = {
            "source_agreement": self.check_source_agreement(sources),
            "coverage": self.check_question_coverage(question, answer),
            "specificity": self.check_specificity(answer),
            "citation_quality": self.check_citations(answer, sources)
        }

        # Weighted average
        total_score = (
            scores["source_agreement"] * 0.4 +
            scores["coverage"] * 0.3 +
            scores["specificity"] * 0.2 +
            scores["citation_quality"] * 0.1
        )

        return {
            "overall_confidence": total_score,
            "breakdown": scores,
            "recommendation": self.get_recommendation(total_score)
        }

    def get_recommendation(self, score):
        if score > 0.8:
            return "HIGH CONFIDENCE: Answer is well-supported"
        elif score > 0.5:
            return "MEDIUM CONFIDENCE: Answer is partially supported"
        else:
            return "LOW CONFIDENCE: Answer may be unreliable"
```

## 5.9 Architecture Diagrams

### Diagram 1: Basic Hybrid Flow

```
┌─────────────────────────────────────────────────────────┐
│                    USER QUERY                           │
│          "What projects is Alice's manager working on?" │
└────────────────────┬────────────────────────────────────┘
                     │
                     ∨
         ┌───────────────────────┐
         │  Entity Extraction    │
         │  Entities: [Alice]    │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ∨                       ∨
┌────────────────┐      ┌────────────────┐
│  KNOWLEDGE     │      │  VECTOR DB     │
│  GRAPH (Neo4j) │      │  (ChromaDB)    │
└────────┬───────┘      └────────┬───────┘
         │                       │
         │ Cypher:               │ Similarity search:
         │ Alice→Manager         │ "manager projects"
         │ Manager→Projects      │
         │                       │
         ∨                       ∨
┌────────────────┐      ┌────────────────┐
│ Graph Results: │      │ Documents:     │
│ - Alice        │      │ - Project      │
│   →Bob         │      │   descriptions │
│ - Bob          │      │ - Meeting      │
│   →Project X   │      │   notes        │
│   →Project Y   │      │ - Status       │
└────────┬───────┘      └────────┬───────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ∨
         ┌───────────────────────┐
         │   Context Fusion      │
         │   Combine KG + RAG    │
         └───────────┬───────────┘
                     │
                     ∨
         ┌───────────────────────┐
         │   LLM Generation      │
         │   (GPT-4)             │
         └───────────┬───────────┘
                     │
                     ∨
┌─────────────────────────────────────────────────────────┐
│                   FINAL ANSWER                          │
│ "Alice's manager is Bob [Graph]. Bob is currently       │
│  working on Project X and Project Y [Graph]. Project X  │
│  focuses on API redesign [Doc 1], while Project Y is    │
│  the mobile app refresh [Doc 2]."                       │
└─────────────────────────────────────────────────────────┘
```

### Diagram 2: Query Routing Decision Tree

```
                    [User Query]
                         │
                         ∨
              ┌──────────────────────┐
              │  Query Classifier    │
              └──────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ∨               ∨               ∨
    [Factual]      [Relational]    [Analytical]
         │               │               │
         ∨               ∨               ∨
    KG Direct      KG Traversal      RAG Heavy
    Lookup         + RAG Lite        + KG Lite
         │               │               │
         └───────────────┴───────────────┘
                         │
                         ∨
                   [Generate Answer]
```

## 5.10 Comparison: Plain RAG vs Hybrid RAG+KG

(This table answers the question everyone asks: "Is the extra complexity worth it?" Short answer: it depends. If your queries are simple lookups over documents, stick with RAG. If you need multi-hop reasoning, entity disambiguation, or explainable answers over structured data, the hybrid approach is worth the complexity. Don't build it because it's cool - build it because your use case demands it.)

| Aspect | Plain RAG | Hybrid RAG + KG |
|--------|-----------|-----------------|
| **Multi-hop questions** | ❌ Struggles, needs many retrievals | ✅ Direct graph traversal |
| **Entity disambiguation** | ❌ No context | ✅ KG provides entity types |
| **Relationship queries** | ❌ Keyword-based, imprecise | ✅ Structured relationships |
| **Unstructured knowledge** | ✅ Excellent | ✅ Same via RAG |
| **Explainability** | ⚠️  Citations only | ✅ Citations + reasoning paths |
| **Setup complexity** | Low | High |
| **Query latency** | Fast (100-300ms) | Medium (300-800ms) |
| **Accuracy (structured)** | Medium (70-80%) | High (85-95%) |
| **Accuracy (unstructured)** | High (85-90%) | High (85-95%) |

### Example Comparison

**Query**: "What technology does Alice's manager's company use?"

**Plain RAG**:
```
Retrieved docs:
- "Alice is an engineer..."
- "The company uses Python and AWS..."
- "Manager Bob oversees engineering..."

Answer: "The company uses Python and AWS" ❌
Problem: Doesn't identify manager or verify company connection
```

**Hybrid RAG + KG**:
```
KG traversal:
Alice →[REPORTS_TO]→ Bob →[WORKS_FOR]→ Acme Corp

KG query for Acme's tech:
Acme Corp →[USES_TECHNOLOGY]→ [Python, AWS, PostgreSQL]

RAG retrieval:
"Acme Corp's tech stack includes..."

Answer: "Alice's manager is Bob, who works at Acme Corp [Graph].
Acme Corp uses Python, AWS, and PostgreSQL [Graph + Doc 3]." ✅
```

(You now understand hybrid RAG + KG architectures conceptually. The next section is about making them work in production - deployment, monitoring, evaluation, and all the messy details tutorials skip. This is where theory meets reality, and where most systems break in ways you didn't anticipate.)

---
