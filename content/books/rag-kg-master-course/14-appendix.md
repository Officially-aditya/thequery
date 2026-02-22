# APPENDIX: Quick Reference

(This appendix has the code snippets you'll actually copy-paste when building systems. Bookmark this page. You'll be back here constantly until this stuff becomes muscle memory.)

## Common Code Patterns

### 1. Basic RAG Query
```python
def rag_query(question, vector_db, llm):
    # Retrieve
    docs = vector_db.query(get_embedding(question), top_k=5)

    # Generate
    context = "\n".join([d['content'] for d in docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    answer = llm.generate(prompt)

    return answer
```

### 2. KG Entity Lookup
```cypher
MATCH (e:Entity {name: $entity_name})
RETURN e, properties(e)
```

### 3. Hybrid Retrieval
```python
# Get from both sources
kg_facts = graph_db.query(cypher_query)
rag_docs = vector_db.query(embedding)

# Fuse
context = format_kg(kg_facts) + format_docs(rag_docs)
answer = llm.generate(context + question)
```

## Essential Tools Installation

```bash
# Python packages
pip install openai chromadb neo4j langchain sentence-transformers \
    faiss-cpu rank-bm25 spacy fastapi uvicorn pytest ragas

# Download spaCy model
python -m spacy download en_core_web_sm

# Neo4j (Docker)
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

## Glossary

- **RAG**: Retrieval-Augmented Generation
- **KG**: Knowledge Graph
- **Embedding**: Dense vector representation of text
- **Chunking**: Splitting documents into smaller pieces
- **Triple**: (Subject, Predicate, Object) relationship
- **Cypher**: Neo4j query language
- **BM25**: Keyword-based ranking algorithm
- **Reranker**: Model that re-scores retrieval results
- **Context Fusion**: Combining KG and RAG contexts

---
