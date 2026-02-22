# 7. 10 HANDS-ON PROJECTS

Each project builds your skills progressively, from simple RAG to complex hybrid systems.

(These projects are where learning happens. Reading about RAG is easy. Building a system that actually works is hard. Each project will break in ways you didn't expect - PDFs with weird encoding, queries that return garbage, graphs that are too slow to query. That's the point. Fix the breakage and you'll understand the material. Skip the projects and you'll forget everything in a week.)

## Project 1: Simple PDF RAG Chatbot

**Goal**: Build a basic RAG system that answers questions about PDF documents.

**Skills Required**:
- PDF text extraction
- Chunking
- Embeddings
- Vector search
- Basic prompting

**Architecture**:
```
PDF → Extract Text → Chunk → Embed → Store in ChromaDB
User Query → Embed → Retrieve Chunks → LLM → Answer
```

**Step-by-Step Tasks**:
1. Extract text from 3-5 PDF documents (use PyPDF2)
2. Chunk text into 500-token segments with 50-token overlap
3. Generate embeddings using `text-embedding-3-small`
4. Store in ChromaDB with metadata (file_name, page_number)
5. Implement query function: embed query → retrieve top 5 chunks → generate answer
6. Add citation: include source file and page number

**Evaluation Criteria**:
- ✅ Correctly extracts text from PDFs
- ✅ Answers questions with relevant context
- ✅ Includes citations (file + page)
- ✅ Handles "I don't know" when answer not in docs

**Dataset**: Use 3-5 research papers from arXiv or your domain

---

## Project 2: Multi-Hop RAG System

**Goal**: Handle complex questions requiring multiple retrieval steps.

**Skills Required**:
- Query decomposition
- Multi-step retrieval
- Context aggregation

**Architecture**:
```
Complex Query → Decompose into Sub-Queries → Retrieve for Each → Aggregate → Answer
```

**Step-by-Step Tasks**:
1. Implement query decomposition using GPT-4
2. For each sub-query, retrieve relevant chunks
3. Aggregate all retrieved contexts
4. Generate final answer synthesizing all sub-answers
5. Test on multi-hop questions like:
   - "Compare the methodologies of paper A and paper B"
   - "What are the advantages and disadvantages of approach X?"

**Evaluation Criteria**:
- ✅ Successfully decomposes complex queries
- ✅ Retrieves relevant context for each sub-query
- ✅ Synthesizes coherent final answer
- ✅ Handles at least 3-hop reasoning

**Test Questions**:
- "How does the transformer architecture differ from RNNs and what are the performance implications?"
- "What are the trade-offs between the approaches discussed in documents X, Y, and Z?"

---

## Project 3: Automatic KG Builder from Text

**Goal**: Extract entities and relationships from text and build a knowledge graph.

**Skills Required**:
- Named entity recognition
- Relationship extraction
- Triple generation
- Neo4j graph construction

**Architecture**:
```
Text Documents → NER + Relation Extraction → Triples → Neo4j Knowledge Graph
```

**Step-by-Step Tasks**:
1. Use spaCy for entity extraction (Person, Org, Location, etc.)
2. Implement relationship extraction using LLM
3. Generate triples (Subject, Predicate, Object)
4. Deduplicate entities (entity linking)
5. Load triples into Neo4j
6. Visualize graph in Neo4j Browser

**Evaluation Criteria**:
- ✅ Extracts at least 50+ entities
- ✅ Identifies at least 30+ relationships
- ✅ Graph is queryable in Neo4j
- ✅ Entity deduplication works (e.g., "Alice" = "Alice Smith")

**Dataset**: News articles, Wikipedia pages, or company documents

---

## Project 4: Entity/Relationship Extractor

**Goal**: Build a production-grade entity and relationship extraction pipeline.

**Skills Required**:
- Advanced NLP
- LLM-based extraction
- Schema validation
- Batch processing

**Architecture**:
```
Text → Entity Extraction → Relationship Extraction → Validation → Structured Output
```

**Step-by-Step Tasks**:
1. Define schema (entity types, relationship types)
2. Implement entity extraction with confidence scores
3. Implement relationship extraction between identified entities
4. Add validation layer (ensure relationships make sense)
5. Output structured JSON with entities + relationships
6. Handle batch processing for multiple documents

**Evaluation Criteria**:
- ✅ Precision > 80% on entity extraction
- ✅ Recall > 70% on relationship extraction
- ✅ Handles at least 100 documents
- ✅ Output is valid against schema

**Bonus**: Add support for custom entity types

---

## Project 5: Cypher Query Generator Using LLMs

**Goal**: Convert natural language to Cypher queries.

**Skills Required**:
- Text-to-Cypher prompting
- Query validation
- Error handling

**Architecture**:
```
Natural Language → LLM (with schema) → Cypher Query → Validate → Execute → Results
```

**Step-by-Step Tasks**:
1. Create graph schema description (nodes, relationships, properties)
2. Implement text-to-Cypher using GPT-4 with few-shot examples
3. Add query validation (syntax check, no destructive operations)
4. Implement self-correction (if query fails, retry with error message)
5. Execute query on Neo4j and return results
6. Format results in human-readable way

**Evaluation Criteria**:
- ✅ Generates syntactically valid Cypher 90% of the time
- ✅ Correctly answers factual queries ("Who works at Company X?")
- ✅ Handles relationship queries ("Who does Alice report to?")
- ✅ Self-corrects failed queries

**Test Queries**:
- "Find all employees in the engineering department"
- "What projects is Alice working on?"
- "Show me the management chain for Bob"

---

## Project 6: KG Search Engine

**Goal**: Build a search engine powered by knowledge graph traversal.

**Skills Required**:
- Graph algorithms (PageRank, shortest path)
- Cypher query optimization
- Result ranking

**Architecture**:
```
Search Query → Entity Recognition → Graph Traversal → Rank Results → Display
```

**Step-by-Step Tasks**:
1. Implement entity recognition in search queries
2. Build query expansion using graph neighborhoods
3. Implement PageRank to rank important nodes
4. Find shortest paths between entities
5. Build search result page showing:
   - Direct matches
   - Related entities
   - Connection paths
6. Add filters (entity type, relationship type)

**Evaluation Criteria**:
- ✅ Returns relevant results for entity searches
- ✅ Shows relationship paths between entities
- ✅ Ranks results by importance (PageRank)
- ✅ Sub-second query performance

**Features**:
- Auto-complete for entity names
- "People also searched for" recommendations
- Graph visualization of results

---

## Project 7: RAG with Reranker + Query Rewrite

**Goal**: Build an advanced RAG system with reranking and query optimization.

**Skills Required**:
- Hybrid retrieval (BM25 + semantic)
- Cross-encoder reranking
- Query rewriting

**Architecture**:
```
Query → Rewrite → Hybrid Retrieval → Rerank → LLM Generation
```

**Step-by-Step Tasks**:
1. Implement query rewriting using LLM
2. Build hybrid retriever (BM25 + dense embeddings)
3. Retrieve top 100 candidates
4. Rerank using cross-encoder (e.g., `ms-marco-MiniLM`)
5. Take top 5 after reranking
6. Generate answer with GPT-4
7. Compare performance: plain RAG vs. this system

**Evaluation Criteria**:
- ✅ Query rewriting improves retrieval recall by 15%+
- ✅ Reranking improves answer quality by 20%+
- ✅ Outperforms baseline RAG on test set
- ✅ Handles ambiguous queries well

**Metrics to Track**:
- Retrieval recall @ 5
- Answer relevance score
- Latency

---

## Project 8: Graph-RAG with Neighborhood Expansion

**Goal**: Implement Microsoft GraphRAG pattern - use KG to expand retrieval context.

**Skills Required**:
- Graph traversal
- Context fusion
- Hybrid reasoning

**Architecture**:
```
Query → Extract Entities → KG Neighborhood → RAG Retrieval → Fuse → Answer
```

**Step-by-Step Tasks**:
1. Extract entities from user query
2. Find entities in knowledge graph
3. Expand to 2-hop neighborhood (get related entities)
4. Use neighborhood entities to expand RAG query
5. Retrieve documents mentioning these entities
6. Fuse KG facts + RAG documents
7. Generate answer using both sources

**Evaluation Criteria**:
- ✅ Successfully expands query context using KG
- ✅ Answers multi-hop questions correctly
- ✅ Outperforms plain RAG on relationship queries
- ✅ Provides graph-based reasoning in answer

**Test Scenarios**:
- "What technology does Alice's team use?" (requires: Alice → Team → Technology)
- "Which products are related to Project X?" (requires: Project X → (relationships) → Products)

---

## Project 9: Hybrid RAG + KG Chatbot

**Goal**: Build a full conversational chatbot with hybrid retrieval.

**Skills Required**:
- Chat memory
- Context management
- Query routing
- Streaming responses

**Architecture**:
```
User Message → Route (KG vs RAG vs Hybrid) → Retrieve → Generate → Stream Response
```

**Step-by-Step Tasks**:
1. Implement conversation memory (store last 5 turns)
2. Build query router (classify query type)
3. Route to appropriate retrieval strategy:
   - Factual → KG
   - Analytical → RAG
   - Relationship → Hybrid
4. Maintain context across turns
5. Stream responses for better UX
6. Add conversation reset functionality

**Evaluation Criteria**:
- ✅ Maintains context across conversation
- ✅ Routes queries correctly 85%+ of the time
- ✅ Handles follow-up questions ("What about his manager?")
- ✅ Streams responses smoothly

**Features**:
- Chat UI (Streamlit or Gradio)
- Conversation export
- Source citations
- Reasoning explanation toggle

---

## Project 10: Production-Ready Enterprise Knowledge Assistant

**Goal**: Build a complete, deployable enterprise knowledge system.

**Skills Required**:
- Full-stack development
- API design
- Deployment
- Monitoring
- Testing

**Architecture**:
```
FastAPI Backend → Docker → Neo4j + ChromaDB → Frontend (React/Streamlit)
```

**Step-by-Step Tasks**:
1. **Backend (FastAPI)**:
   - `/query` endpoint (POST)
   - `/ingest` endpoint (upload docs)
   - `/health` endpoint
   - Authentication (API keys)

2. **Data Processing**:
   - Async document processing
   - Progress tracking
   - Error handling

3. **Deployment**:
   - Dockerize application
   - Docker Compose for all services
   - Environment variables for config

4. **Monitoring**:
   - Query logging
   - Performance metrics
   - Cost tracking

5. **Testing**:
   - Unit tests for core functions
   - Integration tests for API
   - Load testing (100 concurrent queries)

6. **Frontend**:
   - Chat interface
   - Document upload
   - Admin panel (view metrics)

**Evaluation Criteria**:
- ✅ Handles 100+ concurrent users
- ✅ 99% uptime over 1 week
- ✅ Sub-second p95 latency
- ✅ Full test coverage (>80%)
- ✅ Deployed and accessible via URL

**Deliverables**:
- GitHub repository with README
- Deployed application (AWS/Vercel/etc.)
- API documentation
- Demo video (5 min)

---
