# 6. PRACTICAL ENGINEERING SKILLS

(This section is about making your system actually work in production. Everything before this assumed clean data, perfect uptime, and users who ask well-formed questions. None of that is true. Real documents are messy PDFs with broken encoding. Real users ask ambiguous questions. Real systems crash at 3am. The code below handles these realities.)

## Document Processing Pipeline

### End-to-End Pipeline

```python
class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.md', '.html']

    def process_document(self, file_path):
        """
        Complete document processing pipeline
        """
        # Step 1: Extract text
        raw_text = self.extract_text(file_path)

        # Step 2: Clean text
        cleaned_text = self.clean_text(raw_text)

        # Step 3: Extract metadata
        metadata = self.extract_metadata(file_path, cleaned_text)

        # Step 4: Chunk text
        chunks = self.chunk_text(cleaned_text)

        # Step 5: Generate embeddings
        chunk_objects = self.create_chunk_objects(chunks, metadata)

        return chunk_objects

    def extract_text(self, file_path):
        """Extract text from various formats"""
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self.extract_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_from_docx(file_path)
        elif ext in ['.txt', '.md']:
            return Path(file_path).read_text(encoding='utf-8')
        elif ext == '.html':
            return self.extract_from_html(file_path)

    def extract_from_pdf(self, file_path):
        """Extract text from PDF"""
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def clean_text(self, text):
        """Clean extracted text"""
        import re

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers (simple heuristic)
        text = re.sub(r'\n\d+\n', '\n', text)

        # Fix broken words (simple version)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

        return text.strip()

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """Chunk text with overlap"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)
        return chunks

    def create_chunk_objects(self, chunks, metadata):
        """Create chunk objects with embeddings"""
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_obj = {
                "id": f"{metadata['file_name']}_{i}",
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "char_count": len(chunk)
                },
                "embedding": get_embedding(chunk)
            }
            chunk_objects.append(chunk_obj)

        return chunk_objects
```

### Handling Different File Types

```python
def process_code_files(file_path):
    """Special handling for code files"""
    from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

    # Detect language
    ext = Path(file_path).suffix
    language_map = {
        '.py': Language.PYTHON,
        '.js': Language.JS,
        '.java': Language.JAVA,
    }

    language = language_map.get(ext)
    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=500,
            chunk_overlap=50
        )
        code = Path(file_path).read_text()
        chunks = splitter.split_text(code)
        return chunks
```

## Metadata Extraction

### Extracting Rich Metadata

```python
class MetadataExtractor:
    def extract_metadata(self, file_path, content):
        """Extract comprehensive metadata"""
        metadata = {
            # File metadata
            "file_name": Path(file_path).name,
            "file_type": Path(file_path).suffix,
            "file_size": Path(file_path).stat().st_size,
            "created_date": datetime.fromtimestamp(
                Path(file_path).stat().st_ctime
            ).isoformat(),

            # Content metadata
            "char_count": len(content),
            "word_count": len(content.split()),

            # Extracted metadata
            "title": self.extract_title(content),
            "author": self.extract_author(content),
            "summary": self.extract_summary(content),
            "keywords": self.extract_keywords(content),
            "entities": self.extract_entities(content),
        }

        return metadata

    def extract_title(self, content):
        """Extract title from content"""
        # Method 1: First heading
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                return line.strip('#').strip()

        # Method 2: First line
        return lines[0][:100] if lines else "Untitled"

    def extract_entities(self, content):
        """Extract named entities"""
        import spacy
        nlp = spacy.load("en_core_web_sm")

        # Limit content for performance
        doc = nlp(content[:5000])

        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        # Deduplicate
        entities = {k: list(set(v)) for k, v in entities.items()}

        return entities

    def extract_summary(self, content):
        """Generate summary using LLM"""
        prompt = f"""
        Summarize this document in 2-3 sentences:

        {content[:2000]}

        Summary:
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        return response.choices[0].message.content
```

## Evaluation Frameworks

(Evaluation is where most RAG projects fail. You ship a system that seems to work, users complain it's wrong 30% of the time, and you have no systematic way to measure or fix it. The frameworks below give you actual numbers. Yes, setting up evaluation is tedious. No, you can't skip it and expect to improve your system. If you're not measuring, you're guessing.)

### RAGAS (RAG Assessment)

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

def evaluate_rag_system(test_questions, answers, contexts, ground_truths):
    """
    Evaluate RAG system using RAGAS metrics
    """
    from datasets import Dataset

    # Prepare data
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    # Evaluate
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )

    return result

# Example
test_questions = ["Who is the CEO?", "What is our Q4 revenue?"]
answers = ["The CEO is Alice", "Q4 revenue was $5.2M"]
contexts = [
    [["Alice Smith was appointed CEO in 2020"]],
    [["Our Q4 2024 revenue reached $5.2 million"]]
]
ground_truths = ["Alice Smith is the CEO", "Q4 2024 revenue was $5.2M"]

scores = evaluate_rag_system(test_questions, answers, contexts, ground_truths)
print(scores)
# {
#   'faithfulness': 0.95,
#   'answer_relevancy': 0.92,
#   'context_recall': 0.88,
#   'context_precision': 0.90
# }
```

### Custom Evaluation Metrics

```python
class RAGEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate_answer_quality(self, question, answer, retrieved_docs, ground_truth=None):
        """
        Comprehensive answer evaluation
        """
        metrics = {
            "relevance": self.score_relevance(question, answer),
            "groundedness": self.score_groundedness(answer, retrieved_docs),
            "completeness": self.score_completeness(question, answer),
            "citation_quality": self.score_citations(answer, retrieved_docs),
        }

        if ground_truth:
            metrics["accuracy"] = self.score_accuracy(answer, ground_truth)

        metrics["overall"] = sum(metrics.values()) / len(metrics)

        return metrics

    def score_relevance(self, question, answer):
        """Does the answer address the question?"""
        prompt = f"""
        Rate how well this answer addresses the question (0.0 to 1.0):

        Question: {question}
        Answer: {answer}

        Score (just the number):
        """

        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 0.5

    def score_groundedness(self, answer, retrieved_docs):
        """Is the answer supported by retrieved documents?"""
        docs_text = "\n".join([d['content'] for d in retrieved_docs])

        prompt = f"""
        Rate how well this answer is grounded in the provided documents (0.0 to 1.0):

        Documents:
        {docs_text}

        Answer: {answer}

        Score (just the number):
        """

        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 0.5
```

### TruLens Integration

```python
from trulens_eval import TruChain, Feedback, Tru

def setup_trulens_monitoring(rag_chain):
    """
    Set up TruLens for RAG monitoring
    """
    # Initialize TruLens
    tru = Tru()
    tru.reset_database()

    # Define feedback functions
    from trulens_eval.feedback import Groundedness

    grounded = Groundedness(groundedness_provider=openai)

    # Groundedness feedback
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on_input_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Answer relevance
    f_answer_relevance = (
        Feedback(openai.relevance)
        .on_input_output()
    )

    # Context relevance
    f_context_relevance = (
        Feedback(openai.qs_relevance)
        .on_input()
        .on(TruChain.select_context())
        .aggregate(np.mean)
    )

    # Wrap chain with TruLens
    tru_chain = TruChain(
        rag_chain,
        app_id="RAG_App",
        feedbacks=[
            f_groundedness,
            f_answer_relevance,
            f_context_relevance
        ]
    )

    return tru_chain
```

## Deployment Considerations

### FastAPI Application

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RAG+KG API")

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_kg: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    reasoning: str = None

# Global system (initialize once)
rag_kg_system = HybridRAGKGSystem()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG+KG system
    """
    try:
        result = rag_kg_system.query(
            request.question,
            top_k=request.top_k,
            use_kg=request.use_kg
        )

        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result.get('confidence', 0.0),
            reasoning=result.get('reasoning')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(file_path: str):
    """
    Ingest a new document
    """
    try:
        processor = DocumentProcessor()
        chunks = processor.process_document(file_path)

        # Add to vector DB
        for chunk in chunks:
            vector_db.add(chunk)

        # Extract entities and relationships for KG
        extractor = KnowledgeExtractor(client)
        triples = extractor.extract_triples(chunks[0]['content'])

        # Add to KG
        kg_builder = KnowledgeGraphBuilder(graph_db)
        for triple in triples:
            kg_builder.add_triple(*triple)

        return {"status": "success", "chunks_added": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data

  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
    depends_on:
      - neo4j

volumes:
  neo4j_data:
```

## Scaling Strategies

(Your prototype handles 10 queries per minute fine. Then someone puts it in production Slack and 1000 employees start using it simultaneously. Now you're paying $500/day in OpenAI API costs and queries take 15 seconds. Caching and batching aren't optimizations - they're requirements for anything beyond a demo.)

### Caching Layer

```python
from functools import lru_cache
import hashlib
import redis

class CacheLayer:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379)

    def cache_query(self, query, result, ttl=3600):
        """Cache query results"""
        key = self.get_cache_key(query)
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(result)
        )

    def get_cached_result(self, query):
        """Get cached result"""
        key = self.get_cache_key(query)
        result = self.redis_client.get(key)
        if result:
            return json.loads(result)
        return None

    def get_cache_key(self, query):
        """Generate cache key"""
        return f"query:{hashlib.md5(query.encode()).hexdigest()}"

# Usage
cache = CacheLayer()

def query_with_cache(question):
    # Check cache
    cached = cache.get_cached_result(question)
    if cached:
        return cached

    # Query system
    result = rag_kg_system.query(question)

    # Cache result
    cache.cache_query(question, result)

    return result
```

### Batch Processing

```python
async def process_documents_batch(file_paths, batch_size=10):
    """
    Process multiple documents in batches
    """
    import asyncio

    processor = DocumentProcessor()
    results = []

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]

        # Process batch in parallel
        tasks = [
            asyncio.to_thread(processor.process_document, fp)
            for fp in batch
        ]

        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    return results
```

### Load Balancing Multiple Vector DBs

```python
class ShardedVectorDB:
    def __init__(self, num_shards=3):
        self.shards = [
            ChromaClient(f"shard_{i}")
            for i in range(num_shards)
        ]

    def get_shard(self, document_id):
        """Route document to shard based on ID"""
        shard_idx = hash(document_id) % len(self.shards)
        return self.shards[shard_idx]

    def add(self, document_id, embedding, metadata):
        """Add to appropriate shard"""
        shard = self.get_shard(document_id)
        shard.add(document_id, embedding, metadata)

    def query(self, query_embedding, top_k=5):
        """Query all shards and merge results"""
        all_results = []

        for shard in self.shards:
            results = shard.query(query_embedding, top_k=top_k)
            all_results.extend(results)

        # Re-rank and return top-k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
```

## Cost Optimization

(Cost optimization sounds boring until you get your first $10,000 API bill. Embeddings are cheap per call but expensive at scale. LLM calls are expensive per call. Every retrieval spawns both. The optimizations below aren't premature - they're the difference between a sustainable product and bankruptcy. Implement them before you launch, not after.)

### Embedding Cost Optimization

```python
class EmbeddingOptimizer:
    def __init__(self):
        self.embedding_cache = {}

    def get_embedding_with_cache(self, text):
        """Cache embeddings to avoid redundant API calls"""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        # Generate embedding
        embedding = get_embedding(text)

        # Cache
        self.embedding_cache[text_hash] = embedding

        return embedding

    def batch_embed(self, texts, batch_size=100):
        """Batch embeddings for cost efficiency"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            response = client.embeddings.create(
                model="text-embedding-3-small",  # Cheaper model
                input=batch
            )

            embeddings.extend([item.embedding for item in response.data])

        return embeddings
```

### LLM Cost Optimization

```python
def optimize_llm_usage(question, contexts):
    """
    Reduce LLM costs by:
    1. Using smaller models when possible
    2. Reducing context size
    3. Caching common queries
    """
    # Use cheaper model for simple queries
    if is_simple_query(question):
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4"

    # Compress contexts
    compressed_context = compress_context(contexts, max_tokens=2000)

    # Generate answer
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {compressed_context}\n\nQuestion: {question}"}
        ],
        max_tokens=300  # Limit output tokens
    )

    return response.choices[0].message.content

def compress_context(contexts, max_tokens=2000):
    """
    Compress contexts to fit token limit
    """
    # Estimate tokens (rough: 1 token ≈ 4 chars)
    total_text = "\n".join(contexts)
    estimated_tokens = len(total_text) / 4

    if estimated_tokens <= max_tokens:
        return total_text

    # Truncate
    char_limit = max_tokens * 4
    return total_text[:char_limit] + "..."
```

### Monitoring Costs

```python
class CostTracker:
    def __init__(self):
        self.costs = {
            "embeddings": 0.0,
            "llm_calls": 0.0,
            "total": 0.0
        }

        # Pricing (as of 2026)
        self.pricing = {
            "text-embedding-3-small": 0.00002 / 1000,  # per token
            "gpt-3.5-turbo": 0.0015 / 1000,  # per token
            "gpt-4": 0.03 / 1000,  # per token
        }

    def track_embedding_cost(self, num_tokens, model="text-embedding-3-small"):
        cost = num_tokens * self.pricing[model]
        self.costs["embeddings"] += cost
        self.costs["total"] += cost

    def track_llm_cost(self, input_tokens, output_tokens, model="gpt-4"):
        cost = (input_tokens + output_tokens) * self.pricing[model]
        self.costs["llm_calls"] += cost
        self.costs["total"] += cost

    def get_report(self):
        return {
            "embeddings": f"${self.costs['embeddings']:.4f}",
            "llm_calls": f"${self.costs['llm_calls']:.4f}",
            "total": f"${self.costs['total']:.4f}"
        }
```

---
