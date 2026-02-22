# 1. COURSE OVERVIEW

## What You Will Learn

This comprehensive program transforms you from a beginner into a production-ready RAG + Knowledge Graph engineer capable of building enterprise-grade hybrid retrieval systems. You'll master:

- **Core RAG Systems**: Build sophisticated retrieval-augmented generation pipelines with chunking, embedding, indexing, and reranking
- **Knowledge Graph Engineering**: Design, build, and query complex knowledge graphs using Neo4j and graph databases
- **Hybrid RAG + KG Architecture**: Combine structured graph reasoning with unstructured retrieval for superior AI systems
- **Production Engineering**: Deploy, evaluate, scale, and optimize real-world systems
- **Enterprise Patterns**: Implement hallucination control, cited answers, query routing, and explainability

**Learning Path**: Theory → Fundamentals → RAG Deep Dive → KG Deep Dive → Hybrid Systems → 10 Projects → Capstone

(If this looks like a lot, don't worry. We'll build up systematically. If it looks too simple, also don't worry - there's plenty of depth ahead.)

## Why RAG + KG is a High-Demand Skill

### Market Reality (2026)
- **Salary Range**: $120k-$250k for RAG/KG engineers at top companies
- **Job Growth**: 347% increase in RAG-related job postings (2023-2026)
- **Enterprise Adoption**: 78% of Fortune 500 companies investing in RAG systems

### Why This Matters
1. **LLMs alone aren't enough**: ChatGPT hallucinates, lacks context, can't access private data
2. **Pure RAG has limits**: Struggles with multi-hop reasoning, structured knowledge, relationships
3. **Hybrid = Competitive Advantage**: Companies need engineers who can combine both approaches

### The Gap You'll Fill
Most engineers know either:
- LLMs (prompting, fine-tuning) OR
- Traditional search/databases

**You'll be rare**: An engineer who masters both unstructured (RAG) and structured (KG) knowledge systems.

(You might be thinking "but I can just glue together a vector database and call it a day." Please don't do that. Your future self will thank you for learning this properly.)

## Real Industry Applications

### Where RAG + KG Systems Are Deployed

| Industry | Use Case | Why Hybrid RAG + KG? |
|----------|----------|---------------------|
| **Healthcare** | Clinical decision support | Need both medical literature (RAG) and drug interactions graph (KG) |
| **Finance** | Investment research assistant | Combine news/reports (RAG) with company relationship graphs (KG) |
| **Legal** | Contract analysis | Find similar clauses (RAG) + track legal precedent chains (KG) |
| **E-commerce** | Product recommendation | Product descriptions (RAG) + user-product-category graph (KG) |
| **Customer Support** | Intelligent help desk | FAQs/docs (RAG) + issue resolution paths (KG) |
| **Scientific Research** | Literature discovery | Papers (RAG) + citation/author networks (KG) |

### Real Companies Using This Stack
- **Microsoft**: GraphRAG for enterprise search
- **Amazon**: Product knowledge graphs + semantic search
- **Google**: Knowledge Graph + BERT for search
- **Meta**: Social graph + content retrieval
- **OpenAI**: Retrieval plugins with structured data

### Detailed Case Studies from Industry

(The case studies below are long and detailed. This is intentional - you need to see how these systems actually work in production, not just toy examples. Skim them now if you want, but come back when you're building your own systems. The patterns here will save you months of trial and error.)

#### Case Study 1: Healthcare - Clinical Decision Support System

**Company**: Mayo Clinic (anonymized implementation)

**Problem Statement**:
Physicians need to make treatment decisions based on:
- Latest research papers (100,000+ published annually)
- Drug interaction databases (structured knowledge)
- Patient history (unstructured clinical notes)
- Treatment protocols (semi-structured guidelines)

**Traditional Approach Limitations**:
- Pure keyword search: Misses semantic similarity ("myocardial infarction" vs "heart attack")
- Manual review: Impossible to read all relevant literature
- Static databases: Don't incorporate latest research

**Hybrid RAG + KG Solution**:

```
Architecture:
1. RAG Component:
   - Ingest: PubMed papers, clinical guidelines, case reports
   - Chunking: Section-based (Methods, Results, Conclusions separate)
   - Embeddings: BioBERT (domain-specific for medical text)
   - Vector DB: Pinecone with metadata filtering (date, journal impact factor)

2. KG Component:
   - Nodes: Diseases, Drugs, Symptoms, Treatments, Contraindications
   - Relationships:
     * Drug -[TREATS]-> Disease
     * Drug -[INTERACTS_WITH]-> Drug
     * Symptom -[INDICATES]-> Disease
     * Patient -[HAS_CONDITION]-> Disease
   - Graph DB: Neo4j with temporal properties

3. Hybrid Query Flow:
   User Query: "Treatment options for diabetic patient with hypertension?"

   Step 1: Entity Extraction
   - Entities: {Diabetes, Hypertension}

   Step 2: KG Reasoning
   ```cypher
   MATCH (d1:Disease {name: 'Diabetes'})<-[:TREATS]-(drug:Drug)-[:TREATS]->(d2:Disease {name: 'Hypertension'})
   WHERE NOT EXISTS {
     (drug)-[:CONTRAINDICATED_FOR]->(d1)
   } AND NOT EXISTS {
     (drug)-[:CONTRAINDICATED_FOR]->(d2)
   }
   RETURN drug.name, drug.effectiveness_score
   ORDER BY drug.effectiveness_score DESC
   ```
   → Returns: [Metformin, Lisinopril, ...]

   Step 3: RAG Retrieval
   - Query: "Latest research on {Metformin} AND {Lisinopril} for diabetic hypertensive patients"
   - Retrieve top 10 papers from vector DB
   - Filter by publication date > 2020

   Step 4: Context Fusion
   ```python
   context = f"""
   Structured Knowledge (Knowledge Graph):
   - Recommended drugs: {kg_results}
   - Known interactions: {interaction_paths}
   - Contraindications: {contraindications}

   Research Evidence (RAG):
   {retrieved_papers}

   Patient Context:
   - Age: {patient.age}
   - Current medications: {patient.meds}
   - Allergies: {patient.allergies}
   """
   ```

   Step 5: LLM Generation with Citations
   ```
   Output: "Based on current guidelines [Source: KG] and recent research [Paper 1, 2023],
   Metformin combined with Lisinopril is recommended for diabetic patients with
   hypertension [Paper 2, 2024]. Note: Monitor potassium levels due to potential
   interaction [Source: Drug Interaction DB]."
   ```
```

**Results**:
- **Accuracy**: 94% agreement with expert physician decisions (vs 76% with pure RAG)
- **Time Saved**: Reduced research time from 45 min to 5 min per complex case
- **Safety**: Zero missed drug interactions (vs 12% miss rate with manual lookup)
- **Adoption**: 87% of physicians use system daily after 3 months

**Key Success Factors**:
1. Domain-specific embeddings (BioBERT) improved retrieval by 23%
2. Temporal KG properties track when research was published
3. Mandatory citation forcing prevents hallucination
4. Integration with EHR (Electronic Health Records) for patient context

**Challenges Overcome**:
- **Privacy**: Deployed on-premise, no data leaves hospital network
- **Latency**: Cached common queries, pre-computed KG paths
- **Trust**: Extensive validation against historical cases before deployment

(At this point, you might be thinking "this looks straightforward enough." It wasn't. Getting physicians to trust an AI system with medical decisions took 18 months of validation, countless edge cases, and a lot of "the system was technically correct but clinically useless" feedback. Production AI is hard.)

---

#### Case Study 2: Finance - Investment Research Assistant

**Company**: Goldman Sachs (public information synthesis)

**Problem Statement**:
Investment analysts need to:
- Monitor 10,000+ companies daily
- Track relationships (ownership, partnerships, competition)
- Analyze news sentiment and quarterly reports
- Identify hidden connections and risk factors

**Hybrid RAG + KG Solution**:

```
KG Schema:
(Company)-[:OWNS]->(Subsidiary)
(Company)-[:PARTNERS_WITH {since: date}]->(Company)
(Company)-[:COMPETES_IN]->(Market)
(Executive)-[:SERVES_ON_BOARD]->(Company)
(Company)-[:SUPPLIES_TO]->(Company)
(Fund)-[:HOLDS {shares: int, value: float}]->(Company)

RAG Sources:
- SEC filings (10-K, 10-Q, 8-K)
- Earnings call transcripts
- News articles
- Analyst reports
- Social media sentiment

Example Query: "What are the supply chain risks for Tesla?"

Hybrid Retrieval:
1. KG: Find supply chain graph
   ```cypher
   MATCH path = (tesla:Company {name: 'Tesla'})<-[:SUPPLIES_TO*1..3]-(supplier)
   RETURN supplier.name, supplier.country, supplier.revenue_dependency
   ```
   → {Panasonic (batteries, Japan, 45% revenue from Tesla),
      CATL (batteries, China, 12% revenue from Tesla), ...}

2. RAG: Retrieve news about suppliers
   - "Recent news about Panasonic battery production"
   - "CATL supply chain disruptions"
   - Semantic search in earnings transcripts mentioning suppliers

3. Risk Synthesis:
   ```
   LLM Analysis:
   "Tesla faces significant supply chain concentration risk:

   Tier-1 Suppliers (from KG):
   - Panasonic: 45% revenue dependency [High risk if Tesla switches]
   - Geographic: 67% suppliers in Asia [Geopolitical risk]

   Recent Events (from RAG):
   - Panasonic announced $4B investment in US battery plant [Positive, reduces geographic risk]
   - CATL affected by COVID lockdowns Q2 2023 [Temporary disruption]

   Recommendation: Monitor Panasonic partnership closely. Diversification
   efforts underway but 18-24 month timeline to reduce dependency."
   ```
```

**Results**:
- **Coverage**: Tracks 50,000+ company relationships automatically
- **Speed**: Generates comprehensive research report in 10 minutes (vs 8 hours manual)
- **Hidden Insights**: Discovered 15% more risk factors through multi-hop KG traversal
- **ROI**: $12M annual savings in analyst time

**Architecture Highlights**:
```python
class FinancialHybridRAG:
    def __init__(self):
        self.kg = Neo4jKnowledgeGraph()
        self.vector_db = PineconeVectorDB(index="financial-docs")
        self.llm = GPT4()

    def analyze_company(self, company_name, query):
        # 1. Entity linking
        company_node = self.kg.find_company(company_name)

        # 2. Graph analysis
        supply_chain = self.kg.get_supply_chain(company_node, depth=3)
        ownership_structure = self.kg.get_ownership_tree(company_node)
        board_connections = self.kg.get_board_interlocks(company_node)

        # 3. RAG retrieval with KG-guided filters
        entities_of_interest = supply_chain + ownership_structure.entities

        rag_results = self.vector_db.query(
            query=f"{query} {company_name}",
            filter={
                "company": [e.name for e in entities_of_interest],
                "date": {"$gte": "2023-01-01"},
                "doc_type": ["10-K", "8-K", "news", "transcript"]
            },
            top_k=20
        )

        # 4. Rerank by relevance + recency
        reranked = self.rerank(rag_results, recency_weight=0.3)

        # 5. Generate structured analysis
        return self.llm.analyze(
            kg_context=supply_chain + ownership_structure,
            documents=reranked,
            query=query,
            output_format="structured_risk_report"
        )
```

---

#### Case Study 3: E-Commerce - Personalized Product Discovery

**Company**: Amazon (approximated based on public patents)

**Problem Statement**:
- 400M+ products in catalog
- Users struggle to describe what they want ("comfortable shoes for walking" = 100K results)
- Need to understand product relationships, not just descriptions

**Hybrid RAG + KG Architecture**:

```
Product Knowledge Graph:
(Product)-[:BELONGS_TO]->(Category)
(Product)-[:COMPATIBLE_WITH]->(Product)
(Product)-[:SIMILAR_TO {similarity_score: float}]->(Product)
(User)-[:VIEWED]->(Product)
(User)-[:PURCHASED]->(Product)
(Product)-[:FREQUENTLY_BOUGHT_WITH]->(Product)
(Review)-[:MENTIONS {sentiment: float}]->(Feature)

Query: "I need running shoes for marathon training, but my feet pronate"

Step 1: Query Understanding (RAG)
- Embed query: text-embedding-3-large
- Retrieve similar past queries + their resolutions
- Extract: {
    use_case: "marathon running",
    foot_type: "overpronation",
    intent: "purchase"
  }

Step 2: KG Constraint Satisfaction
```cypher
MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: 'Running Shoes'})
WHERE p.pronation_support = 'overpronation' OR p.pronation_support = 'neutral'
AND p.use_case CONTAINS 'marathon'

// Find products with positive reviews for relevant features
MATCH (p)-[:HAS_REVIEW]->(r:Review)-[:MENTIONS {sentiment: > 0.7}]->(f:Feature)
WHERE f.name IN ['cushioning', 'stability', 'durability']

// Boost products similar users liked
OPTIONAL MATCH (similar_user:User)-[:PURCHASED]->(p)
WHERE similar_user.foot_type = 'overpronation'

RETURN p,
       count(r) as review_count,
       avg(r.rating) as avg_rating,
       count(similar_user) as similar_user_purchases
ORDER BY similar_user_purchases DESC, avg_rating DESC
LIMIT 50
```

Step 3: RAG Semantic Search
- Embed product descriptions
- Find semantically similar to "marathon stability overpronation"
- Retrieve product reviews mentioning relevant features

Step 4: Hybrid Ranking
```python
final_score = (
    0.3 * kg_popularity_score +     # Graph-based popularity
    0.3 * rag_semantic_similarity +  # Description similarity
    0.2 * user_personalization +     # Based on user history graph
    0.2 * review_sentiment           # From review embeddings
)
```

Step 5: Explanation Generation
```
Result:
"ASICS Gel-Kayano 29 - $160
★★★★★ 4.7 (12,453 reviews)

Why we recommend this:
- Designed for overpronation [Graph: product_features]
- 89% of users with your foot type rated 4+ stars [Graph: similar_user_purchases]
- Highly cushioned for long distances [Reviews: "comfortable for 20+ miles"]
- Frequently bought by marathon runners [Graph: FREQUENTLY_BOUGHT_WITH other marathon gear]

Alternatives: {other_recommendations with explanations}
"
```
```

**Results**:
- **Conversion Rate**: +18% compared to pure keyword search
- **Customer Satisfaction**: 4.6/5 for recommendations (vs 3.8/5 baseline)
- **Discovery**: 34% of purchases from products user wouldn't have found via search
- **Explainability**: 92% of users found explanations helpful

---

#### Case Study 4: Legal Tech - Contract Analysis System

**Company**: LegalTech Startup (Series B, $50M ARR)

**Problem**: Lawyers spend 60% of time on document review

**Solution**:

```
KG Schema:
(Clause)-[:APPEARS_IN]->(Contract)
(Clause)-[:SIMILAR_TO {similarity: float}]->(Clause)
(Clause)-[:STANDARD_FOR]->(ContractType)
(Clause)-[:RISKY_IF_COMBINED_WITH]->(Clause)
(Clause)-[:PRECEDENT_FROM]->(LegalCase)

Use Case: "Review this NDA for unusual clauses"

Process:
1. Extract clauses from new NDA (spaCy + custom NER)
2. For each clause:
   a. RAG: Find similar clauses in clause library (100K+ contracts)
   b. KG: Check if clause is standard for NDA type
   c. KG: Identify risky clause combinations

3. Risk Scoring:
   ```cypher
   MATCH (clause:Clause {from_doc: 'new_nda.pdf'})

   // Find how common this clause is in similar contracts
   MATCH (clause)-[:SIMILAR_TO {similarity: > 0.9}]->(similar)
   -[:APPEARS_IN]->(contract:Contract {type: 'NDA'})
   WITH clause, count(contract) as frequency

   // Check for risky combinations
   OPTIONAL MATCH (clause)-[:RISKY_IF_COMBINED_WITH]->(other)
   WHERE EXISTS((other)-[:APPEARS_IN]->({from_doc: 'new_nda.pdf'}))

   RETURN clause.text,
          frequency,
          CASE
            WHEN frequency < 5 THEN 'UNUSUAL'
            WHEN other IS NOT NULL THEN 'RISKY_COMBINATION'
            ELSE 'STANDARD'
          END as risk_level
   ```

4. Generate Report:
   ```
   Unusual Clause Detected (Clause 7.3):

   Text: "Non-compete extends to 5 years post-termination"

   Analysis:
   - Standard duration: 1-2 years [RAG: Similar NDAs show 89% use 1-2 years]
   - Legal precedent: Courts often void >3 year non-competes [KG: LegalCase connections]
   - Risk: HIGH - May be unenforceable, reduces employee mobility

   Recommendation: Negotiate to 2 years maximum

   Similar Clauses (for comparison): [RAG retrieves 5 examples]
   ```
```

**Impact**:
- **Review Time**: 3 hours → 30 minutes per contract
- **Risk Detection**: 95% accuracy identifying non-standard clauses
- **Cost Savings**: $200K/year per lawyer in billable hours
- **Competitive Advantage**: Win 40% more clients due to faster turnaround

(Notice a pattern in these case studies? The hybrid approach isn't just "nice to have" - in every case, pure RAG or pure KG would have failed. The finance case needs the graph to track relationships but RAG to analyze news. The legal case needs RAG for similar clauses but the graph to identify risky combinations. This is why you're learning both.)

## Skills & Tools You'll Master

### Core Technologies

**Languages & Frameworks**
- Python (primary)
- LangChain / LlamaIndex
- FastAPI (deployment)

**LLM Tools**
- OpenAI API (GPT-4)
- Anthropic Claude
- Open-source models (Llama, Mistral)
- Embedding models (text-embedding-3, BGE)

**Vector Databases**
- FAISS (local)
- ChromaDB
- Pinecone / Weaviate (production)

**Graph Databases**
- Neo4j (primary)
- GraphDB / Stardog (optional)

**Evaluation & Monitoring**
- RAGAS
- TruLens
- LangSmith

**Supporting Tools**
- Docker
- Git
- Jupyter Notebooks
- Pytest

### Skills Matrix

By course completion, you'll have:

| Skill Category | Beginner Level | Your Level (After Course) |
|----------------|----------------|---------------------------|
| LLM Prompting | Basic ChatGPT use | Advanced prompt engineering + function calling |
| Retrieval Systems | Google search concepts | Custom hybrid retrievers with reranking |
| Graph Theory | No experience | Design complex schemas, write Cypher queries |
| Production Deployment | Scripts only | Dockerized APIs, monitoring, evaluation |
| System Architecture | Single-file code | Multi-component production systems |

---
