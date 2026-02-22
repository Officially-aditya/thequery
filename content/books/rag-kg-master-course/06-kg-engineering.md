# 4. KNOWLEDGE GRAPH ENGINEERING MODULE

(Knowledge graphs are where structured reasoning lives. RAG gives you semantic search, but knowledge graphs give you logical traversal - "find the manager's manager's direct reports who work on ML projects." You can't do that with vector similarity alone. The challenge isn't the tech, it's figuring out what your schema should look like before you've loaded a million nodes.)

## Graph Schema Design

### What is a Schema?

**Definition**: The blueprint of your graph - what types of nodes and relationships exist, and what properties they have.

**Analogy**: Like a database schema, but for graphs.

(Unlike SQL schemas, graph schemas are flexible - you can add new node types and relationships without migrations. This is both a blessing and a curse. Blessing: easy to evolve. Curse: people abuse this flexibility and end up with an unmaintainable mess of ad-hoc relationships. Design your schema properly from the start.)

### Schema Design Process

#### Step 1: Identify Entities (Nodes)

**Example Domain**: Company Knowledge Base

Entities:
- Person (employees, customers)
- Company
- Product
- Project
- Document

#### Step 2: Identify Relationships (Edges)

Relationships:
- Person WORKS_FOR Company
- Person MANAGES Person
- Person AUTHORED Document
- Company PRODUCES Product
- Project USES Product

#### Step 3: Define Properties

```cypher
// Node properties
Person: {name, email, role, hire_date}
Company: {name, industry, founded_year}
Product: {name, version, release_date}
Document: {title, content, created_date}

// Relationship properties
WORKS_FOR: {since, position}
MANAGES: {since}
AUTHORED: {date, contribution_type}
```

### Complete Schema Example

```cypher
// Create constraints (ensures data quality)
CREATE CONSTRAINT person_email IF NOT EXISTS
FOR (p:Person) REQUIRE p.email IS UNIQUE;

CREATE CONSTRAINT company_name IF NOT EXISTS
FOR (c:Company) REQUIRE c.name IS UNIQUE;

// Example data following schema
CREATE (alice:Person {
    name: "Alice Smith",
    email: "alice@example.com",
    role: "Engineer",
    hire_date: date("2020-01-15")
})

CREATE (acme:Company {
    name: "Acme Corp",
    industry: "Technology",
    founded_year: 2010
})

CREATE (alice)-[:WORKS_FOR {
    since: date("2020-01-15"),
    position: "Senior Engineer"
}]->(acme)
```

### Schema Best Practices

1. **Use Clear Labels**: `Person` not `P`, `WORKS_FOR` not `W4`
2. **Normalize Data**: Store shared properties once
3. **Plan for Queries**: Design schema around your query patterns
4. **Use Constraints**: Enforce uniqueness and data integrity

## Triple Extraction from Text

### What is Triple Extraction?

**Goal**: Convert unstructured text into (Subject, Predicate, Object) triples.

(This is harder than it looks. The examples below make it seem easy - extract some nouns and verbs, done. Real text is messy. Pronouns, implied relationships, context-dependent meaning, ambiguous references. Rule-based extraction gets you 60% accuracy at best. LLM-based extraction gets you 85-90% but costs money and is slow. Pick your tradeoff based on your quality requirements.)

**Example**:
```
Text: "Alice works at Acme Corp as a senior engineer."

Triples:
(Alice, WORKS_AT, Acme Corp)
(Alice, HAS_ROLE, Senior Engineer)
(Acme Corp, TYPE, Company)
```

### Method 1: Rule-Based Extraction

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_triples_basic(text):
    doc = nlp(text)
    triples = []

    for sent in doc.sents:
        subject = None
        relation = None
        object_ = None

        for token in sent:
            # Find subject (noun)
            if token.dep_ in ("nsubj", "nsubjpass") and not subject:
                subject = token.text

            # Find relation (verb)
            if token.pos_ == "VERB" and not relation:
                relation = token.lemma_

            # Find object
            if token.dep_ in ("dobj", "pobj") and not object_:
                object_ = token.text

        if subject and relation and object_:
            triples.append((subject, relation.upper(), object_))

    return triples

# Example
text = "Alice works at Acme Corp. Bob manages the engineering team."
triples = extract_triples_basic(text)
# [("Alice", "WORK", "Acme Corp"), ("Bob", "MANAGE", "team")]
```

### Method 2: LLM-Based Extraction (Better Quality)

```python
def extract_triples_llm(text):
    prompt = f"""
    Extract knowledge graph triples from the text below.
    Format: (Subject, Relationship, Object)

    Text: {text}

    Triples (as JSON list):
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    # Parse response
    triples = eval(response.choices[0].message.content)
    return triples

# Example
text = """
Alice Smith is a senior engineer at Acme Corp.
She has been working there since 2020.
Acme Corp is a technology company founded in 2010.
"""

triples = extract_triples_llm(text)
# [
#   ("Alice Smith", "IS_A", "Senior Engineer"),
#   ("Alice Smith", "WORKS_AT", "Acme Corp"),
#   ("Alice Smith", "WORKS_SINCE", "2020"),
#   ("Acme Corp", "IS_A", "Technology Company"),
#   ("Acme Corp", "FOUNDED_IN", "2010")
# ]
```

### Method 3: Production-Grade Extraction

```python
from typing import List, Tuple
import json

class KnowledgeExtractor:
    def __init__(self, client):
        self.client = client

    def extract_triples(self, text: str) -> List[Tuple]:
        prompt = f"""
        Extract structured knowledge from the text.

        For each entity and relationship:
        1. Identify entities (people, companies, products, concepts)
        2. Identify relationships between entities
        3. Extract properties of entities

        Format output as JSON:
        {{
            "entities": [
                {{"id": "e1", "type": "Person", "name": "Alice Smith", "properties": {{}}}},
                ...
            ],
            "relationships": [
                {{"subject": "e1", "predicate": "WORKS_AT", "object": "e2", "properties": {{}}}},
                ...
            ]
        }}

        Text:
        {text}

        JSON:
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

# Usage
extractor = KnowledgeExtractor(client)
result = extractor.extract_triples(text)
```

## Entity Linking

### What is Entity Linking?

**Problem**: Different text mentions refer to the same entity.

```
Text 1: "Alice works at Acme"
Text 2: "Alice Smith is an engineer"
Text 3: "A. Smith wrote the report"

Question: Are these the same Alice?
```

**Solution**: Entity linking resolves mentions to canonical entities.

### Simple Entity Linking

```python
class EntityLinker:
    def __init__(self):
        self.entities = {}  # Canonical entities
        self.aliases = {}   # Alias → Canonical mapping

    def add_entity(self, canonical_name, aliases=None):
        self.entities[canonical_name] = {"name": canonical_name}
        if aliases:
            for alias in aliases:
                self.aliases[alias.lower()] = canonical_name

    def link(self, mention):
        mention_lower = mention.lower()
        return self.aliases.get(mention_lower, mention)

# Usage
linker = EntityLinker()
linker.add_entity("Alice Smith", aliases=["Alice", "A. Smith", "alice"])

print(linker.link("Alice"))      # "Alice Smith"
print(linker.link("A. Smith"))   # "Alice Smith"
print(linker.link("alice"))      # "Alice Smith"
```

### LLM-Based Entity Linking

```python
def link_entities_llm(mentions, known_entities):
    prompt = f"""
    Match each mention to a known entity, or mark as NEW.

    Known entities:
    {json.dumps(known_entities, indent=2)}

    Mentions:
    {json.dumps(mentions, indent=2)}

    Output format (JSON):
    [
        {{"mention": "Alice", "linked_to": "Alice Smith"}},
        {{"mention": "Bob", "linked_to": "NEW"}}
    ]

    JSON:
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

## Building KGs Using Neo4j

(Neo4j is the most popular graph database for good reason - it's mature, fast, and has excellent tooling. The Docker setup below takes 30 seconds. The hard part isn't installation, it's designing your schema and remembering to create indexes before you load a million nodes and wonder why queries take 10 seconds.)

### Setting Up Neo4j

```bash
# Using Docker
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

### Connecting from Python

```python
from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

# Connect
conn = Neo4jConnection(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

### Building a KG from Triples

```python
class KnowledgeGraphBuilder:
    def __init__(self, neo4j_conn):
        self.conn = neo4j_conn

    def add_triple(self, subject, predicate, object_, properties=None):
        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATION {type: $predicate}]->(o)
        """
        if properties:
            query += "\nSET r += $properties"

        self.conn.query(query, {
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "properties": properties or {}
        })

    def build_from_text(self, text):
        # Extract triples
        triples = extract_triples_llm(text)

        # Add to graph
        for subject, predicate, object_ in triples:
            self.add_triple(subject, predicate, object_)

# Usage
builder = KnowledgeGraphBuilder(conn)
builder.build_from_text("""
    Alice Smith is a senior engineer at Acme Corp.
    She manages the data team.
    Acme Corp was founded in 2010.
""")
```

### Production KG Builder

```python
class ProductionKGBuilder:
    def __init__(self, neo4j_conn):
        self.conn = neo4j_conn
        self.create_indexes()

    def create_indexes(self):
        """Create indexes for performance"""
        self.conn.query("""
            CREATE INDEX entity_name IF NOT EXISTS
            FOR (e:Entity) ON (e.name)
        """)

    def add_entity(self, entity_type, name, properties):
        query = f"""
        MERGE (e:{entity_type} {{name: $name}})
        SET e += $properties
        RETURN e
        """
        return self.conn.query(query, {
            "name": name,
            "properties": properties
        })

    def add_relationship(self, from_entity, rel_type, to_entity, properties=None):
        query = """
        MATCH (a {name: $from})
        MATCH (b {name: $to})
        MERGE (a)-[r:REL {type: $rel_type}]->(b)
        SET r += $properties
        RETURN r
        """
        return self.conn.query(query, {
            "from": from_entity,
            "to": to_entity,
            "rel_type": rel_type,
            "properties": properties or {}
        })

    def bulk_import(self, triples):
        """Efficient bulk import"""
        for subject, predicate, object_ in triples:
            self.add_entity("Entity", subject, {})
            self.add_entity("Entity", object_, {})
            self.add_relationship(subject, predicate, object_)
```

## Querying with Cypher

(Cypher looks weird if you're coming from SQL. The ASCII-art syntax (`()-[]->()`) feels gimmicky at first. It's not. It's actually brilliant - you can read queries visually as graph patterns. Give it a week and you'll prefer it to SQL JOINs for graph traversals.)

### Basic Queries

```cypher
// Find all people
MATCH (p:Person)
RETURN p.name, p.email

// Find who works where
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)
RETURN p.name, c.name

// Find with filters
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)
WHERE c.industry = "Technology"
RETURN p.name, p.role, c.name
```

### Multi-Hop Queries

```cypher
// Friends of friends
MATCH (me:Person {name: "Alice"})-[:FRIEND]->(friend)-[:FRIEND]->(fof)
RETURN DISTINCT fof.name

// People who work at same company as Alice's friends
MATCH (alice:Person {name: "Alice"})-[:FRIEND]->(friend)-[:WORKS_FOR]->(c:Company)
MATCH (colleague:Person)-[:WORKS_FOR]->(c)
WHERE colleague <> alice AND colleague <> friend
RETURN colleague.name, c.name

// Path finding: How is Alice connected to Bob?
MATCH path = shortestPath((alice:Person {name: "Alice"})-[*]-(bob:Person {name: "Bob"}))
RETURN path
```

### Aggregation Queries

```cypher
// Count employees per company
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)
RETURN c.name, COUNT(p) AS employee_count
ORDER BY employee_count DESC

// Average team size
MATCH (manager:Person)-[:MANAGES]->(employee:Person)
RETURN manager.name, COUNT(employee) AS team_size
ORDER BY team_size DESC
```

### Advanced Pattern Matching

```cypher
// Find triangles (A knows B, B knows C, C knows A)
MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a)
RETURN a.name, b.name, c.name

// Find influential people (many incoming connections)
MATCH (p:Person)
WITH p, size((p)<-[:REPORTS_TO]-()) AS subordinates
WHERE subordinates > 5
RETURN p.name, subordinates
ORDER BY subordinates DESC

// Recommendation: Products used by similar people
MATCH (me:Person {name: "Alice"})-[:USES]->(p:Product)
MATCH (similar:Person)-[:USES]->(p)
MATCH (similar)-[:USES]->(rec:Product)
WHERE NOT (me)-[:USES]->(rec)
RETURN rec.name, COUNT(similar) AS score
ORDER BY score DESC
LIMIT 5
```

## Graph Traversal Reasoning

### Neighborhood Expansion

```python
def get_neighborhood(entity_name, max_depth=2):
    query = f"""
    MATCH path = (e:Entity {{name: $name}})-[*1..{max_depth}]-(neighbor)
    RETURN DISTINCT neighbor.name, length(path) AS distance
    ORDER BY distance
    """
    return conn.query(query, {"name": entity_name})

# Example
neighbors = get_neighborhood("Alice Smith", max_depth=2)
# Returns all entities within 2 hops of Alice
```

### Path-Based Reasoning

```cypher
// Find all paths between two entities
MATCH path = (a:Person {name: "Alice"})-[*..5]-(b:Person {name: "Bob"})
RETURN path
LIMIT 10

// Find shortest path
MATCH path = shortestPath((a:Person {name: "Alice"})-[*]-(b:Person {name: "Bob"}))
RETURN [node in nodes(path) | node.name] AS path_nodes,
       [rel in relationships(path) | type(rel)] AS path_relationships
```

### Complex Multi-Hop Reasoning

```python
def find_expertise_path(person, skill):
    """
    Find how a person is connected to a skill
    (e.g., through projects, colleagues, training)
    """
    query = """
    MATCH path = (p:Person {name: $person})-[*..4]-(s:Skill {name: $skill})
    WITH path,
         [node in nodes(path) | labels(node)[0] + ': ' + node.name] AS path_desc,
         length(path) AS dist
    ORDER BY dist
    LIMIT 5
    RETURN path_desc, dist
    """
    return conn.query(query, {"person": person, "skill": skill})

# Example: How does Alice connect to "Machine Learning"?
paths = find_expertise_path("Alice Smith", "Machine Learning")
# Might return:
# [Person: Alice] -> [WORKS_ON] -> [Project: ML Pipeline] -> [REQUIRES] -> [Skill: Machine Learning]
```

### Graph Algorithms

```cypher
// PageRank (find influential nodes)
CALL gds.pageRank.stream('my-graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
LIMIT 10

// Community Detection
CALL gds.louvain.stream('my-graph')
YIELD nodeId, communityId
RETURN communityId, collect(gds.util.asNode(nodeId).name) AS members

// Shortest paths with weights
MATCH (start:Person {name: "Alice"}), (end:Person {name: "Bob"})
CALL gds.shortestPath.dijkstra.stream('my-graph', {
    sourceNode: start,
    targetNode: end,
    relationshipWeightProperty: 'weight'
})
YIELD path
RETURN path
```

## Micro-Projects

### Project 4A: Build a Movie Knowledge Graph

**Goal**: Create a KG from movie data (actors, directors, genres).

**Dataset**: Use IMDB CSV or API

**Tasks**:
1. Design schema (Movie, Person, Genre nodes)
2. Extract triples from movie descriptions
3. Load into Neo4j
4. Query: "Find actors who worked with Christopher Nolan"

```cypher
// Schema
CREATE (m:Movie {title: "Inception", year: 2010})
CREATE (p:Person {name: "Leonardo DiCaprio"})
CREATE (d:Person {name: "Christopher Nolan"})
CREATE (g:Genre {name: "Sci-Fi"})

CREATE (p)-[:ACTED_IN {role: "Cobb"}]->(m)
CREATE (d)-[:DIRECTED]->(m)
CREATE (m)-[:HAS_GENRE]->(g)

// Query
MATCH (actor:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(director:Person {name: "Christopher Nolan"})
RETURN DISTINCT actor.name
```

### Project 4B: Academic Citation Graph

**Goal**: Build citation network from research papers.

**Tasks**:
1. Extract (Paper, CITES, Paper) relationships
2. Find most influential papers (PageRank)
3. Find papers in same research cluster

```cypher
// Build graph
CREATE (p1:Paper {title: "Attention Is All You Need", year: 2017})
CREATE (p2:Paper {title: "BERT", year: 2018})
CREATE (p2)-[:CITES]->(p1)

// Most cited papers
MATCH (p:Paper)
WITH p, size((p)<-[:CITES]-()) AS citations
WHERE citations > 10
RETURN p.title, citations
ORDER BY citations DESC
```

### Project 4C: Company Org Chart

**Goal**: Model organizational hierarchy.

**Queries to Implement**:
- Who reports to whom?
- What is the management chain for person X?
- Which teams are largest?

```cypher
// Build org structure
CREATE (ceo:Person {name: "Jane Doe", title: "CEO"})
CREATE (cto:Person {name: "John Smith", title: "CTO"})
CREATE (eng:Person {name: "Alice", title: "Engineer"})

CREATE (eng)-[:REPORTS_TO]->(cto)
CREATE (cto)-[:REPORTS_TO]->(ceo)

// Find management chain
MATCH path = (emp:Person {name: "Alice"})-[:REPORTS_TO*]->(top)
WHERE NOT (top)-[:REPORTS_TO]->()
RETURN [person in nodes(path) | person.name] AS chain
```

(You now have both pieces: RAG for semantic search over unstructured text, and knowledge graphs for structured traversal over relationships. Separately, they're useful. Together, they're transformative. Section 5 shows you how to combine them - and why most attempts to do this fail.)

---
