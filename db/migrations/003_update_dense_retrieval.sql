WITH dense_retrieval AS (
  SELECT $body$
Dense retrieval is a search method that represents a query and each document, passage, image, or other item as learned numerical vectors called embeddings. The system retrieves items whose vectors are closest to the query vector under a similarity function such as dot product, cosine similarity, or Euclidean distance.

The practical advantage is semantic matching. A sparse keyword system may see “automobile maintenance” and “car repair” as different strings. A well-trained dense retriever can place them near each other in embedding space because they express a similar meaning. Dense retrieval powers semantic search, retrieval-augmented generation (RAG), recommendation, duplicate detection, and multimodal search.

## How dense retrieval works

Dense retrieval normally has an offline indexing stage and an online query stage:

1. **Split and prepare the corpus.** Documents are cleaned and often divided into passages or chunks. Chunk size and boundaries determine what can be retrieved as one result.
2. **Encode the corpus.** A neural encoder converts every item into a fixed-size vector. These document embeddings can be computed before users search.
3. **Build an index.** The vectors are stored in an exact or approximate-nearest-neighbor index. Metadata and the original content are stored alongside their vector IDs.
4. **Encode the query.** The same model, or a paired query encoder, converts the user's query into the same vector space.
5. **Retrieve the nearest items.** The index returns the top-k vectors under the configured similarity metric. Filters, sparse scores, or a reranker may then refine the candidates.

For a single-vector bi-encoder, the core score is commonly written as **score(q, d) = E_q(q) · E_d(d)**, where the query encoder produces one vector for the query and the document encoder produces one vector for the document. Because document vectors are independent of the query, they can be precomputed and searched efficiently.

## Dense retrieval vs semantic search vs vector search

These terms overlap but are not identical. **Dense retrieval** describes the learned representation and retrieval method. **Vector search** describes the operation of finding nearby vectors and can be used for non-semantic data such as image features, product attributes, or recommendations. **Semantic search** describes the user-facing goal of matching meaning rather than exact wording and may use dense retrieval, learned sparse retrieval, reranking, or a hybrid system.

A vector database is also not a requirement. Small collections can use an in-memory matrix or a library such as Faiss. A vector database becomes useful when an application needs persistence, metadata filtering, replication, access control, distributed indexing, or operational management at scale.

## Dense retrieval architectures

| Architecture | Representation and scoring | Strength | Main cost |
| --- | --- | --- | --- |
| Single-vector bi-encoder | One query vector and one vector per document or passage | Fast retrieval and compact indexes | A single vector can lose token-level detail |
| Multi-vector or late interaction | Multiple token-level vectors with a lightweight interaction such as ColBERT's MaxSim | Preserves finer-grained matches | Larger indexes and more scoring work |
| Cross-encoder reranker | Jointly encodes each query-document pair | Strong relevance judgments | Too expensive to score an entire large corpus, so it normally reranks candidates |

Dense Passage Retrieval (DPR) helped establish the modern bi-encoder pattern for open-domain question answering. It uses separate BERT-based query and passage encoders trained so relevant pairs receive higher dot-product scores than irrelevant pairs. ColBERT keeps independently computed document representations but stores token-level vectors and applies late interaction, trading more storage and computation for finer matching.

## Dense retrieval vs sparse retrieval

| Property | Dense retrieval | Sparse retrieval such as BM25 |
| --- | --- | --- |
| Representation | Learned, mostly non-zero vectors with hundreds or thousands of dimensions | Very high-dimensional vectors dominated by zero values |
| Best-known strength | Paraphrases, concepts, multilingual or cross-modal similarity | Exact terms, names, identifiers, rare words, and transparent lexical matches |
| Training | Usually depends on a pretrained or task-trained encoder | Traditional BM25 needs no neural training |
| Index | Vector or nearest-neighbor index | Inverted index |
| Failure pattern | Can retrieve a semantically related passage that lacks the exact fact | Can miss a relevant passage that uses different wording |
| Score meaning | Model- and metric-specific similarity | Corpus-dependent lexical relevance |

Neither method wins every workload. The BEIR benchmark found BM25 to be a robust zero-shot baseline and showed that dense retrievers trained in one domain can generalize poorly to another. Production search systems therefore often use **hybrid retrieval**: combine dense and sparse candidates or scores, then optionally rerank the merged set.

## How dense retrievers are trained

Most dense retrievers use contrastive learning. Training data supplies a query, a relevant document called a positive, and irrelevant documents called negatives. The loss raises the positive score and lowers the negative scores. Other examples in the same training batch can act as in-batch negatives, which makes large batches useful.

Negative selection matters. Random negatives may be too easy and teach little. Hard negatives—documents that look plausible but are not relevant—force the model to learn finer distinctions. They can come from BM25, an earlier dense retriever, or mined model errors. Bad negatives can also damage training when they are actually relevant but unlabeled.

## Exact search, ANN indexes, and similarity metrics

Exact search compares the query with every stored vector and returns the true nearest neighbors. Its cost grows with the collection, so large systems commonly use approximate nearest neighbor (ANN) methods such as graph-based HNSW or inverted-file indexes with product quantization. ANN indexes trade some recall for lower latency or memory use.

The similarity function must match model training and vector normalization. Cosine similarity compares direction. Dot product includes vector magnitude unless embeddings are normalized. Euclidean distance measures geometric distance. For unit-normalized vectors, cosine ranking and dot-product ranking are equivalent. Changing a database setting without checking the model's specification can silently worsen retrieval.

## Dense retrieval in RAG

In RAG, dense retrieval selects passages before a language model writes an answer. The retriever defines the evidence the generator can see, so retrieval mistakes become answer mistakes. A strong language model cannot cite a missing document, and a high similarity score does not prove that a passage supports the answer.

Useful RAG pipelines usually combine chunking, metadata filters, query rewriting, top-k retrieval, deduplication, and reranking. They may retrieve more candidates than the context window ultimately receives. Index freshness matters as much as model quality: changed or deleted documents must be re-embedded or removed, and switching embedding models generally requires rebuilding the document index because vectors from different models are not automatically comparable.

## Limitations and common failure modes

- **Domain shift:** an encoder trained on web questions may perform poorly on legal, medical, code, or company-specific language.
- **Exact-match weakness:** product codes, error messages, names, dates, and rare entities may be better handled by lexical retrieval.
- **Embedding compression:** one vector can blur separate claims or topics inside a long document.
- **Chunking errors:** the relevant sentence may be separated from the context needed to understand it.
- **Similarity is not evidence:** nearby vectors can be topically related while contradicting the query or lacking the requested fact.
- **Operational drift:** stale embeddings, mixed model versions, incorrect normalization, or mismatched distance metrics can quietly reduce quality.
- **Cost and memory:** encoding a large corpus and storing high-dimensional vectors can be expensive, especially for multi-vector systems.

## How to evaluate dense retrieval

Evaluate on real queries with relevance judgments from the target domain. **Recall@k** asks whether relevant material appears in the first k results. **MRR** rewards placing the first relevant result near the top. **nDCG@k** handles graded relevance and rank. Precision, success rate, and hit rate can also be useful depending on the product.

Quality metrics should be reported alongside query latency, indexing time, memory, storage, update cost, and ANN recall. For RAG, also measure downstream grounded answer quality and citation support. A retriever that improves a public benchmark but misses your users' identifiers or current documents is not the better production system.

## Bottom line

Dense retrieval turns meaning into geometry: encode the query and corpus into a shared vector space, then search for nearby representations. It is powerful for semantic matching but is not a universal replacement for keywords. The most reliable systems choose the encoder, chunking, index, similarity metric, hybrid strategy, and evaluation set as one connected retrieval design.
$body$::text AS body
)
UPDATE content_items AS item
SET
  title = 'Dense Retrieval',
  summary = 'A neural search method that maps queries and documents into learned vector embeddings, then retrieves the nearest matches by semantic similarity.',
  body = dense_retrieval.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', dense_retrieval.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Dense Passage Retrieval for Open-Domain Question Answering', 'url', 'https://arxiv.org/abs/2004.04906'),
    jsonb_build_object('title', 'ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT', 'url', 'https://arxiv.org/abs/2004.12832'),
    jsonb_build_object('title', 'BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models', 'url', 'https://arxiv.org/abs/2104.08663'),
    jsonb_build_object('title', 'Faiss documentation', 'url', 'https://faiss.ai/')
  ),
  metadata = COALESCE(item.metadata, '{}'::jsonb) || jsonb_build_object(
    'category', 'Language, Vision & Retrieval',
    'relatedTerms', jsonb_build_array('sparse-retrieval', 'embedding', 'vector-database', 'semantic-search', 'hybrid-search', 'bi-encoder', 'retrieval-augmented-generation', 'reranking'),
    'analogy', 'Dense retrieval is like asking a librarian for books about an idea rather than books containing the exact words you used.',
    'seoDescription', 'Dense retrieval maps queries and documents into embeddings for semantic search. Learn how bi-encoders, vector indexes, hybrid search, and RAG work.',
    'seoKeywords', jsonb_build_array('what is dense retrieval', 'dense retrieval explained', 'dense retrieval vs sparse retrieval', 'dense retrieval vs semantic search', 'dense retrieval in RAG', 'dense passage retrieval', 'bi-encoder retrieval', 'vector search', 'approximate nearest neighbor search', 'hybrid search', 'ColBERT late interaction')
  ),
  published_at = DATE '2026-08-31',
  updated_at = NOW()
FROM dense_retrieval
WHERE item.kind = 'glossary'
  AND item.slug = 'dense-retrieval'
  AND item.parent_slug = '';
