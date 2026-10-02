-- Rewrite the bi-encoder glossary page into a full explanation of the
-- architecture.
--
-- The seeded row (seed:glossary/bi-encoder, 1167 chars, no sources, no SEO
-- description, empty seoKeywords, five relatedTerms) covered only the bare
-- definition, the offline-precomputation advantage, the cross-encoder
-- tradeoff, and a DPR/sentence-transformers mention. Everything a reader
-- searching "bi-encoder vs cross-encoder" or "how are bi-encoders trained"
-- needed was missing, and the page had no references at all.
--
-- The new body keeps the same opening claim and then covers:
--
--   1. "The two-tower structure" - shared vs separate weights, vector pooling,
--      cosine vs dot product.
--   2. "Why it is fast" - precomputed document vectors, one query-side encode,
--      D @ q, and HNSW/IVF index lookup.
--   3. "How bi-encoders are trained" - contrastive objective, in-batch,
--      hard, and cross-device negatives, DPR as the reference recipe.
--   4. "Where the single vector leaks" - multi-intent queries, long-document
--      compression, near-duplicate corpora, hubness, and late interaction
--      (ColBERT) plus sparse models as the structural responses.
--   5. "Bi-encoder vs the alternatives" - side-by-side table of the four
--      retrieval architectures with vectors-per-document and query cost.
--   6. "Putting it together" - recall caps end-to-end quality, measure
--      recall@k before tuning the reranker, normalize before comparing, and
--      fine-tune the encoder on in-domain pairs.
--
-- Sources consulted:
-- - Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question
--   Answering" (EMNLP 2020)
--   (https://arxiv.org/abs/2004.04906)
-- - Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese
--   BERT-Networks" (EMNLP 2019)
--   (https://arxiv.org/abs/1908.10084)
-- - Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via
--   Contextualized Late Interaction over BERT" (SIGIR 2020)
--   (https://arxiv.org/abs/2004.12832)
-- - Johnson, Douze & Jégou, "Billion-scale similarity search with GPUs"
--   (https://arxiv.org/abs/1702.08734)
-- - Malkov & Yashunin, "Hierarchical Navigable Small World graphs" (HNSW)
--   (https://arxiv.org/abs/1603.09320)
-- - Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
--   of Information Retrieval Models" (2021)
--   (https://arxiv.org/abs/2104.08663)
--
-- Every inline link and every relatedTerms slug resolves against a live
-- glossary row (337 rows at the time of writing). late-interaction and
-- colbert have no pages, so late interaction is described in prose and
-- ColBERT is cited rather than linked.
--
-- The body carries no "## References & Resources" or "## Related Terms"
-- heading because app/glossary/[term]/page.tsx renders both from the
-- `references` and `relatedTerms` fields.
--
-- published_at is left alone: this rewrites the page, it does not republish
-- it. lastUpdated is derived from updated_at in lib/glossary.ts, so
-- updated_at = NOW() is what surfaces on the page and in the sitemap.

WITH bi_encoder_entry AS (
  SELECT $body$A **bi-encoder** is a retrieval architecture that encodes a query and a document *separately*, producing one vector each, then scores the pair with a cheap similarity function. No forward pass ever sees both texts at once. That single constraint is why bi-encoders are the default first stage of modern search and retrieval systems.

## The two-tower structure

A bi-encoder has two towers, usually two copies of the same transformer weights:

- The **query tower** maps the query text to a single fixed-size vector.
- The **document tower** maps each passage to a single fixed-size vector.

Relevance is then a vector operation:

```
score(q, d) = cos(q, d)  or  dot(q, d)
```

Because each side is encoded independently, a query and a document can never attend to each other while encoding. Each vector has to capture, on its own, everything that matters about that text.

| Choice | Common setting | Why |
| --- | --- | --- |
| Shared weights | One model used for both towers | Symmetric tasks, halves the model count |
| Separate towers | One model for queries, one for documents | Asymmetric tasks where queries and passages look nothing alike |
| Vector pooling | Mean, CLS, or last token | Decides which tokens get the most weight in the vector |
| Distance | Cosine or dot product | Dot product only equals cosine on L2-normalized vectors |

## Why it is fast

The asymmetry that costs accuracy buys the speed that makes retrieval possible at all.

Document [embeddings](/glossary/embedding) are computed **once**, offline, and stored in a [vector database](/glossary/vector-database). At query time the pipeline encodes only the query, one forward pass, then runs approximate nearest neighbor search over the prebuilt index. There is no per-candidate model call, which is what separates retrieval from [reranking](/glossary/reranking).

Scaling that out is mostly arithmetic. With a document matrix `D` of shape `N × d`, scoring every document is a single matrix-vector product, `D @ q`. An HNSW or IVF index gets that down to a handful of vector comparisons over millions of documents, usually in under a millisecond.

This is why [dense retrieval](/glossary/dense-retrieval) systems work at web scale and cross-encoders do not: the expensive computation moved from query time to index time.

## How bi-encoders are trained

Bi-encoders are usually trained with a contrastive objective. A query and a known-relevant document form a positive pair, and everything else in the batch is a negative. The model is trained to pull positive pairs together in vector space and push the rest apart.

- **In-batch negatives** are cheap negatives drawn from other examples in the batch.
- **Hard negatives** are documents retrieved by a first pass but judged irrelevant, which is where most of the quality gain comes from.
- **Cross-device negatives** are sampled across GPUs to make the batch large enough to be useful.

Dense Passage Retrieval (DPR) is the standard reference point for this recipe, and Sentence-BERT is the standard reference point for the sentence-level version. Common model families include `sentence-transformers` models, the BGE and GTE families, and commercial embedding APIs. They are used **frozen** at inference time, so nothing is [fine-tuned](/glossary/fine-tuning) per query.

## Where the single vector leaks

The single-vector constraint is the whole trade. One vector has to summarize an entire document, so distinctions that matter for relevance can collapse.

- **Multi-vector queries.** A query like "how do I fix billing?" carries several intents that fight for one point in the space.
- **Long documents.** Compression cost grows with length, so a long document is compressed harder than a short one.
- **Narrow topical differences.** In a corpus of near-identical API docs, "retrieval" and "generation" can land almost on top of each other.
- **Hubness and popularity bias.** In high-dimensional spaces some vectors are close to far more queries than others, which quietly promotes generic passages.

The main structural response is **late interaction**. ColBERT-style models emit one vector per token and score with MaxSim over token-level matches, keeping most of the speed while recovering a lot of the lost precision. Sparse models like [BM25](/glossary/bm25) and SPLADE attack the same problem from the other side by matching literal terms, which is also the basis of [hybrid search](/glossary/hybrid-search).

## Bi-encoder vs the alternatives

| Architecture | Query and document | Vectors per document | Cost per query | Typical role |
| --- | --- | --- | --- | --- |
| Bi-encoder (dense) | Encoded separately | 1 | 1 encode + ANN search | First-stage retrieval |
| Late interaction | Encoded separately | One per token | 1 encode + MaxSim | Higher-recall retrieval |
| [Sparse retrieval](/glossary/sparse-retrieval) | Indexed as term weights | About document length | Sparse dot product | Lexical first stage, hybrids |
| [Cross-encoder](/glossary/cross-encoder) | Concatenated, full cross-attention | 0, no index | One forward pass per candidate | Reranking top 20 to 100 |

## Putting it together

In a production pipeline the two stages divide the work cleanly. The bi-encoder buys recall cheaply by casting the problem as vector similarity. The cross-encoder buys precision by actually reading the pair.

The important consequence is that **first-stage recall caps end-to-end quality**. A [reranker](/glossary/reranking) can only reorder what the retriever returned. Measure recall@k on the first stage before tuning anything downstream, or you will spend a week improving a reranker that was never given the right answer.

Two details bite more often than people expect. First, **normalize before you compare**: unnormalized dot products let long or high-magnitude vectors dominate every ranking. Second, a general-purpose model is a baseline, not a ceiling, since [fine-tuning](/glossary/fine-tuning) the encoder on your own query-document pairs is usually the highest-leverage improvement available. [FAISS](/glossary/faiss) is where a first working index usually gets built.
$body$::text AS body
)
UPDATE content_items AS item
SET
  body = bi_encoder_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', bi_encoder_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Karpukhin et al: Dense Passage Retrieval for Open-Domain Question Answering (2020)', 'url', 'https://arxiv.org/abs/2004.04906'),
    jsonb_build_object('title', 'Reimers and Gurevych: Sentence-BERT - Sentence Embeddings using Siamese BERT-Networks (2019)', 'url', 'https://arxiv.org/abs/1908.10084'),
    jsonb_build_object('title', 'Khattab and Zaharia: ColBERT - Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (2020)', 'url', 'https://arxiv.org/abs/2004.12832'),
    jsonb_build_object('title', 'Johnson, Douze and Jégou: Billion-scale similarity search with GPUs', 'url', 'https://arxiv.org/abs/1702.08734'),
    jsonb_build_object('title', 'Malkov and Yashunin: Hierarchical Navigable Small World graphs - HNSW (2016)', 'url', 'https://arxiv.org/abs/1603.09320'),
    jsonb_build_object('title', 'Thakur et al: BEIR - A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models (2021)', 'url', 'https://arxiv.org/abs/2104.08663')
  ),
  metadata = jsonb_set(
    jsonb_set(
      jsonb_set(
        COALESCE(item.metadata, '{}'::jsonb),
        '{seoKeywords}',
        jsonb_build_array(
          'what is a bi-encoder',
          'bi encoder explained',
          'bi-encoder vs cross-encoder',
          'two tower retrieval model',
          'bi-encoder architecture',
          'dense retrieval bi-encoder',
          'late interaction colbert',
          'embedding model for retrieval',
          'vector similarity search',
          'ANN search HNSW',
          'contrastive training embeddings',
          'in-batch negatives',
          'hard negatives retrieval',
          'multi-vector retrieval',
          'first stage retrieval',
          'reranking pipeline',
          'MaxSim scoring',
          'why bi-encoders are fast',
          'bi-encoder vector database',
          'bi-encoder fine tuning'
        ),
        true
      ),
      '{seoDescription}',
      '"Bi-encoder explained: the two-tower architecture behind dense retrieval, why document embeddings are precomputed, contrastive training with hard negatives, and how it differs from cross-encoders and late interaction."'::jsonb,
      true
    ),
    '{relatedTerms}',
    jsonb_build_array(
      'cross-encoder',
      'embedding',
      'dense-retrieval',
      'reranking',
      'vector-database',
      'sparse-retrieval',
      'semantic-search',
      'hybrid-search',
      'faiss',
      'vectorless-retrieval'
    ),
    true
  ),
  updated_at = NOW()
FROM bi_encoder_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'bi-encoder'
  AND item.parent_slug = '';
