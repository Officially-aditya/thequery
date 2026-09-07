WITH glossary_updates(slug, title, summary, body, sources, metadata) AS (
  VALUES
  (
    'knn',
    'KNN (K-Nearest Neighbors)',
    'A supervised learning method that predicts a label or value from the k closest training examples under a chosen distance metric.',
    $knn$
KNN, short for **K-Nearest Neighbors**, is a supervised machine-learning method that predicts from the training examples closest to a new input. For classification, it usually assigns the most common class among the **k** neighbors. For regression, it usually averages their target values.

KNN is called **instance-based** or **lazy learning** because ordinary training does not fit a large set of model coefficients. It stores the reference examples and does most of its work when a prediction is requested. “Non-parametric” does not mean it has no settings or costs. The choice of k, distance metric, feature representation, weighting rule, and search algorithm defines the model, while retaining the training set creates a real memory cost.

## How KNN works

For each new query point, KNN follows four basic steps:

1. Represent the query and training examples in the same feature space.
2. Calculate or retrieve their distances from the query.
3. Select the k closest examples.
4. Combine their labels or target values into a prediction.

For a classification problem with k = 5, suppose the closest labels are cat, cat, dog, cat, and dog. Uniform voting predicts cat. A distance-weighted version may predict dog if the two dog examples are much closer than the three cat examples.

For regression, uniform KNN commonly returns the arithmetic mean of the neighbors' values. Distance-weighted regression gives closer observations more influence. Other aggregation rules, such as a median, can be used when the implementation and evaluation protocol state them.

## KNN classification vs KNN regression

| Variant | Neighbor information | Typical prediction | Common metric |
| --- | --- | --- | --- |
| KNN classification | Discrete class labels | Majority or weighted vote | Accuracy, precision, recall, F1, ROC-AUC |
| KNN regression | Continuous target values | Mean or distance-weighted mean | MAE, RMSE, R² |
| Radius neighbors | All examples within a fixed distance | Vote or aggregate within the radius | Task-specific classification or regression metric |

Radius-based methods adapt the number of neighbors to local density. They can work better when one region is crowded and another is sparse, but a query may have no neighbor inside the selected radius.

## Choosing k

The value of k controls how local and smooth the prediction is:

- **Small k** creates flexible boundaries and can capture fine local structure, but it is sensitive to noise, mislabeled examples, and outliers.
- **Large k** produces smoother predictions with lower variance, but it can erase small classes or local patterns and drift toward the global majority.
- **k = 1** copies the nearest training target and can fit the training data extremely closely.

There is no universally best k. Select it with cross-validation using the metric and data split that match the intended use. Odd values can reduce two-class voting ties, but class count, sample count, weights, and tie-breaking still matter.

Do not choose k on the final test set. Feature scaling, dimensionality reduction, metric selection, and k should be fitted or selected inside the validation pipeline to avoid leakage.

## Distance metrics

| Metric | Basic idea | Typical use or caution |
| --- | --- | --- |
| Euclidean distance | Straight-line distance | Common for continuous, similarly scaled numeric features |
| Manhattan distance | Sum of absolute coordinate differences | Can be less dominated by a single coordinate than squared Euclidean distance |
| Minkowski distance | General family containing Manhattan and Euclidean distance | The order parameter changes neighborhood shape |
| Cosine distance | Compares vector direction rather than raw magnitude | Common for text embeddings and high-dimensional representations |
| Hamming distance | Counts positions that differ | Binary or categorical encodings |
| Mahalanobis distance | Accounts for feature scale and covariance | Requires a stable covariance estimate or learned transformation |

The metric defines what “near” means. A geographic distance, cosine distance between embeddings, and Euclidean distance between standardized measurements produce different neighborhoods even on the same rows.

## Why feature scaling matters

Distance is dominated by features with larger numerical ranges. If age ranges from 18 to 90 while annual income ranges from 20,000 to 500,000, raw Euclidean distance will mostly reflect income. Standardization, min-max scaling, robust scaling, or a domain-specific transformation can make features comparable.

Scaling must be learned from the training fold and then applied to validation, test, and production data. Fitting a scaler on all available data leaks information from the evaluation set.

Categorical features need a meaningful representation. Arbitrarily encoding red, green, and blue as 1, 2, and 3 invents an ordering and spacing that may not exist. Missing values also require deliberate handling because many distance metrics cannot compare incomplete vectors directly.

## Uniform vs distance-weighted KNN

Uniform KNN gives every selected neighbor the same vote or regression weight. Distance weighting gives closer points more influence, often with a weight related to the inverse distance.

Weighting can reduce the effect of relatively distant neighbors, but it introduces edge cases. An exact duplicate has zero distance, so implementations need a defined zero-distance rule. Noisy or mislabeled duplicates can dominate the result. Weighting should be validated rather than assumed to improve accuracy.

## KNN search cost and indexes

With **n** stored examples and **d** features, brute-force search compares the query with every example and costs roughly O(nd) per query, plus selection of the nearest results. Storage is roughly O(nd). Training appears cheap because most work has moved into prediction.

KD trees and ball trees can reduce search work on suitable low-dimensional data, but their advantage often fades as dimensionality rises. Large or high-dimensional deployments may use approximate nearest-neighbor indexes such as HNSW, IVF, or product quantization to retrieve candidate neighbors faster.

Using an ANN index changes the search stage, not the KNN prediction rule. The returned neighbors may omit some true nearest examples, so classification or regression quality must be evaluated end to end alongside retrieval recall and latency.

## KNN vs nearest-neighbor search vs ANN vs k-means

| Term | Purpose | Uses labels? | Output |
| --- | --- | --- | --- |
| KNN classifier or regressor | Predict a target from nearby training examples | Yes | Class or numeric value |
| Exact k-nearest-neighbor search | Find the true k closest stored items | No | Neighbor IDs and distances |
| Approximate nearest-neighbor search | Find likely near neighbors faster at scale | No | Approximate neighbor IDs and distances |
| k-means clustering | Partition data around k learned centroids | No | Cluster assignments and centroids |

The letter k means different things in KNN and k-means. In KNN it is the number of neighbors consulted for one query. In k-means it is the number of clusters learned from the dataset.

ANN is also an overloaded abbreviation. In vector-search discussions it usually means approximate nearest neighbor. In other machine-learning writing it can mean artificial neural network. Context matters.

## The curse of dimensionality

As the number of dimensions grows, data becomes sparse and the contrast between near and far distances can shrink. Many irrelevant features create noise, and a fixed sample count covers the space poorly. This is one reason raw KNN can perform badly on high-dimensional inputs.

Feature selection, dimensionality reduction, metric learning, or learned embeddings can help, but they change the representation on which proximity is judged. An embedding may make semantic neighbors useful even though the original raw dimensions were not.

## Class imbalance, ties, and probability estimates

A majority class occupies more of the training set and can dominate large neighborhoods. Useful responses include class-aware weighting, resampling, distance weighting, per-class metrics, or choosing a smaller local neighborhood. Evaluate minority-class recall rather than relying only on overall accuracy.

Classification probabilities are often estimated from the fraction or normalized weight of neighboring classes. These values are local vote shares, not automatically calibrated probabilities. Calibration should be measured separately if downstream decisions depend on confidence.

Ties require a deterministic rule, such as class order, nearest individual neighbor, or a documented secondary score. Different libraries can resolve the same tie differently.

## When KNN works well

KNN is useful when similar inputs genuinely have similar targets, the feature space and metric are meaningful, the dataset fits comfortably in memory, and low-latency prediction is not the only priority. It is a strong interpretable baseline because a prediction can be explained by showing the retrieved examples.

Common uses include small tabular classification, local regression, recommendation candidates, anomaly scores, missing-value methods, pattern recognition, and evaluation of learned embeddings.

## Limitations

- Prediction can be slow and memory-heavy because reference examples must be searched and retained.
- Irrelevant or badly scaled features distort neighborhoods.
- High dimensionality weakens distance contrast and indexing performance.
- Noise, duplicates, label errors, and outliers can dominate small neighborhoods.
- Large k can erase minority classes or local structure.
- Ordinary KNN does not learn which features or examples should matter unless preprocessing or metric learning does that work.
- Updating the dataset changes future predictions and may require rebuilding a search index.

KNN can also expose sensitive training examples through neighbor explanations or membership attacks. Access control and privacy review matter when stored records contain personal data.

## Bottom line

KNN predicts by consulting nearby labeled examples. Its simplicity is real, but so is its dependence on representation, scaling, distance, k, weighting, and search quality. Keep it distinct from k-means clustering and from ANN retrieval, which can accelerate neighbor lookup without defining the final prediction.
$knn$::text,
    jsonb_build_array(
      jsonb_build_object('title', 'Nearest Neighbor Pattern Classification — Cover and Hart (1967)', 'url', 'https://doi.org/10.1109/TIT.1967.1053964'),
      jsonb_build_object('title', 'scikit-learn Nearest Neighbors user guide', 'url', 'https://scikit-learn.org/stable/modules/neighbors.html'),
      jsonb_build_object('title', 'The Elements of Statistical Learning', 'url', 'https://hastie.su.domains/ElemStatLearn/')
    ),
    jsonb_build_object(
      'category', 'Foundations',
      'relatedTerms', jsonb_build_array('machine-learning', 'vector-database', 'embedding', 'cosine-similarity', 'approximate-nearest-neighbor', 'hnsw'),
      'analogy', 'KNN is like asking the closest comparable cases what happened to them, then using their vote or average to predict what will happen in the new case.',
      'seoDescription', 'KNN predicts from nearby training examples. Learn how k, distance metrics, feature scaling, weighting, dimensionality, and ANN indexes affect its results.',
      'seoKeywords', jsonb_build_array('what is KNN', 'K-nearest neighbors explained', 'KNN classification', 'KNN regression', 'how KNN works', 'how to choose k in KNN', 'KNN distance metrics', 'KNN feature scaling', 'weighted KNN', 'KNN vs k-means', 'KNN vs ANN', 'curse of dimensionality KNN', 'KNN algorithm complexity')
    )
  ),
  (
    'approximate-nearest-neighbor',
    'Approximate Nearest Neighbor (ANN)',
    'A family of search methods that returns likely nearest items without exhaustively comparing every stored vector, trading exactness for speed, memory, or scale.',
    $ann$
Approximate Nearest Neighbor, abbreviated **ANN**, is a family of search methods that finds items likely to be closest to a query without guaranteeing the exact top results from an exhaustive scan. ANN indexes trade some retrieval accuracy for lower latency, higher throughput, reduced memory, or the ability to search collections too large for brute force.

In vector-search contexts, ANN means **approximate nearest neighbor**, not artificial neural network. The query and stored items are usually vectors produced from text, images, audio, products, users, or other data. “Nearest” is defined by a configured distance or similarity function.

## Exact vs approximate nearest-neighbor search

Exact search compares the query with every stored vector, or uses an exact index that still guarantees the true nearest results. A flat scan is simple and can be fast for small datasets, batches, or GPUs, but its work grows with the number and dimension of vectors.

Approximate search avoids much of that work. It may navigate a graph, search selected clusters, compare compressed codes, use hash buckets, or read a disk-aware index. Because it skips candidates or stores a compressed representation, it can miss a true nearest neighbor.

| Property | Exact search | ANN search |
| --- | --- | --- |
| Result guarantee | Returns the true nearest items under the implemented metric | Returns likely nearest items with measured recall |
| Query work | Often scans all vectors | Searches a reduced candidate set or index structure |
| Tuning | Mostly metric, batching, and hardware choices | Adds accuracy-speed-memory tuning parameters |
| Best fit | Smaller collections, ground-truth evaluation, or cases requiring exact order | Large collections and latency-sensitive similarity search |
| Main risk | Excessive latency or compute | Missing relevant neighbors because of approximation |

Approximate does not necessarily mean poor quality. A well-tuned index can achieve high recall on a particular dataset. It also does not guarantee a particular recall percentage or time complexity. Results depend on vector distribution, dimensionality, metric, hardware, index family, parameters, filters, updates, and workload.

## How ANN search works

A typical ANN system has an indexing phase and a query phase:

1. Generate vectors with one consistent embedding or feature pipeline.
2. Choose a distance metric that matches the representation.
3. Build an index that organizes or compresses the vectors.
4. Encode a query with the compatible pipeline.
5. Use the index to produce a candidate set.
6. Optionally calculate exact distances on those candidates and rerank them.
7. Return the top k IDs, distances, and associated metadata.

The ANN index does not create semantic meaning. The embedding model and data representation determine which concepts are near each other. ANN only accelerates search within that geometry.

## Major ANN index families

| Index family | Core idea | Strength | Main tradeoff |
| --- | --- | --- | --- |
| HNSW | Navigates a multilayer proximity graph from coarse to fine neighborhoods | High recall and low query latency for many in-memory workloads | Graph memory overhead and nontrivial build cost |
| IVF | Trains centroids, assigns vectors to inverted lists, and probes selected nearby lists | Adjustable candidate pruning and good composition with compression | Requires training and can miss neighbors in unprobed lists |
| Product quantization | Splits vectors into subvectors and stores compact codebook IDs | Large memory reduction and fast approximate distance calculations | Compression reduces distance fidelity |
| IVF-PQ | Uses coarse IVF partitions plus product-quantized vectors | Scales to large collections with controlled memory | More parameters and often needs exact reranking for best quality |
| Locality-sensitive hashing | Hashes nearby points into likely shared buckets | Theoretical guarantees under specific assumptions and simple lookup | Can require many tables or memory for strong recall |
| Tree-based indexes | Recursively partition the vector space | Effective for some low- or moderate-dimensional datasets | Performance often degrades in high dimensions |
| Disk-aware graphs such as DiskANN | Keeps a graph-oriented index on SSD with selected data in memory | Supports collections larger than RAM | Storage latency, caching, build, and hardware become central |

No index family wins every dataset. HNSW may be strong at high recall in memory, IVF-PQ may be preferable under strict memory limits, a flat GPU scan may beat indexing for large batches, and disk-based methods may be required when the corpus exceeds RAM.

## HNSW parameters

HNSW builds hierarchical graph layers whose upper levels provide long-range navigation and whose lower level contains the full dataset. Common tuning controls include:

- **M**, the approximate number of graph connections retained per node, affects memory, build time, and reachability.
- **efConstruction**, the candidate-list size used while building the graph, generally trades a slower build for a stronger index.
- **efSearch**, the candidate-list size at query time, generally trades latency for recall.

The exact meanings and limits vary by implementation. Increasing a parameter does not make results free: higher recall consumes more CPU, memory bandwidth, or time.

## IVF and product quantization parameters

An inverted-file index partitions vectors around trained centroids. At query time, **nprobe** controls how many partitions are searched. Probing more lists usually improves recall and costs more work.

Product quantization compresses segments of a vector with learned codebooks. The number of subquantizers, bits per code, training sample, and optional residual or rotation transform affect memory and distortion. Compressed candidates can be reranked using original vectors when higher accuracy justifies additional storage and computation.

## Distance metrics and normalization

| Metric | Search objective | Important detail |
| --- | --- | --- |
| Euclidean or L2 distance | Minimize geometric distance | Sensitive to scale and vector magnitude |
| Inner product | Maximize the dot product | Larger vector norms can influence rank |
| Cosine similarity | Maximize directional similarity | Commonly implemented as inner product after unit normalization |
| Hamming distance | Minimize differing bits | Used for binary codes rather than ordinary floating-point embeddings |

The index metric must match how the embedding model was trained and how stored vectors were normalized. Unit-normalized vectors make cosine and inner-product rankings equivalent, but only if both queries and documents use the same normalization. Mixing embedding models or dimensions requires re-embedding and rebuilding the index.

## Measuring ANN quality

ANN is a multi-objective system problem. Evaluate at the operating point the application will actually use:

| Metric | What it measures |
| --- | --- |
| Recall@k | Fraction of the exact top-k neighbors recovered by the approximate search |
| Queries per second | Throughput under a stated concurrency and batch size |
| p50, p95, and p99 latency | Typical and tail response times |
| Index build time | Cost of creating or rebuilding the index |
| Memory and disk footprint | Storage required for vectors, graph edges, codes, and metadata |
| Update performance | Cost and quality impact of inserts, deletes, and compaction |
| Filtered recall | Retrieval quality after applying metadata constraints |
| End-to-end task quality | Whether the retrieved items improve search, recommendation, or RAG outcomes |

Recall@k requires exact ground-truth neighbors for the same query set and metric. A statement such as “99% accurate” is incomplete unless it specifies k, dataset, metric, filters, index parameters, and how recall was calculated.

ANN-Benchmarks commonly plots recall against queries per second and also reports build time and index size. Those results are useful for comparing implementations on the listed hardware and datasets, not universal rankings for every production corpus.

## ANN in vector databases and RAG

Vector databases commonly wrap ANN indexes with persistence, metadata filters, replication, access control, backups, ingestion, and distributed operation. ANN is the search algorithm or index layer. A vector database is the larger operational system, and a small application can use an ANN library without a database.

In retrieval-augmented generation, ANN retrieves passages whose embeddings are near the query. The result can then be combined with lexical search, filters, or a reranker before context reaches the language model. ANN recall is only one quality layer. Weak chunking, stale data, a mismatched embedding model, or a poor reranker can hurt answer quality even when the index closely matches exact vector search.

Hybrid search is useful when exact names, product IDs, error codes, and rare terms matter. Dense ANN search captures semantic similarity, while sparse retrieval preserves lexical evidence.

## Metadata filtering

Filters can be applied before, during, or after vector search. Post-filtering a small ANN result set may return too few items when most candidates fail the filter. Pre-filtering can make the remaining subset too small or fragmented for the existing index. Filter-aware indexes integrate attributes into traversal or maintain partitions for common constraints.

Evaluate filtered workloads using realistic filter selectivity. An index tuned on unfiltered queries can lose recall or latency when asked for one tenant, language, date range, or permission group.

## Updates, deletion, and index freshness

Some ANN structures support incremental insertion well but handle deletion through tombstones or periodic rebuilding. Heavy updates can degrade graph quality, cluster balance, compression assumptions, or storage locality. Production evaluation should include ingestion rate, deletion correctness, compaction, rebuild duration, and behavior while an index is being replaced.

Changing the embedding model creates a new vector space. Old and new vectors should not normally share one index, even when their dimensions match. Re-embed the corpus, build and validate a new index, then switch traffic deliberately.

## ANN vs KNN

**KNN classification or regression** is a prediction rule that combines labels or values from k nearby training examples. **ANN search** is a retrieval strategy for finding likely nearby items quickly. KNN can use exact search or an ANN index, and ANN can support many tasks that do not make a supervised prediction.

| Question | KNN | ANN search |
| --- | --- | --- |
| Primary goal | Predict from labeled neighbors | Retrieve similar stored items |
| Approximation required? | No | Yes by definition |
| Produces a class or value? | Yes | No, it returns neighbors and scores |
| Typical scale issue | Prediction cost over the training set | Search latency, memory, and index maintenance |

## When exact search is better

Use exact search when the collection is small, queries can be efficiently batched, every true neighbor matters, or exact results are needed to evaluate an ANN index. Brute-force GPU search can also be competitive when arithmetic throughput and batching outweigh index traversal overhead.

Start with an exact baseline. Add ANN only when measured latency, throughput, memory, or scale requires it. Approximation creates tuning and operational complexity that a small corpus may not need.

## Common mistakes

- Assuming every ANN query runs in O(log n), regardless of algorithm or data
- Publishing latency without recall at the same parameter setting
- Comparing indexes on different hardware, datasets, metrics, or batch sizes
- Using cosine search without consistent normalization
- Mixing vectors from different embedding models
- Ignoring metadata-filter selectivity and permission filtering
- Measuring vector recall but not downstream relevance or answer quality
- Forgetting build time, memory, deletes, compaction, and recovery
- Treating a similarity score as a calibrated probability or proof of factual relevance

## Bottom line

ANN search avoids comparing a query with every stored vector by navigating, partitioning, hashing, or compressing the search space. It makes large-scale vector retrieval practical, but there is no free or universal speedup. Choose and tune the index against exact ground truth, report recall with latency and resource costs, and validate the complete application rather than the vector index alone.
$ann$::text,
    jsonb_build_array(
      jsonb_build_object('title', 'Efficient and Robust Approximate Nearest Neighbor Search Using HNSW', 'url', 'https://arxiv.org/abs/1603.09320'),
      jsonb_build_object('title', 'Faiss documentation', 'url', 'https://faiss.ai/'),
      jsonb_build_object('title', 'Product Quantization for Nearest Neighbor Search', 'url', 'https://doi.org/10.1109/TPAMI.2010.57'),
      jsonb_build_object('title', 'DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node', 'url', 'https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/'),
      jsonb_build_object('title', 'ScaNN — Google Research', 'url', 'https://github.com/google-research/google-research/tree/master/scann'),
      jsonb_build_object('title', 'ANN-Benchmarks', 'url', 'https://ann-benchmarks.com/')
    ),
    jsonb_build_object(
      'category', 'Language, Vision & Retrieval',
      'relatedTerms', jsonb_build_array('vector-database', 'faiss', 'hnsw', 'embedding', 'dense-retrieval', 'cosine-similarity', 'knn'),
      'analogy', 'ANN search is like using neighborhoods and express routes to find a very close shop without measuring the distance to every shop in the country.',
      'seoDescription', 'Approximate nearest neighbor search finds similar vectors quickly at scale. Learn HNSW, IVF, PQ, recall-latency tradeoffs, filtering, and RAG indexing.',
      'seoKeywords', jsonb_build_array('what is approximate nearest neighbor', 'ANN search explained', 'approximate nearest neighbor algorithm', 'ANN vs KNN', 'exact vs approximate nearest neighbor', 'HNSW vs IVF', 'product quantization', 'ANN vector search', 'ANN index vector database', 'ANN search for RAG', 'recall latency tradeoff', 'cosine similarity ANN', 'filtered vector search', 'Faiss ANN index')
    )
  )
)
UPDATE content_items AS item
SET
  title = glossary_updates.title,
  summary = glossary_updates.summary,
  body = glossary_updates.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', glossary_updates.body)
  ),
  sources = glossary_updates.sources,
  metadata = COALESCE(item.metadata, '{}'::jsonb) || glossary_updates.metadata,
  published_at = DATE '2026-09-07',
  updated_at = NOW()
FROM glossary_updates
WHERE item.kind = 'glossary'
  AND item.slug = glossary_updates.slug
  AND item.parent_slug = '';
