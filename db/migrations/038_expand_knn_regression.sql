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

## KNN regression

KNN regression predicts a continuous number from the target values of the k nearest training examples. It does not fit a global line, surface, or coefficient vector. The prediction at a query is a local summary of nearby observed outcomes.

The usual uniform rule is the arithmetic mean of the neighbors' targets. If a house has five nearest comparables priced at 80, 82, 85, 90, and 200, uniform KNN with k = 5 predicts 107.4. That average is pulled hard by the 200 outlier. A median of those same neighbors predicts 85, which is why some implementations use a median or a truncated mean when the local sample is noisy.

Distance-weighted KNN regression gives closer examples more influence, often with a weight that falls as distance grows, such as the inverse of the distance. Weighting does not remove the need to choose k, and a zero-distance duplicate still needs an explicit rule. If two identical feature vectors have different targets, the nearest-neighbor average is only as honest as the labels.

KNN regression is a local interpolator of the training set. Small k follows local wiggles. Large k approaches a globally smoothed estimate. Unlike linear regression, it does not assume a single slope across the whole domain. Unlike a parametric model, it also does not extrapolate far outside the observed feature range in a principled way. Far from training data, it still returns a combination of the nearest available points.

| Choice | Linear regression | KNN regression |
| --- | --- | --- |
| What is stored | Coefficients | Labeled training examples |
| Shape of the fit | One global function | Local neighborhood summaries |
| Outliers | Can shift the entire fitted trend | Affect nearby queries, and large k as well |
| Extrapolation | Continues the fitted trend | Reuses the nearest observed values |

Use KNN regression when similar inputs have similar numeric targets, the feature space is meaningful after scaling, and you want a prediction that can be inspected by listing the retrieved cases. Prefer a parametric model when you need a compact equation, stable extrapolation, or a fit that stays cheap as the training set grows.

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
      jsonb_build_object('title', 'Nearest Neighbor Pattern Classification, Cover and Hart (1967)', 'url', 'https://doi.org/10.1109/TIT.1967.1053964'),
      jsonb_build_object('title', 'scikit-learn Nearest Neighbors user guide', 'url', 'https://scikit-learn.org/stable/modules/neighbors.html'),
      jsonb_build_object('title', 'scikit-learn KNeighborsRegressor', 'url', 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html'),
      jsonb_build_object('title', 'The Elements of Statistical Learning', 'url', 'https://hastie.su.domains/ElemStatLearn/')
    ),
    jsonb_build_object(
      'category', 'Foundations',
      'relatedTerms', jsonb_build_array('machine-learning', 'vector-database', 'embedding', 'cosine-similarity', 'approximate-nearest-neighbor', 'hnsw'),
      'analogy', 'KNN is like asking the closest comparable cases what happened to them, then using their vote or average to predict what will happen in the new case.',
      'seoDescription', 'KNN predicts a class or number from nearby training examples. Learn KNN regression, k, distance metrics, scaling, weighting, and search indexes.',
      'seoKeywords', jsonb_build_array('what is KNN', 'K-nearest neighbors explained', 'KNN classification', 'KNN regression', 'KNN vs linear regression', 'how KNN works', 'how to choose k in KNN', 'KNN distance metrics', 'KNN feature scaling', 'weighted KNN', 'KNN vs k-means', 'KNN vs ANN', 'curse of dimensionality KNN', 'KNN algorithm complexity')
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
  published_at = DATE '2026-09-08',
  updated_at = NOW()
FROM glossary_updates
WHERE item.kind = 'glossary'
  AND item.slug = glossary_updates.slug
  AND item.parent_slug = '';
