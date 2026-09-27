-- Merge the AdaBoost, XGBoost, and LightGBM glossary pages into the boosting
-- page, then delete the three merged rows.
--
-- The three pages were thin slices of the same concept. A reader searching
-- "what is XGBoost" landed on a page that explained XGBoost and never said how
-- it relates to the family it belongs to, and a reader searching "boosting"
-- got four paragraphs and three outbound links. The merged page keeps the
-- existing opening and the family framing, then covers the mechanism, all four
-- implementations, the tuning surface, the failure modes, and where the family
-- sits against tabular foundation models.
--
-- Sources fetched and verified on September 27 2026:
-- - Freund & Schapire, "A Decision-Theoretic Generalization of Boosting"
--   (https://www.jmlr.org/papers/volume18/15-145.html)
-- - Friedman, "Greedy Function Approximation: A Gradient Boosting Machine"
--   (https://jmlr.org/papers/volume1/friedman00a.html)
-- - Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016
--   (https://dl.acm.org/doi/10.1145/2939672.2939785)
-- - XGBoost parameter documentation
--   (https://xgboost.readthedocs.io/en/stable/parameter.html)
-- - Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
--   NeurIPS 2017
--   (https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
-- - LightGBM parameter documentation
--   (https://lightgbm.readthedocs.io/en/latest/Parameters.html)
-- - LightGBM 4.7.0 release notes
--   (https://github.com/lightgbm-org/LightGBM/releases/tag/v4.7.0)
-- - Prokhorenkova et al., "CatBoost: unbiased boosting with categorical
--   features", NeurIPS 2018
--   (https://proceedings.neurips.cc/paper/14464-catboost-generalized-predictive-categorical-features)
-- - scikit-learn ensemble learning guide
--   (https://scikit-learn.org/stable/modules/ensemble.html)
-- - TabPFN-3 technical report, "A new performance standard", May 2026
--   (https://arxiv.org/abs/2605.13986)
--
-- Body and blocks stay in sync, sources are attached, and metadata gains the
-- merged keyword set plus related terms that all exist as glossary slugs. The
-- adaboost, xgboost, and lightgbm slugs are dropped, so the three inbound
-- relatedTerms entries on the merged row are removed with them.

WITH boosting_entry AS (
  SELECT $body$
Boosting is an ensemble strategy that builds models one at a time, with each new model trained to fix the errors left by the ensemble so far. Unlike bagging (used in [random forests](/glossary/random-forest)), which trains models independently in parallel, boosting is inherently sequential - the order matters because each model learns from the failures of its predecessors.

The core insight is that many weak learners - models only slightly better than random guessing - can be combined into a single strong learner. Each iteration focuses learning on the hardest examples: the data points that previous models got wrong receive higher weight, forcing the next model to pay more attention to them. The final prediction is a weighted combination of everything built along the way.

On tabular data this remains the strongest default for most production problems. Three implementations carry nearly all of the weight today - XGBoost, LightGBM, and CatBoost - and AdaBoost, the 1997 original, still explains the idea better than any of them.

## How a boosting round works

Write the current ensemble as a sum of weak learners, `F(x) = sum over m of eta * f_m(x)`, where each `f_m` is a tree or stump and `eta` is the learning rate, the shrinkage applied to every new learner's contribution.

Each round does three things. Fit a base learner to the current errors - AdaBoost reweights misclassified rows, gradient boosting fits the negative gradient of the loss. Weight that learner by how much it actually reduces the loss. Add it to the ensemble, damped by `eta`, and move on.

Two consequences follow from the shrinkage. A lower learning rate means each tree changes the model less, so you need more rounds, which is why the learning rate and the round count are always tuned together and why 500 rounds at `eta` 0.03 is a different model from 50 rounds at `eta` 0.3. And because the ensemble only ever moves in the direction that reduces loss, boosting is a form of [gradient descent](/glossary/gradient-descent) in function space, which is exactly how gradient boosting generalized the idea.

## AdaBoost: the original

Freund and Schapire introduced AdaBoost in 1997, and it remains the cleanest statement of what boosting is. Each round, misclassified examples get their weight multiplied up, correctly classified ones get multiplied down, and the next stump trains on the reweighted data. Every learner also receives a vote proportional to its error rate, and the prediction is the weighted majority vote. In practice the base learners are [decision stumps](/glossary/decision-tree) - a single split - which is what makes the reweighting visible rather than buried inside a deep tree.

The theoretical guarantee is worth knowing because it explains the design. The training error of the ensemble is bounded by roughly `half of the exponential of minus gamma times M`, so error falls exponentially in the number of rounds as long as every learner does better than random. It was the first clean result to show that a weak-learner assumption plus enough rounds yields a strong learner, and it reset how ensembles were thought about.

What AdaBoost cannot do is survive messy data. Misclassified outliers receive exponentially increasing weight, so mislabeled rows get amplified rather than smoothed, and the ensemble chases them. The AdaBoost.M1 loss is exponential only, so multiclass variants (SAMME, SAMME.R) exist as patches, and gradient boosting later replaced that restricted loss with any differentiable objective. Practical verdict: AdaBoost is a teaching tool and a reasonable baseline on clean, small, binary problems. It is not what you deploy.

## Gradient boosting: the general form

Friedman's 2001 greedy function approximation reframed boosting as stagewise additive modeling over any [loss function](/glossary/loss-function), and it is the direct ancestor of everything that follows. Instead of reweighting rows, you compute the negative gradient of the loss with respect to the current prediction - the residual under squared error, the `p - y` direction under log loss - and fit a tree to that. The tree is a local correction, not a replacement.

Two refinements matter. Second-order (Newton) boosting uses both the first and second derivatives of the loss to score candidate splits, which is more accurate per round and is what XGBoost does. And regularization moved into the objective itself, so leaf weights are penalized as part of the loss rather than only shrunk after the fact.

## XGBoost: regularization plus engineering

XGBoost (2016, Tianqi Chen and Carlos Guestrin) is the implementation that turned boosted trees into a production default, and its contribution is a dense stack of engineering wins on top of gradient boosting:

- An explicit regularized objective - the loss plus a complexity penalty on tree shape plus L1 and L2 penalties on leaf weights.
- Second-order split evaluation, scoring candidates with the Hessian as well as the gradient.
- Sparsity-aware split finding, so missing values get a learned default direction per node instead of being imputed.
- Quantile-based sketching instead of sorting, which cut the cost of finding split points on wide data.
- Cache-aware and distributed tree construction, so the data is loaded once and reused across rounds.

The defaults are deliberately conservative and are a usable starting point on their own: `eta` 0.3, `max_depth` 6, `min_child_weight` 1, `gamma` 0, `subsample` 1, `colsample_bytree` 1, `reg_lambda` 1, `reg_alpha` 0. XGBoost 3.0 (March 2025) unified the CPU and GPU feature sets, made NaN the default missing value, and requires CUDA 12 or later for GPU work.

XGBoost's practical reputation comes from its growth policy. It grows trees level-wise and ships an L2 penalty by default, which makes it more robust out of the box than LightGBM on small or noisy data. That is why it remains a safe first model even when LightGBM would score higher after tuning.

## LightGBM: the same idea, a different growth policy

LightGBM (2017, Microsoft Research) attacks training time. The idea that catches most people first is leaf-wise growth: instead of finishing a depth level before moving on, it splits whichever leaf gives the largest gain, producing deeper, asymmetric trees that reduce loss faster per round. The cost is [overfitting](/glossary/overfitting) risk on small data, and the control is `num_leaves` rather than `max_depth`, because leaf count is what actually bounds tree size in this scheme.

The bigger win is that LightGBM stops looking at raw values. Every feature is binned into a histogram once, and each subsequent split compares bin boundaries instead of data points, so split finding is roughly O(bins) rather than O(n) and each row costs one byte of bin index rather than a float. On top of that come Gradient-based One-Side Sampling (keep every large-gradient row, subsample the small-gradient rows, and upweight the sample to correct the bias) and Exclusive Feature Bundling (pack mutually exclusive sparse features into one bundle). That is why the paper reports training up to 20 times faster than conventional GBDT with comparable accuracy. LightGBM also handles categoricals natively, using target statistics smoothed by the target prior instead of one-hot encoding.

The combination wins on large row counts, high-dimensional sparse data, and tables with many categorical columns. On small tables the leaf-wise aggressiveness is a liability. The project reached 4.7 in July 2026 and now lives under the lightgbm-org GitHub organization, with ROCm and multi-GPU support added along the way.

## CatBoost and the rest of the family

CatBoost (2018) attacked the problem the others were ignoring: categorical features. One-hot encoding a high-cardinality column throws away the fact that its values are unordered, and naive target encoding leaks because it is fitted on all the data. CatBoost uses ordered target statistics computed only from preceding rows, plus oblivious (symmetric) trees that split on every feature at the same time. It also offers ordered boosting, which reduces prediction-shift bias on small datasets.

Also worth knowing: scikit-learn's `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` are a first-class gradient boosting implementation with the same histogram idea and no third-party dependency. XGBoost and LightGBM both ship DART (dropout-style boosting) and a `goss` sampling mode. And in AutoML systems such as AutoGluon, LightGBM has become the more common default choice rather than XGBoost.

## The four side by side

| Algorithm | Released | Trains | Growth | Watch out for |
| --- | --- | --- | --- | --- |
| AdaBoost | 1997 | stumps on reweighted rows | level-wise | noise and outliers, exponential loss only |
| XGBoost | 2016 | regularized GBDT | level-wise | more tuning needed to beat LightGBM |
| LightGBM | 2017 | GBDT on binned features | leaf-wise | overfits small tables unless `num_leaves` is capped |
| CatBoost | 2018 | ordered GBDT | symmetric, oblivious | slower per iteration, best-in-class categoricals |

## Boosting versus bagging

| | Bagging (random forest) | Boosting |
| --- | --- | --- |
| Base learners | deep trees, low bias, high variance | shallow trees, high bias, low variance |
| Training | independent, parallel | sequential, each fits the last one's errors |
| Diversity comes from | random row and feature subsets | reweighting or residual fitting |
| Main levers | tree count, minimum samples per leaf | learning rate, depth, rounds, regularization |
| Adding capacity | almost always helps | can hurt, past the point of diminishing returns |
| Tuning tolerance | high | lower, defaults matter more |
| Typical role | strong baseline, cheap | best score, more tuning |

## What actually moves the score

In the order that usually pays off:

- **Rounds and learning rate together, with early stopping** on a held-out split. `eta` between 0.05 and 0.1 with early stopping is a common good default, while a hard 1,000-round cap at `eta` 0.3 is usually worse.
- **Tree capacity.** XGBoost `max_depth` between 4 and 8, LightGBM `num_leaves` between 15 and 63 with `min_data_in_leaf` raised when the table is small.
- **Row and column subsampling.** `subsample` 0.8 and `colsample_bytree` 0.8 buy regularization and decorrelation, and they roughly halve GPU training time.
- **Regularization.** `reg_lambda` for overall shrinkage, `reg_alpha` only if you genuinely want sparse leaf weights, and `gamma` or `min_gain_to_split` to stop worthless splits.
- **Class imbalance.** `scale_pos_weight` or sample weights for a skewed target, paired with the metric you actually care about - PR-AUC rather than accuracy when positives are under one percent of rows.
- **Categorical handling.** The native LightGBM or CatBoost path beats one-hot encoding, and beats target encoding fitted on the full dataset.

## Failure modes worth naming

- **Label noise.** Boosting amplifies it, because mislabeled rows are exactly the ones that keep getting upweighted. Clean the labels or switch to a robust loss before blaming the model.
- **Leaky splits.** Random cross-validation on time-dependent data inflates the estimate badly. Use a chronological split, and confirm every aggregate feature was computed without future information.
- **Leakage through target encoding.** A category statistic fitted on the full dataset is a leak even when the trees themselves are honest.
- **Capacity without early stopping.** Deep trees plus too many rounds produces a beautiful training log and a worse production model.
- **Distribution shift.** The ensemble learned weights calibrated to the training distribution, and they quietly stop applying once the world moves.

## Reading a boosted model

Importance by gain is easy to compute and easy to misread. High-gain features can be redundant, and correlated features split importance between themselves almost arbitrarily. Split count measures how often a feature was used, not how much it mattered.

SHAP values are the honest version. They decompose a single prediction into per-feature contributions, and TreeSHAP is fast enough to run across a whole validation set. Monotonic constraints are available in all three major libraries and are worth setting whenever domain knowledge says a feature cannot push a prediction down.

## Where boosting sits in 2026

Tabular foundation models took the top of the standard small-to-medium i.i.d. benchmarks over the last couple of years, and a single forward pass of a TabPFN-class model now beats tuned and ensembled tree baselines there. It is worth being precise about where that does and does not apply, because the same benchmark work shows a consistent pattern: boosted trees still win or tie on categorical-heavy tables, on large binary problems, and on grouped or temporal splits, and even the strongest foundation models fall behind tuned conventional models once a table has many rows, many features, or text columns.

Boosting also keeps advantages that never show up in an accuracy table. First-class monotonic constraints, cheap CPU-only inference, fast retraining when data moves, and a tuning surface you can reason about instead of hoping a pre-trained prior covers your case. So boosted trees are no longer the automatic winner on every table, and they are still the pragmatic default - especially where the data is messy, categories dominate, deployment is CPU-bound, or someone has to explain a decision.

## How to choose in one minute

- **Any tabular baseline at all:** XGBoost or LightGBM at defaults with early stopping, and keep a plain linear model and a random forest as reference points.
- **Millions of rows, wide or sparse, or mostly categorical:** LightGBM, or CatBoost when the categoricals are high-cardinality and the training budget is comfortable.
- **Small, noisy, or high-stakes:** XGBoost with lower depth and stronger regularization, and treat the validation split as sacred.
- **Learning the mechanism, or a tiny clean binary problem:** AdaBoost, for how clearly the mechanism shows up in the weights.
$body$::text AS body
)
UPDATE content_items AS item
SET
  summary = 'A family of ensemble methods that adds weak learners one at a time, each one fitting the errors of the ensemble so far - the family behind AdaBoost, XGBoost, LightGBM, and CatBoost.',
  body = boosting_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', boosting_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Freund and Schapire: A Decision-Theoretic Generalization of Boosting', 'url', 'https://www.jmlr.org/papers/volume18/15-145.html'),
    jsonb_build_object('title', 'Friedman: Greedy Function Approximation - A Gradient Boosting Machine', 'url', 'https://jmlr.org/papers/volume1/friedman00a.html'),
    jsonb_build_object('title', 'Chen and Guestrin: XGBoost - A Scalable Tree Boosting System (KDD 2016)', 'url', 'https://dl.acm.org/doi/10.1145/2939672.2939785'),
    jsonb_build_object('title', 'XGBoost parameter documentation', 'url', 'https://xgboost.readthedocs.io/en/stable/parameter.html'),
    jsonb_build_object('title', 'Ke et al: LightGBM - A Highly Efficient Gradient Boosting Decision Tree (NeurIPS 2017)', 'url', 'https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree'),
    jsonb_build_object('title', 'LightGBM parameter documentation', 'url', 'https://lightgbm.readthedocs.io/en/latest/Parameters.html'),
    jsonb_build_object('title', 'LightGBM 4.7.0 release notes', 'url', 'https://github.com/lightgbm-org/LightGBM/releases/tag/v4.7.0'),
    jsonb_build_object('title', 'Prokhorenkova et al: CatBoost - unbiased boosting with categorical features (NeurIPS 2018)', 'url', 'https://proceedings.neurips.cc/paper/14464-catboost-generalized-predictive-categorical-features'),
    jsonb_build_object('title', 'scikit-learn: ensemble learning methods and boosting guide', 'url', 'https://scikit-learn.org/stable/modules/ensemble.html'),
    jsonb_build_object('title', 'A new performance standard - tabular foundation models against tuned gradient-boosted trees', 'url', 'https://arxiv.org/abs/2605.13986')
  ),
  metadata = jsonb_set(
    jsonb_set(
      jsonb_set(
        COALESCE(item.metadata, '{}'::jsonb),
        '{seoKeywords}',
        jsonb_build_array(
          'what is boosting',
          'what is boosting in ai',
          'boosting explained',
          'boosting machine learning',
          'boosting ensemble methods',
          'boosting vs bagging',
          'gradient boosting',
          'what is AdaBoost',
          'AdaBoost explained',
          'AdaBoost weak learners',
          'what is XGBoost',
          'XGBoost explained',
          'what is LightGBM',
          'LightGBM explained',
          'XGBoost vs LightGBM',
          'LightGBM vs XGBoost',
          'XGBoost hyperparameter tuning',
          'CatBoost vs LightGBM',
          'boosting for tabular data',
          'gradient boosting parameters',
          'early stopping XGBoost',
          'SHAP values tree models'
        ),
        true
      ),
      '{seoDescription}',
      '"Boosting combines weak learners sequentially so each one fixes the previous model''s errors. Covers AdaBoost, gradient boosting, XGBoost, LightGBM, and CatBoost, how they differ, and how to tune them."'::jsonb,
      true
    ),
    '{relatedTerms}',
    jsonb_build_array(
      'decision-tree',
      'random-forest',
      'machine-learning',
      'supervised-learning',
      'overfitting',
      'underfitting',
      'bias-variance-tradeoff',
      'hyperparameter',
      'loss-function',
      'gradient-descent',
      'feature-engineering',
      'knn',
      'svm',
      'foundation-model',
      'neural-network'
    ),
    true
  ),
  updated_at = NOW()
FROM boosting_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'boosting'
  AND item.parent_slug = '';

-- The three variant pages now live inside the boosting page. Deleting the rows
-- also clears the three inbound relatedTerms links that pointed at them.
DELETE FROM content_items
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug IN ('adaboost', 'xgboost', 'lightgbm');
