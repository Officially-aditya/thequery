-- Expand the learning-rate glossary page into a hub covering the full family.
--
-- The page was three thin paragraphs (1147 chars, no references, no SEO
-- metadata) that named warmup, cosine annealing, step decay, Adam, and the
-- LR range test without ever explaining any of them. A reader searching
-- "what is warmup" or "cosine annealing vs step decay" landed on a page
-- that mentioned the term and moved on, and a reader searching
-- "learning rate" got no tuning guidance at all.
--
-- The expanded page keeps the existing opening (step-size update rule), then
-- covers each member in its own section: too-high vs too-low, where the rate
-- lives (hyperparameter, loss landscape, backprop, epochs, batch size),
-- schedules (warmup, cosine annealing, step decay), adaptive optimizers
-- (SGD vs Adam), tuning (range test plus working defaults), a side-by-side
-- table, a one-minute chooser, and failure modes. Warmup, cosine annealing,
-- step decay, and the range test have no separate pages and are covered here,
-- the same hub pattern used when ReLU and sigmoid were merged into
-- activation-function (083). All inline links and relatedTerms point at
-- slugs that exist in the glossary.
--
-- Sources consulted on September 28 2026:
-- - Goodfellow, Bengio & Courville, "Deep Learning", Chapter 8 -
--   Optimization for Training Deep Models
--   (https://www.deeplearningbook.org/contents/optimization.html)
-- - Kingma & Ba, "Adam: A Method for Stochastic Optimization", 2014
--   (https://arxiv.org/abs/1412.6980)
-- - Smith, "Cyclical Learning Rates for Training Neural Networks", 2017
--   (https://arxiv.org/abs/1506.01186)
-- - Smith, "A Disciplined Approach to Neural Network Hyper-Parameters"
--   (learning rate range test, 2018)
--   (https://arxiv.org/abs/1803.09820)
-- - Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm
--   Restarts" (cosine annealing, 2016)
--   (https://arxiv.org/abs/1608.03983)
--
-- Body and blocks stay in sync, sources are attached, and metadata gains the
-- keyword set, SEO description, and related terms. No pages are merged or
-- deleted; this migration only expands learning-rate in place.

WITH lr_entry AS (
  SELECT $body$A learning rate is the step size applied to every [gradient descent](/glossary/gradient-descent) update. At each training step the weights move against the gradient of the [loss function](/glossary/loss-function): $w_{t+1} = w_t - \eta \cdot \nabla L(w_t)$, where $\eta$ is the learning rate and $\nabla L$ is the gradient produced by [backpropagation](/glossary/backpropagation). This single number has an outsized impact on whether training succeeds or fails, which is why it is the canonical example of a [hyperparameter](/glossary/hyperparameter).

Fixed learning rates, schedules that change the rate during training, and adaptive optimizers that set a different effective rate per parameter are all answers to the same question: how far should the weights move on this step? This page covers the full family in one place — what a learning rate scales, what goes wrong at each extreme, the three schedules that carry nearly all of modern practice (warmup, cosine annealing, step decay), how [Adam](/glossary/adam-optimizer) differs from [stochastic gradient descent](/glossary/stochastic-gradient-descent), and how to find a good value with a learning rate range test. Warmup, cosine annealing, step decay, and the range test have no separate pages; they are covered here because they only make sense as learning-rate strategies.

## Too high vs too low

If the learning rate is too high, updates overshoot the minimum. Loss oscillates wildly, diverges toward infinity, or collapses to NaN within a few steps. Large steps also amplify noisy gradients, which is one route into [exploding gradients](/glossary/exploding-gradients) — [gradient clipping](/glossary/gradient-clipping) caps the damage but does not fix a rate that is fundamentally too large.

If the learning rate is too low, training converges extremely slowly, burns through [epochs](/glossary/epoch) with barely-moving loss, and can stall in a poor local minimum or a flat plateau that a larger step would have crossed. Chronically small steps also mimic [underfitting](/glossary/underfitting): the model never reaches the fit its capacity allows, not because the architecture is wrong but because the optimizer was never allowed to get there.

The ideal rate depends on the loss landscape, the model architecture, the batch size, and the training stage. That last dependence is why fixed rates are rare in serious training — what is correct at step 100 is usually wrong at step 100,000.

## Where the learning rate lives

The learning rate never acts alone. Four relationships define its context:

- **It is a hyperparameter, not a parameter.** Weights and biases are learned from data; the learning rate, batch size, depth, and [regularization](/glossary/regularization) strength are chosen before training, alongside [dropout](/glossary/dropout) rates and [epoch](/glossary/epoch) budgets. Tuning it on the test set leaks test information — use a validation split.
- **The loss function draws the landscape; the rate sets the stride.** [Backpropagation](/glossary/backpropagation) computes $\nabla L$, the direction downhill. The rate decides how far to walk before re-measuring. Sharp, narrow valleys need small strides; broad bowls tolerate large ones.
- **Learning rate × epochs = training budget.** Halving the rate roughly doubles the steps needed to travel the same distance, so rate and epoch count must be tuned together. A schedule is how practitioners spend that budget wisely: large steps early, careful steps late.
- **Batch size and learning rate scale together.** Larger batches give cleaner gradient estimates, which support larger stable steps — the commonly cited linear scaling rule says doubling the batch size roughly supports doubling the rate, up to the point where curvature, not noise, becomes the limit.

Push the rate too far in either direction for too long and the familiar pathologies appear: divergence and oscillation on one side, stagnation and [underfitting](/glossary/underfitting) on the other, and [overfitting](/glossary/overfitting) when too many epochs at a small rate memorize noise instead of learning structure.

## Schedules: warmup, cosine annealing, step decay

A schedule changes $\eta$ as training progresses because no single value is right for the whole run. Early steps start from random weights with unreliable gradient statistics; late steps sit near a minimum where large strides overshoot. Three schedules cover nearly all modern practice:

**Warmup: start small, then ramp up.** Training begins at a near-zero rate and increases linearly (sometimes over the first few percent of steps) to the target value. This matters most for transformers and [Adam](/glossary/adam-optimizer)-family optimizers, where the second-moment estimates are uninitialized and the first batches would otherwise take wild steps. Skipping warmup is a classic cause of early loss spikes that never recover. Warmup is a beginning, not a full schedule — it hands off to one of the decays below.

**Cosine annealing: glide down on a cosine curve.** The rate follows $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi \cdot t / T))$, decaying smoothly from max to min over $T$ steps. The curve stays high longer than exponential decay, then drops steeply, then flattens — a shape that empirically settles into sharper minima without the shock of a sudden drop. The SGDR variant restarts the cosine periodically (warm restarts), briefly raising the rate to jump out of sharp minima before annealing again.

**Step decay: drop by a factor at milestones.** The rate holds constant, then is cut — typically by 10x at fixed epoch boundaries (for example, epochs 30, 60, 90 in a 100-epoch run). Simple, predictable, and still the default in many vision pipelines. Its weakness is the discontinuity: each drop is a small shock, and badly timed milestones waste epochs either crawling before the drop or oscillating after it.

## Adaptive optimizers: SGD vs Adam

[Gradient descent](/glossary/gradient-descent) in its plain form uses one global $\eta$ for every parameter. [Stochastic gradient descent](/glossary/stochastic-gradient-descent) with momentum keeps that single rate but smooths the direction across steps, which rides through ravines faster but still demands a hand-designed schedule and a carefully chosen base rate.

[Adam](/glossary/adam-optimizer) instead maintains a per-parameter effective rate. For gradient $g_t$ it tracks a first moment $m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$ (momentum) and a second moment $v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$ (per-dimension scale), bias-corrects both, and updates $\theta_t = \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$. Parameters with consistently large gradients get smaller effective steps; quiet parameters get larger ones. The full derivation, defaults (`1e-3`, `0.9`, `0.999`), and the AdamW weight-decay fix live on the [Adam](/glossary/adam-optimizer) page.

That adaptivity removes one tuning burden and adds a misconception: Adam still has a learning rate, and it still needs scheduling. Transformer training with Adam almost always pairs a warmup with cosine or linear decay, and fine-tuning typically drops the base rate 10x below pretraining. A well-tuned SGD schedule can still beat Adam on final generalization in vision; Adam wins on speed-to-a-good-loss and on noisy, sparse, or badly-scaled problems where hand-tuning SGD is impractical.

## How to tune: the range test and working defaults

The learning rate range test (Smith) removes most guesswork. Increase $\eta$ exponentially across a single short run — from a tiny value to an obviously-too-large one — and plot loss against rate. Loss barely moves at first, then falls steeply, then rises and diverges. Pick a rate near the steepest descent, roughly one order of magnitude below the divergence point. One cheap sweep replaces dozens of full training runs, and it should be re-run whenever the architecture, batch size, or data change substantially.

Working defaults, as starting points rather than laws:

- **SGD with momentum:** `0.01` to `0.1` with momentum `0.9`, plus step decay or cosine annealing. Lower end for small batches, higher end for large ones.
- **Adam / AdamW:** `1e-3` for from-scratch training, `3e-4` down to `5e-5` for transformers, and roughly 10x below your pretraining rate when fine-tuning. Always pair with warmup on transformers.
- **Fine-tuning any model:** start 10x below the pretraining rate. The weights are already good; large steps destroy them.
- **New architecture or dataset:** run the range test before anything else. It is the single highest-value hyperparameter experiment available.

Tune the learning rate before touching optimizer betas, initialization schemes, or architecture width. In most failed training runs the rate — not the model — was the bug.

## The family side by side

| Strategy | What it does | When to use | Watch out for |
| --- | --- | --- | --- |
| Constant rate | One $\eta$ for the whole run | Tiny models, debugging, short runs | Wrong at one end of training by construction |
| Warmup | Ramps $\eta$ from near zero to target | Transformer starts, Adam starts, large batches | Not a full schedule — must hand off to a decay |
| Cosine annealing | Smooth cosine glide from max to min | Long single runs, transformers, SGDR restarts | Decays too early if $T$ is mis-set; restarts add tuning surface |
| Step decay | 10x cuts at fixed milestones | Vision pipelines, reproducible baselines | Discontinuity shocks; badly timed milestones waste epochs |
| Adam adaptive rates | Per-parameter $\eta / \sqrt{\hat{v}_t}$ scaling | Noisy, sparse, or badly-scaled problems; fast first loss | Still needs warmup + decay; can generalize worse than tuned SGD |
| LR range test | One sweep to find the usable window | Before any serious tuning, after data/arch changes | Findings do not transfer across batch sizes or architectures |

## How to choose in one minute

- **Starting from scratch, small model:** SGD or Adam at the defaults above, constant rate until loss moves, then add a schedule.
- **Training a transformer:** AdamW with warmup (first few percent of steps) into cosine decay. Never skip warmup.
- **Fine-tuning:** Same optimizer, 10x lower base rate, shorter schedule. Watch validation loss — it turns before training loss does.
- **Loss explodes in the first epochs:** Cut the rate 10x and add warmup before touching anything else.
- **Loss plateaus early but capacity remains:** The rate decayed too far or too fast — extend $T$, delay milestones, or raise $\eta_{min}$.
- **Unsure of any value:** Run the range test. It costs a fraction of one training run.

## Failure modes worth naming

- **Divergence.** Loss goes to infinity or NaN within steps or epochs. The rate is too high, warmup is missing, or a mixed-precision overflow amplified a borderline rate. Cut 10x, add warmup, re-run.
- **Frozen training.** Loss barely moves across epochs while gradients are non-zero. The rate is orders of magnitude too small, or cosine/step decay dropped it to $\eta_{min}$ far too early. Check the schedule position, not just the base rate.
- **Warmup-skipped blowup.** Transformer loss spikes violently in the first percent of training under Adam. Uninitialized second moments plus a full base rate from step zero. Add linear warmup; nothing else needs to change.
- **Oscillation near the minimum.** Loss bounces in a band instead of settling. The late-training rate is too large for the valley width — the schedule never decayed, or $\eta_{min}$ is too high.
- **Memorization mistaken for learning.** Tiny rate plus far too many [epochs](/glossary/epoch) drives training loss down while validation loss rises — textbook [overfitting](/glossary/overfitting). Fix with early stopping and [regularization](/glossary/regularization), not a still-smaller rate.
- **Adam under-generalizes.** Adam reaches good training loss fast but validates worse than published SGD baselines. Known trade-off on some vision tasks; try tuned SGD with momentum and step decay, or AdamW with proper decay, before concluding the architecture is at fault.
- **Vanishing and exploding gradients blamed on the rate alone.** Saturated activations cause [vanishing gradients](/glossary/vanishing-gradients) and deep unclipped stacks cause [exploding gradients](/glossary/exploding-gradients) at any reasonable $\eta$. If early layers never move while late layers learn, inspect activations and [gradient clipping](/glossary/gradient-clipping) before re-tuning the rate.
$body$::text AS body
)
UPDATE content_items AS item
SET
  summary = 'The step-size hyperparameter behind gradient descent — what it scales, why warmup, cosine annealing, step decay, and Adam exist, and how to tune it.',
  body = lr_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', lr_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Goodfellow, Bengio and Courville: Deep Learning, Chapter 8 - Optimization for Training Deep Models', 'url', 'https://www.deeplearningbook.org/contents/optimization.html'),
    jsonb_build_object('title', 'Kingma & Ba: Adam - A Method for Stochastic Optimization (2014)', 'url', 'https://arxiv.org/abs/1412.6980'),
    jsonb_build_object('title', 'Smith: Cyclical Learning Rates for Training Neural Networks (2017)', 'url', 'https://arxiv.org/abs/1506.01186'),
    jsonb_build_object('title', 'Smith: A Disciplined Approach to Neural Network Hyper-Parameters - LR Range Test (2018)', 'url', 'https://arxiv.org/abs/1803.09820'),
    jsonb_build_object('title', 'Loshchilov & Hutter: SGDR - Stochastic Gradient Descent with Warm Restarts (2016)', 'url', 'https://arxiv.org/abs/1608.03983')
  ),
  metadata = jsonb_set(
    jsonb_set(
      jsonb_set(
        COALESCE(item.metadata, '{}'::jsonb),
        '{seoKeywords}',
        jsonb_build_array(
          'what is learning rate',
          'learning rate explained',
          'learning rate vs batch size',
          'learning rate warmup',
          'what is warmup in deep learning',
          'cosine annealing explained',
          'step decay learning rate',
          'learning rate schedule compared',
          'adam vs sgd learning rate',
          'adam optimizer learning rate',
          'learning rate range test',
          'how to tune learning rate',
          'learning rate too high',
          'learning rate too low',
          'fine-tuning learning rate',
          'transformer learning rate warmup',
          'which learning rate to use',
          'learning rate divergence'
        ),
        true
      ),
      '{seoDescription}',
      '"Learning rate explained: step size in gradient descent, warmup, cosine annealing, step decay schedules, Adam, LR range test, and tuning defaults."'::jsonb,
      true
    ),
    '{relatedTerms}',
    jsonb_build_array(
      'gradient-descent',
      'stochastic-gradient-descent',
      'adam-optimizer',
      'hyperparameter',
      'loss-function',
      'backpropagation',
      'epoch',
      'overfitting',
      'underfitting',
      'gradient-clipping',
      'vanishing-gradients',
      'exploding-gradients'
    ),
    true
  ),
  published_at = DATE '2026-09-28',
  updated_at = NOW()
FROM lr_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'learning-rate'
  AND item.parent_slug = '';
