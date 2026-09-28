-- Merge the ReLU and Sigmoid glossary pages into the activation-function
-- page, then delete the two merged rows.
--
-- The two pages were thin slices of the same concept. A reader searching
-- "what is ReLU" landed on a page that explained ReLU and never said how
-- it relates to the family it belongs to, and a reader searching
-- "activation function" got three short paragraphs that named tanh, GELU,
-- Leaky ReLU, PReLU, and SiLU/Swish without ever explaining any of them.
-- The merged page keeps the existing opening, then covers each type in its
-- own section: ReLU, sigmoid, tanh, Leaky ReLU, PReLU, GELU, SiLU/Swish,
-- and softmax (as output normalization, not a hidden-layer activation),
-- plus a side-by-side table, a one-minute chooser, and failure modes.
--
-- Sources consulted on September 28 2026:
-- - Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann
--   Machines", ICML 2010
--   (https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)
-- - He et al., "Delving Deep into Rectifiers", 2015
--   (https://arxiv.org/abs/1502.01852)
-- - Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", 2016
--   (https://arxiv.org/abs/1606.08415)
-- - Ramachandran et al., "Searching for Activation Functions", 2017
--   (https://arxiv.org/abs/1710.05941)
-- - Goodfellow, Bengio & Courville, "Deep Learning", Chapter 6
--   (https://www.deeplearningbook.org/contents/mlp.html)
-- - Vaswani et al., "Attention Is All You Need", 2017
--   (https://arxiv.org/abs/1706.03762)
--
-- Body and blocks stay in sync, sources are attached, and metadata gains the
-- merged keyword set plus related terms that all exist as glossary slugs. The
-- relu and sigmoid slugs are dropped, and the three inbound relatedTerms
-- entries that pointed at them (vanishing-gradients, weight-initialization,
-- logits) are repointed at activation-function.

WITH activation_entry AS (
  SELECT $body$
An activation function is a non-linear transformation applied to the weighted sum of inputs at each neuron in a [neural network](/glossary/neural-network). Without activation functions, a neural network would be equivalent to a single linear transformation regardless of its depth, severely limiting its ability to model complex relationships.

Seven elementwise functions carry nearly all of the weight in hidden layers today - ReLU, sigmoid, tanh, Leaky ReLU, PReLU, GELU, and SiLU/Swish - plus softmax, which is not a hidden-layer activation at all but the output normalization that turns scores into probabilities. This page covers the full family in one place: what each one computes, why it was introduced, where it still wins, and where it breaks.

## ReLU: the default

ReLU (Rectified Linear Unit) computes `f(x) = max(0, x)`: it outputs the input directly if positive, and zero otherwise. Despite its simplicity, ReLU was a breakthrough that helped make deep learning practical. Its gradient is either 1 (for positive inputs) or 0 (for negative inputs), which eliminated the [vanishing gradient](/glossary/vanishing-gradients) problem that plagued sigmoid and tanh activations in deep stacks.

ReLU offers three compounding advantages: its gradient does not saturate for positive values (unlike sigmoid/tanh), enabling effective training of deep networks, and it produces sparse activations (roughly half of neurons output zero on random input), which is computationally efficient and creates more separable representations, and `max(0, x)` is trivially fast to compute compared to the exponentials in sigmoid and tanh.

The main drawback is the dying ReLU problem: neurons that receive only negative inputs always output zero and stop learning entirely, because their gradient is permanently zero. This shows up most on sparse inputs, bad initialization, or learning rates set too high. The standard fix is [weight initialization](/glossary/weight-initialization) designed for ReLU - He initialization uses `Var(w) = 2/n_in` instead of Xavier's `2/(n_in + n_out)` precisely to account for ReLU zeroing half of activations - plus the Leaky and parametric variants below.

## Sigmoid: the original

The sigmoid function `sigma(x) = 1/(1 + e^(-x))` squashes any input into the range (0, 1), making it useful for producing probability-like outputs. It was one of the earliest activation functions used in neural networks and is still used in the output layer for binary classification and in gating mechanisms that need a smooth 0-to-1 gate.

However, sigmoid has decisive drawbacks for hidden layers in deep networks. It saturates for large positive or negative inputs (`sigma'(x)` approaches 0 at both ends), causing the vanishing gradient problem: gradients shrink exponentially as they propagate backward, making early layers nearly impossible to train. Additionally, sigmoid outputs are not zero-centered (always positive), which causes inefficient zigzag gradient updates because all gradients into a layer share the same sign.

For these reasons, ReLU and its variants have largely replaced sigmoid in hidden layers of modern deep networks. Sigmoid remains important in three specific contexts: binary classification output layers, gating inside recurrent cells, and any situation where you need a smooth function mapping to (0, 1). Understanding why sigmoid fails in deep hidden layers is a key insight in the history of deep learning.

## tanh: the zero-centered sibling

The hyperbolic tangent `tanh(x)` maps values to (-1, 1). It is sigmoid rescaled and shifted - `tanh(x) = 2*sigmoid(2x) - 1` - and that shift matters: unlike sigmoid, tanh is zero-centered, so its outputs average near zero and gradient updates do not zigzag the way sigmoid's do.

That fixes one of sigmoid's two problems but not the other. tanh still saturates at both extremes, so its derivative still collapses toward zero for large-magnitude inputs and deep stacks still suffer vanishing gradients. It trains better than sigmoid in practice, which is why it survived longer in recurrent hidden states, but it lost the same fight to ReLU in feedforward depth. Use tanh today when a bounded, zero-centered output in (-1, 1) is semantically what you want - legacy recurrent states are the main example - not as a general hidden-layer default.

## Leaky ReLU and PReLU: fixing dying ReLU

Leaky ReLU keeps ReLU's shape for positive inputs and gives negative inputs a small slope instead of zero: `f(x) = max(a*x, x)` with `a` typically 0.01. The gradient for negative inputs is `a` rather than 0, so neurons never fully die - there is always a learning signal, however small.

PReLU (Parametric ReLU) is the same function with `a` learned per channel during training instead of fixed. He et al. introduced it alongside He initialization in 2015 and reported gains on ImageNet, the first time a learned rectifier beat fixed ReLU at that scale. The cost is capacity: a per-channel slope is another parameter to fit, so on small datasets PReLU can overfit where Leaky ReLU would not. Practical verdict: Leaky ReLU is the cheap insurance policy against dying neurons on sparse or noisy inputs. PReLU earns its keep on large vision datasets where the extra parameters are affordable.

## GELU: the transformer default

The Gaussian Error Linear Unit (GELU) weights the input by the probability that it is kept under a standard Gaussian: `GELU(x) = x * Phi(x)`, where `Phi` is the Gaussian CDF. Intuitively it is a smooth, probabilistic version of ReLU - small negative values are suppressed gradually rather than cut to exactly zero, and the exact form is usually replaced by the tanh approximation `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))` for speed.

GELU is the default inside transformer feedforward blocks (BERT, GPT family, Vision Transformers) because that smoothness helps optimization in very deep stacks: the gradient is defined everywhere, there is no hard kink at zero, and the probabilistic gating preserves more signal than hard zeroing. It costs an error function or tanh evaluation per element, noticeably more than `max(0, x)`, which is why it lives in transformers where accuracy dominates per-op cost rather than in every small CNN. If you are writing a transformer block from scratch, GELU is the starting choice and ReLU is the ablation.

## SiLU and Swish: the smooth gate

SiLU (Sigmoid Linear Unit) computes `SiLU(x) = x * sigmoid(x)`. Swish is the same family with a temperature: `Swish(x) = x * sigmoid(beta*x)`, so SiLU is Swish with `beta = 1`. Both are smooth and non-monotonic - the curve dips slightly below zero for small negative inputs before rising - which lets them preserve a little more information than ReLU while keeping ReLU's unbounded positive side.

Ramachandran et al. found Swish beating ReLU on several vision benchmarks in 2017, and SiLU has since become common in vision transformers and some large language model variants. Treat GELU and SiLU/Swish as siblings rather than rivals: GELU gates by a Gaussian CDF, SiLU gates by a sigmoid, both smooth out ReLU's kink, and both cost roughly one sigmoid evaluation. When a paper reports Swish, check the `beta` - a learned or tuned `beta` is doing real work, while `beta = 1` is just SiLU under another name.

## Softmax: output normalization, not a hidden activation

Softmax takes a vector of arbitrary real-valued scores ([logits](/glossary/logits)) and transforms them into probabilities: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`. It is covered in full on the [softmax](/glossary/softmax) page, it is included here because beginners routinely list it alongside ReLU and sigmoid - and that category error causes real bugs.

The differences are structural. ReLU, sigmoid, tanh, GELU, and SiLU are elementwise: each neuron's output depends only on its own input. Softmax is collective: each output depends on every input through the shared denominator, so changing one logit reshapes the whole distribution. It amplifies differences through exponentiation, and a [temperature](/glossary/temperature) parameter (`logits / tau` before softmax) controls sharpness - low temperature toward one-hot, high temperature toward uniform. In [transformers](/glossary/transformer), softmax converts attention scores into [attention](/glossary/attention-mechanism) weights, with the `1/sqrt(d_k)` scale factor keeping logits small enough that softmax does not saturate into near-zero gradients. Never use softmax as a hidden-layer non-linearity, use it where a distribution over alternatives is the semantics you want.

## The family side by side

| Function | Formula | Range | Zero-centered? | Watch out for |
| --- | --- | --- | --- | --- |
| ReLU | `max(0, x)` | [0, inf) | No | dying neurons on negative-only input |
| Sigmoid | `1/(1 + e^(-x))` | (0, 1) | No | saturates both ends, vanishing gradients |
| tanh | `tanh(x)` | (-1, 1) | Yes | still saturates, still vanishes in depth |
| Leaky ReLU | `max(a*x, x)`, `a` ~ 0.01 | (-inf, inf) | No | tiny negative slope is still nearly linear there |
| PReLU | `max(a*x, x)`, `a` learned | (-inf, inf) | No | extra parameters can overfit small data |
| GELU | `x * Phi(x)` | ~(−0.17, inf) | No | erf/tanh approx costs more than ReLU |
| SiLU / Swish | `x * sigmoid(beta*x)` | ~(−0.28, inf) | No | same cost class as GELU, `beta` matters |
| Softmax | `exp(x_i)/sum(exp(x_j))` | (0, 1), sums to 1 | N/A (vector) | not elementwise, saturates on large logits |

## How to choose in one minute

- **Hidden MLP or CNN default:** ReLU with He initialization. Add nothing until something breaks.
- **Neurons dying (many permanent zeros, sparse inputs, high learning rate):** Leaky ReLU with `a` 0.01, or PReLU on large vision data.
- **Transformer feedforward block:** GELU. SiLU is the alternate when the architecture was tuned around it.
- **Bounded zero-centered state in (-1, 1), usually legacy recurrent code:** tanh.
- **Binary output or a 0-to-1 gate:** sigmoid.
- **Distribution over classes or attention targets:** softmax (see the [softmax](/glossary/softmax) page), with [cross-entropy](/glossary/cross-entropy) as the matching loss.

## Failure modes worth naming

- **Dying ReLU.** A neuron pushed fully negative never recovers because its gradient is exactly zero. Lower the learning rate, fix initialization, or switch to Leaky ReLU rather than widening the layer and hoping.
- **Saturation.** Sigmoid, tanh, and softmax all flatten at extremes, and flat means near-zero gradient. If early layers stop moving while late layers still learn, check activation magnitudes before blaming the optimizer.
- **Non-zero-centered outputs.** Sigmoid and ReLU outputs skew positive, which biases the next layer's gradients into zigzags. [Normalization](/glossary/batch-normalization) and zero-centered choices (tanh, proper init) exist largely to counteract this.
- **Softmax overconfidence.** Large logits collapse softmax toward one-hot and kill gradients, the `1/sqrt(d_k)` scaling in attention and temperature tuning in classification are load-bearing, not cosmetic.
- **Paying transformer prices everywhere.** GELU and SiLU cost exponentials per element. That trade is correct inside a 100M-parameter transformer and wasteful in a small model where ReLU trains just as well.
$body$::text AS body
)
UPDATE content_items AS item
SET
  summary = 'The non-linear function inside every neuron — the family behind ReLU, sigmoid, tanh, GELU, SiLU/Swish, and output normalization via softmax.',
  body = activation_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', activation_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Nair and Hinton: Rectified Linear Units Improve Restricted Boltzmann Machines (ICML 2010)', 'url', 'https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf'),
    jsonb_build_object('title', 'He et al: Delving Deep into Rectifiers - Surpassing Human-Level Performance (He init + PReLU, 2015)', 'url', 'https://arxiv.org/abs/1502.01852'),
    jsonb_build_object('title', 'Hendrycks and Gimpel: Gaussian Error Linear Units (GELUs, 2016)', 'url', 'https://arxiv.org/abs/1606.08415'),
    jsonb_build_object('title', 'Ramachandran et al: Searching for Activation Functions (Swish, 2017)', 'url', 'https://arxiv.org/abs/1710.05941'),
    jsonb_build_object('title', 'Goodfellow, Bengio and Courville: Deep Learning, Chapter 6 - Feedforward Networks', 'url', 'https://www.deeplearningbook.org/contents/mlp.html'),
    jsonb_build_object('title', 'Vaswani et al: Attention Is All You Need (scaled dot-product + softmax, 2017)', 'url', 'https://arxiv.org/abs/1706.03762')
  ),
  metadata = jsonb_set(
    jsonb_set(
      jsonb_set(
        COALESCE(item.metadata, '{}'::jsonb),
        '{seoKeywords}',
        jsonb_build_array(
          'what is activation function',
          'activation function explained',
          'activation functions compared',
          'what is relu',
          'relu explained',
          'dying relu',
          'what is sigmoid',
          'sigmoid vs relu',
          'what is tanh',
          'tanh vs sigmoid',
          'leaky relu explained',
          'what is prelu',
          'what is gelu',
          'gelu vs relu',
          'what is silu',
          'swish activation explained',
          'what is softmax',
          'softmax vs sigmoid',
          'which activation function to use',
          'vanishing gradient activation'
        ),
        true
      ),
      '{seoDescription}',
      '"ReLU, sigmoid, tanh, GELU, SiLU and softmax explained in one place — formulas, trade-offs, dying ReLU, vanishing gradients, and which to use."'::jsonb,
      true
    ),
    '{relatedTerms}',
    jsonb_build_array(
      'neural-network',
      'backpropagation',
      'deep-learning',
      'vanishing-gradients',
      'weight-initialization',
      'softmax',
      'logits',
      'loss-function',
      'gradient-descent',
      'transformer',
      'attention-mechanism',
      'perceptron'
    ),
    true
  ),
  published_at = DATE '2026-09-28',
  updated_at = NOW()
FROM activation_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'activation-function'
  AND item.parent_slug = '';

-- Repoint inbound relatedTerms that referenced the deleted slugs.
UPDATE content_items AS item
SET metadata = jsonb_set(
  COALESCE(item.metadata, '{}'::jsonb),
  '{relatedTerms}',
  jsonb_build_array(
    'exploding-gradients',
    'backpropagation',
    'residual-connection',
    'activation-function'
  ),
  true
),
updated_at = NOW()
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug = 'vanishing-gradients';

UPDATE content_items AS item
SET metadata = jsonb_set(
  COALESCE(item.metadata, '{}'::jsonb),
  '{relatedTerms}',
  jsonb_build_array(
    'vanishing-gradients',
    'exploding-gradients',
    'neural-network',
    'activation-function'
  ),
  true
),
updated_at = NOW()
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug = 'weight-initialization';

UPDATE content_items AS item
SET metadata = jsonb_set(
  COALESCE(item.metadata, '{}'::jsonb),
  '{relatedTerms}',
  jsonb_build_array(
    'softmax',
    'transformer',
    'pytorch',
    'cross-entropy',
    'activation-function'
  ),
  true
),
updated_at = NOW()
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug = 'logits';

-- The two merged pages now live inside the activation-function page.
DELETE FROM content_items
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug IN ('relu', 'sigmoid');
