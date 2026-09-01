WITH glossary_updates(slug, title, summary, body, sources, metadata) AS (
  VALUES
  (
    'long-short-term-memory',
    'Long Short-Term Memory',
    'A gated recurrent neural network architecture that carries a learned cell state through a sequence, helping it retain and update information over many time steps.',
    $lstm$
Long Short-Term Memory, usually shortened to **LSTM**, is a type of recurrent neural network designed to learn from ordered data while preserving useful information across many time steps. It processes a sequence one step at a time and carries two forms of state forward: a **cell state** that acts as the longer-lived memory path and a **hidden state** that represents the current output and short-term working state.

Sepp Hochreiter and Jürgen Schmidhuber introduced LSTM in 1997 to address insufficient, decaying error flow in recurrent networks. The commonly taught modern LSTM includes a forget gate added by Felix Gers, Jürgen Schmidhuber, and Fred Cummins in later work. This historical distinction matters because “the LSTM” now usually means the updated input-gate, forget-gate, and output-gate design rather than the exact 1997 cell.

## Why ordinary RNNs forget

A recurrent neural network reuses the same transformation at every position in a sequence. During training, backpropagation through time multiplies gradients through those repeated steps. If the relevant derivatives are mostly smaller than one, the gradient can shrink toward zero before it reaches an early event. If they are too large, it can explode.

The result is that a basic RNN may learn nearby relationships but struggle to connect events separated by many steps. For example, it may need an early subject to interpret a verb much later, an old sensor reading to explain a current anomaly, or an earlier musical motif to predict the next phrase.

LSTM creates a more direct, additive path through the cell state. Learned gates decide what to retain, write, and reveal. This improves gradient flow and gives the model a practical mechanism for selective memory.

## How an LSTM cell works

At time step **t**, an LSTM receives the current input **x_t**, the previous hidden state **h_(t-1)**, and the previous cell state **c_(t-1)**. It computes several vectors with learned weights and biases:

1. The **forget gate** decides how much of the previous cell state to keep.
2. The **input gate** decides how much new candidate information to write.
3. A candidate update proposes new content derived from the input and previous hidden state.
4. The cell combines retained old state with gated candidate content to produce the new cell state.
5. The **output gate** decides which part of that cell state becomes the new hidden state.

A common formulation is:

**f_t = sigmoid(W_f · [h_(t-1), x_t] + b_f)**

**i_t = sigmoid(W_i · [h_(t-1), x_t] + b_i)**

**g_t = tanh(W_g · [h_(t-1), x_t] + b_g)**

**c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t**

**o_t = sigmoid(W_o · [h_(t-1), x_t] + b_o)**

**h_t = o_t ⊙ tanh(c_t)**

Here, **⊙** means element-by-element multiplication. Each gate is a vector, so the cell can retain one feature, erase another, and write a third at the same time. A sigmoid output near zero suppresses a component, while a value near one passes most of it through. Gates are soft controls, not binary switches.

## The three LSTM gates

| Gate | Main question | Effect |
| --- | --- | --- |
| Forget gate | What old information should remain? | Multiplies the previous cell state before it is carried forward |
| Input gate | Which candidate information should be written now? | Controls the new contribution to the cell state |
| Output gate | Which stored information should affect the current output? | Controls the hidden state exposed to the next layer or time step |

The cell state is not a database of readable facts. It is a learned continuous vector whose dimensions acquire whatever internal meanings help minimize the training loss. The model does not receive explicit rules saying which gate should remember a name or forget a noise spike.

## Cell state vs hidden state

| State | Role | Passed forward? |
| --- | --- | --- |
| Cell state **c_t** | Longer-lived internal memory path updated additively through gates | Yes, to the next time step |
| Hidden state **h_t** | Current exposed representation and recurrent output | Yes, to the next step and usually to the next layer or prediction head |

“Long short-term memory” does not mean unlimited permanent memory. The name describes a mechanism for keeping information longer than a basic short-term recurrent state. Capacity is finite, state can be overwritten, training often uses truncated sequences, and very long dependencies can still be lost.

## How LSTMs are trained

LSTMs are usually trained with **backpropagation through time**. The recurrent computation is unrolled across a sequence, a loss is calculated from one or more outputs, and gradients flow backward through the unrolled steps and shared parameters. An optimizer then updates the weights.

Long sequences increase memory and compute because activations from many steps may be needed for the backward pass. **Truncated backpropagation through time** limits the number of steps used for each gradient update while carrying state between chunks. Gradient clipping is commonly used to control exploding gradients.

Training data can support several input-output patterns:

- **Many-to-one**, such as classifying an entire time series or sentence
- **One-to-many**, such as generating a sequence from one context vector
- **Many-to-many aligned**, such as labeling every audio frame or token
- **Encoder-decoder**, where one sequence is encoded before another sequence is generated

Padding, masking, sequence length, state resets, and whether state crosses batch boundaries are part of the model definition. A stateful LSTM can carry state between consecutive chunks, but it must not accidentally leak information between unrelated examples.

## LSTM variants

| Variant | What changes | Typical reason to use it |
| --- | --- | --- |
| Stacked LSTM | Places multiple recurrent layers on top of one another | Learns higher-level temporal representations |
| Bidirectional LSTM | Processes the sequence forward and backward | Uses both past and future context when the full sequence is available |
| Peephole LSTM | Lets gates inspect the cell state directly | Gives timing-sensitive tasks additional state access |
| Convolutional LSTM | Replaces dense transformations with convolutions | Preserves spatial structure in video, weather, or image sequences |
| Projected LSTM | Projects the hidden output to a smaller dimension | Reduces recurrent computation and output size |

A bidirectional LSTM is unsuitable for strictly causal streaming when future inputs do not yet exist. It can work well for offline tagging, transcription, or analysis where the complete sequence is available.

## LSTM vs RNN vs GRU vs transformer

| Architecture | Memory mechanism | Parallelism across sequence positions | Main tradeoff |
| --- | --- | --- | --- |
| Basic RNN | One recurrent hidden state | Low | Simple and small, but weak on long dependencies |
| LSTM | Cell state plus input, forget, and output gates | Low | Stronger selective memory with more parameters and computation |
| GRU | One gated state with update and reset gates | Low | Simpler than LSTM and often similarly effective, but not universally better |
| Transformer | Attention over token or patch representations | High during training | Scales well and connects distant positions directly, but attention cost and context storage can be large |

Transformers displaced LSTMs as the default architecture for large language models because attention makes training much more parallel and gives each position a direct path to other positions. That does not make LSTMs obsolete. The better choice depends on data size, latency, memory, sequence length, hardware, and whether the application is streaming.

## Where LSTMs are still useful

LSTMs remain practical for time-series forecasting, anomaly detection, speech and handwriting pipelines, biosignals, industrial sensors, financial sequences, embedded systems, and streaming classification. They can be attractive when data arrives one step at a time, the model must maintain a fixed-size state, training data is modest, or a compact recurrent model is easier to deploy than an attention model.

They are also useful as baselines. A complicated transformer that barely beats a tuned LSTM may not justify its extra memory, latency, or operational complexity.

## Limitations

- Recurrence makes sequence positions difficult to process fully in parallel during training.
- Long unrolled sequences consume memory and increase training time.
- LSTMs mitigate vanishing gradients but do not guarantee retention over arbitrary distances.
- Cell capacity is finite and gates can learn to keep irrelevant information or forget important signals.
- Very long context, retrieval, or global pairwise relationships may be easier for attention-based models.
- Hidden state makes batching, state resets, masking, and deployment behavior more complicated.
- Results are sensitive to sequence construction, scaling, missing values, and temporal leakage.

For forecasting, random train-test splits can leak future information into training. Evaluation should preserve time order and compare against simple seasonal, persistence, and statistical baselines.

## Bottom line

An LSTM is a recurrent neural network with a controlled memory path. Its gates learn how much old state to retain, how much new information to write, and what to expose at each step. It made long-range sequence learning far more practical, but its memory is selective and finite, and transformers are often a better fit for large-scale parallel sequence modeling.
$lstm$::text,
    jsonb_build_array(
      jsonb_build_object('title', 'Long Short-Term Memory — Hochreiter and Schmidhuber (1997)', 'url', 'https://doi.org/10.1162/neco.1997.9.8.1735'),
      jsonb_build_object('title', 'Learning to Forget: Continual Prediction with LSTM — Gers, Schmidhuber, and Cummins', 'url', 'https://doi.org/10.1162/089976600300015015'),
      jsonb_build_object('title', 'LSTM: A Search Space Odyssey', 'url', 'https://arxiv.org/abs/1503.04069'),
      jsonb_build_object('title', 'Sequence to Sequence Learning with Neural Networks', 'url', 'https://arxiv.org/abs/1409.3215'),
      jsonb_build_object('title', 'Attention Is All You Need', 'url', 'https://arxiv.org/abs/1706.03762')
    ),
    jsonb_build_object(
      'category', 'Models & Architectures',
      'relatedTerms', jsonb_build_array('recurrent-neural-network', 'neural-network', 'deep-learning', 'transformer', 'gradient-descent', 'backpropagation'),
      'analogy', 'An LSTM is like a notebook with three learned controls: an eraser for old notes, a pen for new notes, and a window deciding which notes to show right now.',
      'seoDescription', 'Long short-term memory is a gated recurrent neural network. Learn how cell states and gates work and how LSTMs compare with GRUs and transformers.',
      'seoKeywords', jsonb_build_array('what is LSTM', 'long short-term memory explained', 'how LSTM works', 'LSTM gates explained', 'LSTM cell state vs hidden state', 'LSTM equations', 'LSTM vs RNN', 'LSTM vs GRU', 'LSTM vs transformer', 'vanishing gradient LSTM', 'backpropagation through time', 'LSTM time series forecasting')
    )
  ),
  (
    'neural-network',
    'Neural Network',
    'A parameterized computational model that learns layered transformations from data, mapping inputs to predictions, representations, generated content, or actions.',
    $network$
An artificial neural network is a parameterized computational model that learns a mapping from inputs to outputs. It is built from connected operations commonly called neurons, units, or layers. Training adjusts numerical parameters called weights and biases so the network performs a task such as classification, prediction, generation, control, retrieval, or representation learning.

Neural networks were loosely inspired by biological nervous systems, but modern networks are engineered mathematical systems rather than simulations of real brains. A unit usually computes a weighted combination of numbers, adds a bias, and applies an activation function. Large networks compose millions or billions of these simple transformations into a flexible function.

## The basic artificial neuron

For an input vector **x**, a simple neuron computes:

**z = w · x + b**

**y = φ(z)**

The weight vector **w** controls the influence of each input, **b** is a learned bias, and **φ** is an activation function such as ReLU, sigmoid, tanh, GELU, or SiLU. Without nonlinear activations, stacking ordinary linear layers collapses into one linear transformation and cannot represent the nonlinear relationships that make deep networks useful.

A neuron does not independently “understand” a concept. Meaning is distributed across activations, weights, layers, and the training objective. Some units respond to recognizable patterns, while others participate in representations that make sense only as part of the larger network.

## Layers in a neural network

| Layer | Role | Example |
| --- | --- | --- |
| Input representation | Converts raw or encoded data into numbers the network can process | Pixels, audio samples, token embeddings, sensor values |
| Hidden layer | Learns intermediate transformations and features | Edges, shapes, syntax, latent factors, temporal patterns |
| Output head | Maps the final representation to the task output | Class probabilities, a numeric forecast, next-token logits |

A **shallow** network has few learned layers. A **deep neural network** has many layers, allowing it to build hierarchical representations. Depth is not automatically better. Architecture, data, optimization, regularization, compute, and the match between the model and task determine performance.

## How a neural network makes a prediction

During the **forward pass**, input values move through the network in dependency order. Each layer uses its current parameters to produce activations for the next layer. The final layer produces a prediction or representation.

For a classifier, the output might be a score for each class. For a language model, it is commonly a distribution over the next token. For a diffusion model, the network may predict noise or another denoising target. During **inference**, the network performs this forward computation with fixed parameters, plus any decoding, sampling, retrieval, or tool logic supplied by the surrounding system.

## How a neural network learns

A typical supervised training step has four parts:

1. **Forward propagation** computes predictions from a batch of inputs.
2. A **loss function** measures the error between the predictions and training targets.
3. **Backpropagation** applies the chain rule through the computational graph to calculate how the loss changes with each parameter.
4. An **optimizer** such as stochastic gradient descent or Adam uses those gradients to update the parameters.

Backpropagation and gradient descent are related but not identical. Backpropagation computes gradients. The optimizer decides how to use them. Training repeats this process across many batches and epochs.

Networks can also learn through self-supervised objectives, reinforcement-learning signals, contrastive learning, reconstruction, distillation, preference optimization, or combinations of methods. The common idea is that a numerical objective supplies a signal for changing parameters.

## Parameters vs hyperparameters vs activations

| Term | Meaning | Examples |
| --- | --- | --- |
| Parameters | Values learned during training | Weights, biases, embedding tables, normalization scales |
| Hyperparameters | Settings chosen outside ordinary gradient updates | Learning rate, layer count, width, batch size, dropout rate |
| Activations | Intermediate values produced for a particular input | Hidden-layer outputs, attention values, feature maps |
| Gradients | Derivatives showing how a small parameter change affects the loss | Weight and bias gradients from backpropagation |

The phrase “a 7-billion-parameter model” refers to learned parameter count, not the number of neurons, training examples, or active operations per token.

## Major neural network architectures

| Architecture | Structural idea | Common uses |
| --- | --- | --- |
| Multilayer perceptron or MLP | Feedforward stack of dense layers | Tabular data, prediction heads, general function approximation |
| Convolutional neural network or CNN | Reuses local filters across positions | Images, audio, video, spatial signals |
| Recurrent neural network or RNN | Carries hidden state through a sequence | Streaming and sequential data |
| LSTM or GRU | Adds gates to recurrent state | Time series, speech, compact sequence models |
| Transformer | Uses attention and position-aware representations | Language, vision, audio, multimodal and generative models |
| Graph neural network or GNN | Passes messages along graph edges | Molecules, networks, recommendations, relational data |
| Autoencoder | Compresses and reconstructs inputs | Representation learning, denoising, anomaly detection |
| Generative adversarial network or GAN | Trains a generator against a discriminator | Image generation and distribution matching |
| Diffusion network | Learns iterative denoising or flow transformations | Image, audio, video, and structured generation |

These categories overlap. A multimodal system may combine convolution, attention, recurrence, and several specialized heads. “Neural network” is the umbrella term, not one fixed architecture.

## Neural network vs machine learning vs deep learning

| Term | Scope |
| --- | --- |
| Artificial intelligence | Broad field of systems that perform tasks associated with perception, reasoning, language, planning, or action |
| Machine learning | Methods that improve task performance from data or experience |
| Neural network | One family of parameterized machine-learning models |
| Deep learning | Machine learning built around neural networks with multiple representation-learning layers |

Not all AI uses machine learning, not all machine learning uses neural networks, and not every neural network is especially deep.

## Training, validation, and test data

Training data supplies the examples used to update parameters. A validation set helps choose architecture, hyperparameters, checkpoints, and stopping rules. A test set estimates performance after those choices are fixed.

If test information influences training or model selection, the result is contaminated. If near-duplicate examples occur across splits, performance can look better than real generalization. Data quality, coverage, labeling, preprocessing, and split strategy often matter as much as adding more layers.

## Why neural networks are powerful

Neural networks can learn intermediate representations rather than relying entirely on hand-written features. Early layers may detect local or simple patterns, while later layers combine them into task-relevant structures. Parameter sharing, convolution, recurrence, and attention add useful assumptions for particular data types.

Universal approximation results show that certain neural networks can represent broad classes of functions when given sufficient capacity. They do **not** guarantee that training will find the right parameters, that the required network is practical, that finite data is sufficient, or that the model will generalize safely outside its training distribution.

Scale has made neural networks unusually flexible. The same broad training machinery can support text, images, audio, proteins, robotics, and scientific data. Scale also increases demand for compute, memory, data curation, distributed training, evaluation, and deployment engineering.

## Common training problems

- **Underfitting** occurs when the model or training process cannot capture the relevant pattern.
- **Overfitting** occurs when training performance improves while unseen-data performance does not.
- **Vanishing or exploding gradients** make early layers or long recurrent paths difficult to train.
- **Dead or saturated activations** can reduce useful gradient flow.
- **Poor conditioning** can make optimization slow or unstable.
- **Data leakage** allows information from validation or test examples to influence training.
- **Spurious correlations** let the model exploit shortcuts that fail outside the dataset.

Initialization, normalization, residual connections, activation choice, learning-rate schedules, regularization, larger or cleaner datasets, and architecture design can help. None replaces evaluation on data that reflects the intended use.

## Limitations and risks

A neural network learns statistical structure from its objective and data. It does not automatically learn causality, truth, fairness, robustness, or human intent. It can be confidently wrong, sensitive to distribution shift, vulnerable to adversarial inputs, poorly calibrated, or reliant on correlations that developers did not notice.

Large models can also be difficult to interpret. Knowing every weight does not provide a simple human explanation for a prediction. Post-hoc explanations may be useful evidence but should not be mistaken for complete access to the model's reasoning.

Deployment adds system-level risks. Preprocessing, retrieval, prompts, tools, thresholds, caching, quantization, and hardware can change behavior even when the network weights stay fixed. A production evaluation should test the complete pipeline, not only the model file.

## How to evaluate a neural network

Choose metrics that match the task and cost of errors. Accuracy may suit balanced classification, while precision and recall matter when false positives and false negatives have different consequences. Regression may use absolute or squared error. Generative systems need task-specific quality, factuality, safety, diversity, latency, and cost measurements.

Compare against simple baselines, report uncertainty, inspect subgroup and failure-case performance, and test robustness under realistic shifts. A larger neural network is not better if it costs more, responds slower, or fails the cases that matter.

## Bottom line

A neural network is a trainable composition of weighted transformations. The forward pass produces an output, the loss measures error, backpropagation computes gradients, and an optimizer updates parameters. Its capability comes from the interaction of architecture, data, objective, optimization, and scale, not from neurons alone.
$network$::text,
    jsonb_build_array(
      jsonb_build_object('title', 'Learning Representations by Back-Propagating Errors — Rumelhart, Hinton, and Williams', 'url', 'https://doi.org/10.1038/323533a0'),
      jsonb_build_object('title', 'Deep Learning — LeCun, Bengio, and Hinton', 'url', 'https://doi.org/10.1038/nature14539'),
      jsonb_build_object('title', 'Deep Learning — Goodfellow, Bengio, and Courville', 'url', 'https://www.deeplearningbook.org/'),
      jsonb_build_object('title', 'Multilayer Feedforward Networks Are Universal Approximators', 'url', 'https://doi.org/10.1016/0893-6080(89)90020-8')
    ),
    jsonb_build_object(
      'category', 'Foundations',
      'relatedTerms', jsonb_build_array('artificial-intelligence', 'machine-learning', 'deep-learning', 'activation-function', 'backpropagation', 'gradient-descent', 'loss-function', 'convolutional-neural-network', 'recurrent-neural-network', 'transformer'),
      'analogy', 'A neural network is like a layered control board whose dials are tuned from examples: each layer transforms the signal, and feedback adjusts millions of dials to reduce error.',
      'seoDescription', 'A neural network learns weighted transformations from data. Understand neurons, layers, activation functions, backpropagation, training, and inference.',
      'seoKeywords', jsonb_build_array('what is a neural network', 'neural network explained', 'how neural networks work', 'artificial neural network', 'neural network layers', 'artificial neuron equation', 'forward propagation', 'backpropagation explained', 'neural network training', 'parameters vs hyperparameters', 'types of neural networks', 'neural network vs machine learning', 'deep neural network', 'neural network inference')
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
  published_at = DATE '2026-09-01',
  updated_at = NOW()
FROM glossary_updates
WHERE item.kind = 'glossary'
  AND item.slug = glossary_updates.slug
  AND item.parent_slug = '';
