-- Merge the LoRA, adapter-layers, IA3, prefix-tuning, and prompt-tuning glossary pages into the peft hub page, then delete the five merged rows.
--
-- The five pages were thin slices of the same concept (1,503-1,845 chars each). A reader searching "what is LoRA" landed on a page that explained LoRA and never said how it relates to the family it belongs to, and a reader searching "PEFT" got the umbrella framing plus a name-drop list with no per-method detail. The merged page keeps the existing opening and the PEFT-vs-LoRA / PEFT-vs-RAG framing, then covers each method in its own section (LoRA, adapter layers, prefix tuning, prompt tuning, IA3), plus a side-by-side table, a one-minute chooser, and failure modes.
--
-- Sources consulted on September 28 2026:
-- - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
--   (https://arxiv.org/abs/2106.09685)
-- - Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (adapters), 2019
--   (https://arxiv.org/abs/1902.00751)
-- - Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation", 2021
--   (https://arxiv.org/abs/2101.00190)
-- - Lester et al., "The Power of Scale for Parameter-Efficient Prompt Tuning", 2021
--   (https://arxiv.org/abs/2104.08691)
-- - Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (IA3), 2022
--   (https://arxiv.org/abs/2205.05638)
-- - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023
--   (https://arxiv.org/abs/2305.14314)
-- - HuggingFace PEFT documentation
--   (https://huggingface.co/docs/peft)
--
-- Body and blocks stay in sync, sources are attached, and metadata gains the merged keyword set plus related terms that all exist as glossary slugs. The lora, adapter-layers, ia3, prefix-tuning, and prompt-tuning slugs are dropped. No other glossary relatedTerms point at the deleted slugs and no article/guide bodies link to them, so the delete is clean apart from the hub row's own relatedTerms, which are rebuilt without the removed slugs.

WITH peft_entry AS (
  SELECT $body$PEFT stands for Parameter-Efficient Fine-Tuning. It is a category of techniques for adapting a pretrained model to a new task while training only a tiny fraction of the model's parameters. Instead of updating every weight in a large language model or foundation model, PEFT methods freeze most of the base model and learn small task-specific additions or modifications.

The motivation is practical. Full fine-tuning of large models is expensive in GPU memory, storage, and training time. It also creates a separate full model checkpoint for every task, which is wasteful when the base model remains mostly the same. PEFT methods reduce that cost by keeping the original model fixed and storing only the lightweight adaptation layers or vectors needed for the new behavior.

Five methods carry nearly all of modern practice - LoRA (Low-Rank Adaptation), adapter layers, prefix tuning, prompt tuning, and IA3. These approaches make different architectural tradeoffs, but they share the same goal: preserve most of the pretrained model while learning a much smaller set of task-specific parameters. This page covers the full family in one place: what each method trains, where it lives in the network, where it wins, and where it breaks.

## LoRA: the default

LoRA stands for Low-Rank Adaptation. It is the most widely used PEFT technique for adapting large pretrained models. Instead of updating a model's full weight matrices during [fine-tuning](/glossary/fine-tuning), LoRA freezes the original weights and learns two much smaller matrices whose product approximates the desired update.

The core idea is that many useful task-specific changes can be represented as a low-rank update rather than a full dense rewrite. If a weight matrix W would normally be updated by some large matrix Delta W, LoRA constrains that update to the form A x B where A and B are much smaller than W. This dramatically reduces the number of trainable parameters while preserving much of the benefit of full fine-tuning.

LoRA is especially common in [transformer](/glossary/transformer) models, where it is typically applied to attention and sometimes feed-forward projection matrices. Because the base model remains frozen, training requires less memory, and the learned LoRA weights can be saved as a compact adapter checkpoint rather than a full copy of the model.

One reason LoRA became so popular is operational simplicity. You can keep one base model and load different LoRA adapters for different tasks, styles, or domains. At inference time, the LoRA weights can either be applied dynamically or merged into the base weights ahead of deployment, depending on the serving setup.

LoRA is also the foundation for variants such as QLoRA, which combines low-rank adaptation with quantized base weights ([quantization](/glossary/quantization)) to make fine-tuning even more memory efficient. In practice, when people say they "fine-tuned" an open model on consumer hardware, they often mean they trained a LoRA rather than updating the full model.

## Adapter layers: the modular original

Adapter layers are one of the earliest and most influential PEFT methods. Instead of changing the full weights of a large pretrained model, adapters insert small trainable modules between or inside existing layers while keeping the original model frozen.

In transformer models, adapter layers are usually added after attention or feed-forward blocks. They often use a bottleneck structure: project the hidden state down to a smaller dimension, apply a nonlinearity, then project back up. Because only these small inserted layers are trained, the number of trainable parameters is far lower than in full fine-tuning.

Adapter layers are modular and easy to swap. You can keep one base model and load different adapters for sentiment analysis, legal drafting, medical Q&A, or code generation. They work well when you need many task-specific variants of the same model because each adapter checkpoint is much smaller than a full model copy. A company might keep one 7B base model for internal document work, then attach one adapter for finance summarization and another for customer support classification.

Adapters add new layers to the network, which can increase inference latency and architectural complexity compared with methods that add no extra computation at every forward pass. They can also be more cumbersome to integrate than LoRA in ecosystems where LoRA has become the default tooling standard.

## Prefix tuning: steering attention from inside

Prefix tuning adapts a frozen model by learning a set of trainable vectors, called prefixes, that are injected into the model's [attention](/glossary/attention-mechanism) mechanism. These prefixes are not ordinary text tokens. They are learned continuous representations that influence the model's behavior during generation.

In transformer language models, the learned prefix is typically attached to the key and value states used by attention. This gives the model a task-specific setup before it processes the actual user input, steering outputs without modifying the main pretrained weights. A team building a report generator might train one prefix for concise executive summaries and another for long-form analytical explanations, using the same base model under both modes.

Prefix tuning can be extremely parameter-efficient because the learned prefix is small relative to the full model. It keeps the base model fully frozen, which simplifies checkpoint management. But because it steers attention through learned prefixes rather than directly modifying model weights, it may be less expressive than LoRA for harder tasks, and it can be less intuitive to debug since the control signal lives in internal hidden states rather than in obvious weight updates or inserted layers.

## Prompt tuning: the lightest touch

Prompt tuning is a PEFT technique in which a frozen model is adapted by learning a small set of trainable embeddings that function like a soft prompt. These embeddings are prepended to the input at the embedding level rather than written as literal human-readable text.

The idea is the continuous cousin of [prompt engineering](/glossary/prompt-engineering): instead of hand-writing instructions such as "summarize this text in bullet points," the system learns a continuous prompt representation that the model responds to more reliably. Because only the prompt embeddings are trained, the number of trainable parameters is tiny. A company might learn one virtual prompt that makes a base model answer as a terse support bot and another that makes it respond as a formal legal assistant, without modifying the underlying weights.

Prompt tuning is one of the lightest-weight adaptation methods available. It is cheap to store, easy to swap, and attractive when you need very small task-specific checkpoints or when a large, already-capable model needs only a small nudge. But it is often less expressive than LoRA or adapter layers for harder tasks requiring deeper behavioral changes, and it can be fragile: performance may depend heavily on model scale, prompt length, and training setup, and on smaller models it may underperform more direct adaptation methods.

## IA3: rescaling instead of adding

IA3 stands for Infused Adapter by Inhibiting and Amplifying Inner Activations. It is a PEFT method that adapts a frozen model by learning small vectors that scale internal activations, rather than inserting full new layers or training low-rank matrix updates.

In transformer models, IA3 typically learns multiplicative rescaling factors for selected attention and feed-forward components. Instead of changing the full weight matrices, it changes how strongly certain channels are amplified or suppressed during computation. This keeps the number of trainable parameters extremely small, often even smaller than LoRA. A research team might use IA3 to adapt one [foundation model](/glossary/foundation-model) for biomedical question answering and another for software issue classification, storing only tiny learned scaling vectors for each variant.

IA3 is attractive when you want stronger adaptation than prompt tuning but still want a minimal update footprint, and it is cheap to store. But it is less widely supported in tooling than LoRA, harder to explain intuitively to newcomers, and in some settings scaling-only updates may limit expressiveness compared with richer methods like LoRA or adapter layers.

## The five side by side

| Method | What it trains | Where it lives | Watch out for |
| --- | --- | --- | --- |
| LoRA | two small low-rank matrices per target projection | attention and feed-forward weights, mergeable at inference | rank too low underfits, rank too high wastes the savings |
| Adapter layers | bottleneck modules inserted between layers | after attention or feed-forward blocks | extra latency on every forward pass, heavier integration than LoRA |
| Prefix tuning | learned prefix vectors for attention keys and values | inside attention states | less expressive than LoRA, hard to debug in hidden states |
| Prompt tuning | learned soft-prompt embeddings | input embedding level | fragile on small models, sensitive to scale, length, and setup |
| IA3 | learned rescaling vectors | attention and feed-forward activations | thin tooling, scaling-only updates can underfit deep changes |

## PEFT vs LoRA

The simplest distinction is: PEFT is the umbrella category, and LoRA is one specific method inside that category. Saying "I used PEFT" is like saying "I used a compression method"; saying "I used LoRA" is naming the exact technique. Not every PEFT method is LoRA, but every LoRA setup is a form of PEFT.

People often blur the two because LoRA became the default PEFT method for many open-model workflows. But adapter tuning, prompt tuning, prefix tuning, and IA3 are also PEFT methods, even though they work differently under the hood.

## PEFT vs Full Fine-Tuning vs Prompting/RAG

| Approach | What changes | Best for | Main tradeoff |
| --- | --- | --- | --- |
| Prompting | The instructions sent to the model at runtime. | Fast behavior changes, experiments, and tasks the base model already understands. | No durable model adaptation, performance can be fragile across prompts. |
| RAG | The context retrieved and supplied to the model. | Adding fresh or private knowledge without retraining the model. | Depends on retrieval quality and does not deeply change model behavior. |
| PEFT | Small trainable adapters, vectors, or low-rank updates attached to a mostly frozen model. | Domain adaptation, style tuning, task specialization, and maintaining many lightweight variants. | Usually less flexible than full fine-tuning and adds adapter management complexity. |
| Full fine-tuning | Most or all model weights. | Deep behavioral changes, high-stakes specialization, or cases where PEFT underperforms. | Expensive to train, store, validate, and serve. |

PEFT is one of the main reasons smaller teams can customize large models at all. It lowers hardware requirements, shortens experimentation cycles, and makes it easier to maintain many specialized variants of the same base model. In the open-source model ecosystem, PEFT checkpoints are often small enough to distribute independently from the original model weights.

## How to choose in one minute

- **General adaptation default:** LoRA. Best tooling, mergeable at inference, and the safest first try for style, domain, or task tuning.
- **Many swappable task variants on one frozen base:** adapter layers, or LoRA if the stack already serves LoRA adapters. Adapters win on modularity, LoRA wins on ecosystem.
- **Steering generation with minimal checkpoints:** prefix tuning for attention-level control, prompt tuning for the lightest embedding-level nudge on large models.
- **Smallest update footprint beyond prompting:** IA3, when scaling vectors are enough and the team can live outside LoRA tooling.
- **Only new facts or documents, no behavior change:** [retrieval-augmented generation](/glossary/retrieval-augmented-generation), not PEFT.
- **Deep capability change or a weak base model:** full [fine-tuning](/glossary/fine-tuning) or a different base model. No adapter compensates for a wrong foundation.

## Failure modes worth naming

- **Adapter-base mismatch.** Adapters are version-locked to their base checkpoint. Upgrading the base model without revalidating every adapter silently breaks behavior. Track the pair, not the adapter alone.
- **Rank and capacity mis-set.** LoRA rank too low underfits the task, while rank too high plus too many targets re-creates full fine-tuning costs without its flexibility. Start small and scale up only on validation signal.
- **Prompt and prefix fragility.** Short soft prompts on small models fail discontinuously with tiny setup changes. If results swing across seeds or lengths, the method is underpowered for the task. Move to LoRA or adapters.
- **Latency surprise.** Adapter layers add compute to every token. Benchmark serving with adapters attached, and prefer mergeable LoRA or vector-only methods when latency budgets are tight.
- **RAG confusion.** Training an adapter to memorize facts that retrieval could supply is slower, less editable, and harder to cite. PEFT changes behavior, retrieval supplies information.

## When Not to Use PEFT

PEFT is not always the right tool. If the model only needs access to new facts, documents, or product data, retrieval-augmented generation is often safer and easier than training an adapter. PEFT changes behavior; RAG supplies information.

PEFT can also be the wrong choice when the desired change is very deep. If the base model is fundamentally bad at the target task, uses the wrong language or modality, or needs a major change in reasoning style, full fine-tuning or a different base model may work better. A small adapter cannot reliably compensate for a weak foundation.

Operationally, PEFT adds its own complexity. Teams have to track which adapter belongs to which base model version, validate adapter compatibility after model upgrades, decide whether to merge adapters for inference, and manage multi-adapter serving if many tasks share one base model. Those costs are much smaller than full fine-tuning, but they are not zero.

A useful mental model is that full fine-tuning rewrites the whole book, while PEFT adds an annotated layer on top of it. The base knowledge stays in place; the adaptation tells the model how to behave differently for a narrower job.$body$::text AS body
)
UPDATE content_items AS item
SET
  summary = 'Parameter-efficient fine-tuning for large models - LoRA, adapter layers, prefix tuning, prompt tuning, and IA3 - training small adapters instead of the full network.',
  body = peft_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', peft_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Hu et al: LoRA - Low-Rank Adaptation of Large Language Models (2021)', 'url', 'https://arxiv.org/abs/2106.09685'),
    jsonb_build_object('title', 'Houlsby et al: Parameter-Efficient Transfer Learning for NLP (2019)', 'url', 'https://arxiv.org/abs/1902.00751'),
    jsonb_build_object('title', 'Li and Liang: Prefix-Tuning - Optimizing Continuous Prompts for Generation (2021)', 'url', 'https://arxiv.org/abs/2101.00190'),
    jsonb_build_object('title', 'Lester et al: The Power of Scale for Parameter-Efficient Prompt Tuning (2021)', 'url', 'https://arxiv.org/abs/2104.08691'),
    jsonb_build_object('title', 'Liu et al: Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning (2022)', 'url', 'https://arxiv.org/abs/2205.05638'),
    jsonb_build_object('title', 'Dettmers et al: QLoRA - Efficient Finetuning of Quantized LLMs (2023)', 'url', 'https://arxiv.org/abs/2305.14314'),
    jsonb_build_object('title', 'HuggingFace PEFT documentation', 'url', 'https://huggingface.co/docs/peft')
  ),
  metadata = jsonb_set(
    jsonb_set(
      jsonb_set(
        COALESCE(item.metadata, '{}'::jsonb),
        '{seoKeywords}',
        jsonb_build_array(
          'what is peft',
          'peft explained',
          'parameter efficient fine tuning',
          'peft vs fine tuning',
          'what is lora',
          'lora explained',
          'lora vs peft',
          'qlora explained',
          'adapter layers vs lora',
          'what is prefix tuning',
          'prefix tuning vs prompt tuning',
          'what is prompt tuning',
          'soft prompt tuning',
          'what is ia3',
          'ia3 vs lora',
          'which peft method to use'
        ),
        true
      ),
      '{seoDescription}',
      '"Parameter-efficient fine-tuning explained: LoRA, adapters, prefix, prompt tuning, and IA3 compared with trade-offs and a one-minute chooser."'::jsonb,
      true
    ),
    '{relatedTerms}',
    jsonb_build_array(
      'fine-tuning',
      'transfer-learning',
      'transformer',
      'foundation-model',
      'large-language-model',
      'quantization',
      'attention-mechanism',
      'prompt-engineering',
      'retrieval-augmented-generation',
      'overfitting',
      'hyperparameter',
      'loss-function'
    ),
    true
  ),
  published_at = DATE '2026-09-28',
  updated_at = NOW()
FROM peft_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'peft'
  AND item.parent_slug = '';

-- The five merged pages now live inside the peft page. No other glossary relatedTerms point at the deleted slugs and no article/guide bodies link to them.
DELETE FROM content_items
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug IN ('lora', 'adapter-layers', 'ia3', 'prefix-tuning', 'prompt-tuning');
