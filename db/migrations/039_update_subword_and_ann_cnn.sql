WITH subword_entry AS (
  SELECT $body$
Subword tokenization converts text into reusable pieces that are usually smaller than words but larger than individual characters or bytes. A common word may remain one token, while a rare word is split into several known pieces. The resulting tokens are mapped to integer IDs before a language model processes them.

For example, one tokenizer might split **unhappiness** into `un`, `happi`, and `ness`, while another keeps `unhappiness` intact or uses different pieces. There is no universal correct segmentation. The vocabulary, training corpus, normalization rules, pre-tokenizer, and segmentation algorithm determine the result.

Subword tokenization sits between word-level and character-level approaches. It keeps the vocabulary much smaller than a dictionary of every possible word while usually producing shorter sequences than character or byte tokenization.

## Why language models use subwords

Natural language has an open vocabulary. Names, spelling variants, inflections, compounds, code, URLs, and newly coined words appear constantly. A word-level tokenizer either needs an enormous vocabulary or replaces unseen words with an unknown token.

Subword tokenization represents a rare expression as a sequence of familiar pieces. That allows a fixed vocabulary to encode far more text and lets related forms share some parameters. A model can reuse the piece for `play` across `play`, `player`, `playing`, and `replay`, although learned pieces do not always align with human morphemes.

The tradeoff is sequence length. Frequent strings use few tokens, while rare words, unusual scripts, misspellings, or code may fragment into many tokens. More tokens consume context-window capacity and usually increase inference cost and latency.

## Word vs subword vs character vs byte tokenization

| Unit | Example behavior | Main advantage | Main limitation |
| --- | --- | --- | --- |
| Word | Treats `tokenization` as one item | Short, readable sequences for known words | Very large vocabulary and unknown-word problem |
| Subword | May split it into `token` and `ization` | Balances vocabulary size and sequence length | Segmentation and efficiency vary across languages and domains |
| Character | Splits it into individual written characters | Small vocabulary and flexible spelling | Much longer sequences |
| Byte | Represents encoded bytes from a fixed set | Can encode arbitrary byte sequences without an unknown character | Human characters may require multiple bytes and sequences can become long |

Modern tokenizers often combine these ideas. Byte-level BPE begins from bytes and merges frequent byte sequences. A SentencePiece model can use BPE or Unigram over Unicode text and may enable byte fallback. “Subword tokenizer” describes the resulting units, not one mandatory implementation.

## The tokenization pipeline

A production tokenizer usually has several stages:

1. **Normalization** may standardize Unicode, case, accents, or whitespace.
2. **Pre-tokenization** may identify words, punctuation, whitespace, bytes, or other initial regions.
3. **Subword segmentation** breaks those regions into vocabulary pieces.
4. **Special-token insertion** adds markers such as beginning, end, separator, padding, or role tokens.
5. **ID conversion** maps each token string to a vocabulary index.

Detokenization reverses the mapping well enough to produce text, but exact round trips depend on the tokenizer design. Normalization that lowercases or removes accents is intentionally lossy. Byte-aware tokenizers can preserve arbitrary input more reliably, while word pre-tokenization may discard distinctions unless whitespace is explicitly represented.

## BPE vs WordPiece vs Unigram vs SentencePiece

| Method or tool | How the vocabulary is learned | How text is segmented | Important distinction |
| --- | --- | --- | --- |
| Byte Pair Encoding or BPE | Repeatedly adds frequent adjacent-symbol merges | Applies the learned merge rules or equivalent ranks | Adapted from compression for open-vocabulary neural translation |
| WordPiece | Builds a vocabulary using a likelihood- or utility-oriented merge criterion | Commonly uses greedy longest-match-first segmentation | Associated with BERT-family tokenizers, though implementations differ |
| Unigram language model | Starts with many candidate pieces and removes pieces while optimizing a probabilistic objective | Chooses a high-probability segmentation, often with dynamic programming | Supports multiple plausible segmentations and sampling |
| SentencePiece | Toolkit that trains directly from raw text with whitespace represented explicitly | Implements BPE or Unigram models | SentencePiece is not itself a fourth merge algorithm |
| Byte-level BPE | Starts from byte symbols and learns frequent merges | Encodes any byte sequence through bytes and learned byte groups | Avoids character-level unknowns at the cost of possible fragmentation |

Names alone do not fully specify behavior. Two BPE tokenizers can use different normalization, regex pre-tokenization, byte mappings, merge vocabularies, and special tokens and therefore produce different IDs for the same text.

## How BPE tokenization works

The neural-text version of BPE begins with a base symbol vocabulary, often characters or bytes. During tokenizer training, it counts adjacent pairs in a corpus, merges a selected frequent pair into a new symbol, and repeats until reaching the chosen merge count or vocabulary size.

If `low`, `lower`, and `lowest` occur frequently, the procedure may learn pieces such as `low`, `er`, and `est`. At encoding time, learned merge priorities combine the base symbols into available pieces.

BPE optimizes corpus compression or frequency structure rather than linguistic correctness. It can discover useful stems and affixes, but it can also create fragments that have no independent meaning.

## How WordPiece tokenization works

WordPiece also builds a fixed vocabulary of reusable pieces, but its original and later descriptions use a language-model or likelihood-oriented criterion rather than simply selecting the most frequent pair. BERT-style implementations commonly pre-tokenize text and use greedy longest-match-first encoding.

Continuation markers are implementation details. BERT tokenizers often show `##` before pieces that continue a word. The marker is part of the tokenizer's vocabulary notation, not a universal subword symbol.

Some WordPiece implementations return an unknown token when a word cannot be decomposed into available pieces. WordPiece does not inherently guarantee arbitrary Unicode or byte coverage.

## How Unigram tokenization works

The Unigram method begins with a large candidate vocabulary and estimates a probability for each piece. It repeatedly removes pieces whose absence least harms the likelihood objective until reaching the target vocabulary size.

One string can have several valid segmentations. Encoding can select the best-scoring path, while training-time subword regularization can sample alternative paths. Exposing a model to multiple segmentations may improve robustness instead of treating one arbitrary boundary choice as permanent.

## What SentencePiece does

SentencePiece is an open-source tokenizer and detokenizer that can train directly from raw sentences without requiring a language-specific word splitter. It treats whitespace as an ordinary symbol, commonly displayed as `▁`, which makes word boundaries visible inside token pieces and supports unambiguous detokenization under its configured normalization.

SentencePiece supports BPE and Unigram models. Features such as byte fallback, vocabulary size, normalization rules, reserved IDs, and special tokens are configuration choices. Saying that a model “uses SentencePiece” is incomplete unless the model type and tokenizer files are identified.

## Training a tokenizer vs using a tokenizer

**Tokenizer training** learns a vocabulary and segmentation rules from a corpus. It decides which strings deserve dedicated tokens under a fixed vocabulary budget.

**Encoding** applies the frozen tokenizer to new text and returns token IDs. The language model is then trained against those IDs. Once model training begins, changing the token-to-ID mapping makes the existing embedding and output layers incompatible unless the model is deliberately adapted.

Applications should load the exact tokenizer version shipped with the model. A vocabulary with the same size is not enough. Token ID 1234 can represent completely different text in another tokenizer.

## Vocabulary size tradeoffs

| Smaller vocabulary | Larger vocabulary |
| --- | --- |
| Smaller embedding and output matrices | Larger parameter and memory cost in embedding and output layers |
| More splitting and longer sequences | Frequent strings can use fewer tokens |
| More sharing between related strings | More dedicated representations for whole strings |
| Better coverage per vocabulary slot | Rare dedicated tokens may receive little training |

The best size depends on language coverage, model architecture, training data, context length, compute, and deployment goals. Vocabulary size cannot be optimized independently from sequence length because attention and decoding cost operate over tokens.

## Tokens are not words

A token may be a whole word, part of a word, punctuation plus whitespace, several characters, one byte, or a special control symbol. Token counts therefore cannot be estimated reliably by counting words.

Whitespace and capitalization can change segmentation. `apple`, ` Apple`, and `APPLE` may have different IDs. Code indentation, JSON punctuation, long numbers, and URLs can be especially token-heavy. Emoji and complex Unicode sequences may be one token, several character pieces, or many byte tokens.

The visible token strings shown by debugging tools may use markers such as `##`, `Ġ`, or `▁`. These markers reflect tokenizer conventions for continuation or preceding whitespace. They are not normally literal characters the user typed.

## Subword tokenization and context windows

Model context windows are measured in tokens, not characters or words. A prompt that tokenizes inefficiently reaches the limit sooner. If an API charges per input or output token, the tokenizer directly affects cost.

Comparing a “128K context window” across models does not mean both accept the same amount of text. Different tokenizers can produce different token counts for the same document, especially for non-English languages, source code, mathematics, and structured data.

Output latency is also tied to tokens. A word split into three generated tokens needs three decoding steps in an autoregressive model, unless the serving system uses another acceleration method.

## Multilingual tokenization and fertility

**Tokenization fertility** is the average number of tokens produced for a word, character span, or other reference unit. Higher fertility means more fragmentation. A vocabulary trained mainly on one language may encode that language compactly while splitting another script into many pieces.

Fragmentation affects context capacity, cost, latency, and the amount of training signal available per semantic unit. Multilingual tokenizer design may use balanced sampling, larger vocabularies, byte fallback, or language-aware analysis, but improvements for one language can consume vocabulary capacity needed by another.

Evaluate token counts across the actual languages and domains the model will serve. An English average does not establish efficiency for Hindi, Arabic, Japanese, code, or mixed-script text.

## Unknown tokens and byte fallback

Subword tokenization reduces the unknown-word problem but does not automatically eliminate it. A character-based vocabulary can still encounter a Unicode character it never included. A WordPiece model may emit `[UNK]` for an undecomposable input.

Byte-level tokenization guarantees representability because any encoded text reduces to bytes from a fixed set. SentencePiece can also be configured with byte fallback for otherwise unknown characters. The cost of guaranteed coverage can be multiple tokens for one visible character.

## Subwords vs embeddings

The tokenizer converts text into discrete token IDs. The embedding layer maps those IDs to continuous vectors. Tokenization is not semantic search, and the pieces do not arrive with fixed meanings. Their representations are learned with the language model.

The same visible piece can have a context-dependent internal representation after passing through transformer layers. Conversely, a word split into several subwords can still be represented as one concept through contextual processing.

## Security and reliability issues

- Unicode lookalikes can appear visually identical while producing different token sequences.
- Invisible characters and unusual whitespace can change prompts or bypass naive string checks.
- Normalization can merge distinctions an application intended to preserve.
- Token-based truncation can remove instructions or evidence from the beginning or end of a prompt.
- Splitting control-like text into unexpected pieces can expose differences between text filters and model input.
- Decoding individual token bytes may produce invalid text even when the full sequence decodes correctly.

Security filters should operate on carefully normalized text while also considering the exact tokenized input sent to the model. Never assume one token equals one safe character or one human-readable word.

## How to evaluate a tokenizer

Useful measurements include vocabulary size, token count per byte or character, fertility by language and domain, unknown-token rate, byte-fallback rate, round-trip behavior, tokenizer speed, and downstream model quality. Also measure embedding and output-layer parameter cost and the sequence lengths seen during training and inference.

A tokenizer that compresses English prose well may perform poorly on code or another language. The right evaluation corpus should resemble production input rather than the tokenizer's training sample alone.

## Bottom line

Subword tokenization gives language models a fixed vocabulary without requiring every possible word to be a token. BPE, WordPiece, and Unigram make different vocabulary and segmentation choices, while SentencePiece is a toolkit that can implement BPE or Unigram directly from raw text. Those choices shape context usage, cost, multilingual efficiency, and what the model learns, so the tokenizer is part of the model rather than interchangeable preprocessing.
$body$::text AS body
)
UPDATE content_items AS item
SET
  title = 'Subword Tokenization',
  summary = 'A text-encoding strategy that represents frequent strings as whole tokens and splits rarer text into reusable character- or byte-derived pieces.',
  body = subword_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', subword_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Neural Machine Translation of Rare Words with Subword Units', 'url', 'https://arxiv.org/abs/1508.07909'),
    jsonb_build_object('title', 'SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer', 'url', 'https://arxiv.org/abs/1808.06226'),
    jsonb_build_object('title', 'Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates', 'url', 'https://arxiv.org/abs/1804.10959'),
    jsonb_build_object('title', 'Japanese and Korean Voice Search — the WordPiece paper', 'url', 'https://research.google/pubs/japanese-and-korean-voice-search/'),
    jsonb_build_object('title', 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 'url', 'https://arxiv.org/abs/1810.04805'),
    jsonb_build_object('title', 'Neural Machine Translation with Byte-Level Subwords', 'url', 'https://arxiv.org/abs/1909.03341')
  ),
  metadata = COALESCE(item.metadata, '{}'::jsonb) || jsonb_build_object(
    'category', 'Language, Vision & Retrieval',
    'relatedTerms', jsonb_build_array('tokenization', 'byte-pair-encoding', 'token', 'language-modeling', 'transformer', 'embedding', 'context-window', 'large-language-model'),
    'analogy', 'Subword tokenization is like building words from a compact box of reusable letter clusters: common words stay whole, while rare ones are assembled from familiar pieces.',
    'seoDescription', 'Subword tokenization splits text into pieces for language models. Learn BPE, WordPiece, Unigram, SentencePiece, byte fallback, vocabulary, and token costs.',
    'seoKeywords', jsonb_build_array('what is subword tokenization', 'subword tokenization explained', 'BPE vs WordPiece vs Unigram', 'SentencePiece vs BPE', 'how BPE tokenization works', 'how WordPiece works', 'Unigram tokenizer', 'byte-level BPE', 'subword tokens in LLMs', 'tokenization fertility', 'tokenizer vocabulary size', 'subword tokenization example', 'tokens vs words', 'tokenization and context window')
  ),
  published_at = DATE '2026-09-08',
  updated_at = NOW()
FROM subword_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'subword-tokenization'
  AND item.parent_slug = '';

WITH ann_update AS (
  SELECT
    id,
    replace(
      body,
      E'\n## When exact search is better',
      $section$

## ANN vs CNN

ANN and CNN are not competing versions of the same technique. On this page, **ANN means approximate nearest neighbor**, a search method for retrieving similar vectors. **CNN means convolutional neural network**, a trainable neural-network architecture that applies shared filters to spatial or grid-like data such as images.

| Question | ANN search | CNN |
| --- | --- | --- |
| Full name | Approximate Nearest Neighbor | Convolutional Neural Network |
| Type | Search index or retrieval algorithm | Trainable neural-network architecture |
| Input | A query vector and a collection of stored vectors | Images, audio features, video frames, or other grid-like tensors |
| Output | IDs and distances or similarity scores for nearby items | Predictions, feature maps, or learned embeddings |
| Learning | Some indexes train centroids or codebooks, but ANN does not learn the semantic representation itself | Learns convolutional filters and other parameters from data |
| Typical use | Vector databases, semantic search, recommendation, image similarity, and RAG | Image classification, detection, segmentation, audio analysis, and feature extraction |

ANN search and a CNN can appear in one pipeline. A CNN can convert each image into an embedding vector. An ANN index can then retrieve images whose embeddings are close to a new image. The CNN learns the representation, while ANN accelerates search through the stored representations.

The abbreviation **ANN** can also mean **artificial neural network**. Under that meaning, the comparison changes: a CNN is a specialized type of artificial neural network. An artificial neural network is the broad family of trainable connected-layer models, while a CNN adds convolution, local receptive fields, and weight sharing.

| If ANN means artificial neural network | Artificial neural network | CNN |
| --- | --- | --- |
| Scope | Umbrella category containing many architectures | One architecture within that category |
| Connections | May use dense, recurrent, attention, graph, or other operations | Uses convolutional filters across local regions |
| Parameter sharing | Depends on the architecture | Reuses each filter across positions |
| Spatial inductive bias | Not guaranteed | Designed to exploit local and repeated spatial patterns |

For the neural-network meaning, see [Neural Network](/glossary/neural-network) and [Convolutional Neural Network](/glossary/convolutional-neural-network). For vector retrieval, ANN means the approximate search technique described on this page.

## When exact search is better$section$
    ) AS body
  FROM content_items
  WHERE kind = 'glossary'
    AND slug = 'approximate-nearest-neighbor'
    AND parent_slug = ''
    AND position('## ANN vs CNN' in body) = 0
)
UPDATE content_items AS item
SET
  body = ann_update.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', ann_update.body)
  ),
  sources = item.sources || jsonb_build_array(
    jsonb_build_object('title', 'Gradient-Based Learning Applied to Document Recognition — LeCun et al.', 'url', 'https://doi.org/10.1109/5.726791')
  ),
  metadata = COALESCE(item.metadata, '{}'::jsonb) || jsonb_build_object(
    'seoDescription', 'Approximate nearest neighbor search finds similar vectors at scale. Learn HNSW, IVF, PQ, ANN vs CNN, recall-latency tradeoffs, filtering, and RAG.',
    'seoKeywords', COALESCE(item.metadata->'seoKeywords', '[]'::jsonb) || jsonb_build_array('ANN vs CNN', 'approximate nearest neighbor vs convolutional neural network', 'artificial neural network vs CNN')
  ),
  published_at = DATE '2026-09-08',
  updated_at = NOW()
FROM ann_update
WHERE item.id = ann_update.id;
