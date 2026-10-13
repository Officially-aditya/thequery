-- Improve three living glossary entries with worked examples and primary references.
-- Keep URLs, titles, publication dates, and unrelated metadata intact.
-- Only replace the reviewed prior body (or the same expanded body on a rerun).

WITH expanded(slug, previous_body_md5, summary, body, sources) AS (
  VALUES
  ('ai-agent', '6f3cf3bff284400ea6fea9008ae1ff30', $summary_0$An LLM-based system that chooses actions, uses tools, observes results, and decides its next steps toward a specified goal.$summary_0$, $body_0$An AI agent, in the LLM sense, is a system that uses a [language model](/glossary/large-language-model) to choose actions, call tools, observe results, and decide what to do next. The application supplies the tools and controls their permissions.

## AI agent vs chatbot vs workflow

A chatbot is an interface. It can sit in front of a single model call, a fixed workflow, or an agent. A chat window alone tells you little about the architecture behind it.

In [Anthropic's architectural distinction](https://www.anthropic.com/engineering/building-effective-agents), a workflow follows predefined code paths, while an agent lets the model direct the process and tool use. A fixed sequence of search, summarize, and format can be a workflow. A system that chooses another search after inspecting an incomplete result is making an agent-style decision.

## How the agent loop works

The model receives a task, instructions, available tools, and relevant context. It can request a tool call or produce an answer. The application executes an allowed call, returns its result, and gives the model another opportunity to respond. [Google's ADK documentation](https://adk.dev/agents/llm-agents/) describes model-driven choices over tools and outputs, guided by explicit instructions.

The surrounding [agent harness](/glossary/agent-harness) manages execution. A tool request from the model does not itself grant permission to send a message, change a database, or spend money.

## Example: researching a product's privacy policy

Consider this illustrative task: "Does this product say it uses customer documents to train its models? Find the relevant policy and flag anything unclear." This is an example of an agent design, not a report of a deployed system.

1. The agent searches for the vendor's official policy and opens the result.
2. It checks whether the wording covers the product in question. If the page only describes a different plan, it searches for the applicable terms.
3. It records the source URL and the passage supporting each finding.
4. It reports what the policy states, what it does not resolve, and where the reader can verify the answer.

For this task, the tools could allow public search and page reading while disabling account changes and outbound messages. The success check is an answer supported by the applicable policy, with uncertainty preserved. Finding a vaguely related page would not satisfy that check.

## When an agent is useful

Agents suit tasks whose next step depends on newly observed information. A predictable process is often easier to implement as a workflow. [Anthropic recommends](https://www.anthropic.com/engineering/building-effective-agents) starting with simpler approaches and evaluating whether extra autonomy justifies its cost and latency.

For the policy example, compare an agent with a single retrieval-and-answer call. Check whether it finds the applicable document and supports its conclusions. Extra searches are only useful if they improve the result.

## What can go wrong?

Mistakes can compound when later actions depend on earlier results. [Anthropic's guidance](https://www.anthropic.com/engineering/building-effective-agents) calls for testing, guardrails, and stopping conditions. Set a tool-call or time budget and provide a route to human review.

For the example above, record opened URLs and tool failures. Test missing policies, ambiguous plan names, and conflicting terms. An incomplete answer that identifies missing evidence is preferable to a confident unsupported conclusion.

For implementation choices, continue with [How to Build an AI Agent: Start Here](/guides/you-want-to-build-an-ai-agent-here-is-where-to-actually-start).

**Editorial note (October 13, 2026):** An earlier version included an unsourced example claiming thousands of incorrect refund transactions. That example has been removed.$body_0$, $sources_0$[{"title":"Building effective agents (Anthropic)","url":"https://www.anthropic.com/engineering/building-effective-agents"},{"title":"Simple agents with LlmAgent (Google ADK)","url":"https://adk.dev/agents/llm-agents/"}]$sources_0$::jsonb),
  ('retrieval-augmented-generation', '8dd48a9eaaf00399832c92c45ea9dce2', $summary_1$An architecture pattern that retrieves information from external sources and supplies it as context to a language model to help ground its responses.$summary_1$, $body_1$Retrieval-Augmented Generation (RAG) supplies a [language model](/glossary/large-language-model) with information retrieved from an external source before it writes an answer. The source might be documentation, a search index, or a database. The retrieved material becomes context for that request; it does not automatically change the model's weights.

## How does RAG work?

A simple document-based pipeline has two stages: preparing searchable information and retrieving evidence for a question.

1. **Prepare the documents.** Split long material into useful [chunks](/glossary/chunking), retaining source identifiers and relevant metadata.
2. **Search for evidence.** Use the question to retrieve candidate passages.
3. **Choose the context.** Select passages that help answer the question and fit the available context budget.
4. **Generate the answer.** Give the model the question and selected passages, with instructions for using and citing the evidence.

[Microsoft's RAG documentation](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview) covers content preparation, relevance, context limits, and access controls. An implementation must retrieve information the user is permitted to see.

## Does RAG require a vector database?

No. Retrieval can use keyword search, vector search, or [hybrid search](/glossary/hybrid-search). Vector search uses [embeddings](/glossary/embedding) to compare meaning; keyword search can help match specific terms. [Reranking](/glossary/reranking) can reorder candidates before they reach the model. [Google Cloud describes](https://cloud.google.com/use-cases/retrieval-augmented-generation) retrieval systems that combine keyword and semantic search with reranking. A vector database is one implementation choice, not the definition of RAG.

## Worked example: answering a fictional API question

Suppose a fictional service, Cedar API, has these documents:

| Source | Passage |
|---|---|
| Current authentication guide | Requests require an `Authorization: Bearer <token>` header. |
| Current rate-limit guide | A rate-limited request returns HTTP 429 and a `Retry-After` header. |
| Archived authentication guide | An older API version accepted an `X-API-Key` header. |

A reader asks: "Which header does the current API require, and what should my client do after a 429?"

The retriever should return both current guides. With those passages, the answer can identify the Bearer header and recommend following the retry delay, citing each guide. The archived guide should not override current documentation.

If retrieval returns only the authentication guide, the evidence supports the first part of the question but not the retry behavior. The system should retrieve more information or say that the supplied material does not resolve the second part. This example illustrates evidence coverage; it is not a claim about a real API or a measured accuracy improvement.

## Does RAG prevent hallucinations?

It can help ground an answer, but it does not guarantee correctness. [Google Cloud's guidance](https://cloud.google.com/use-cases/retrieval-augmented-generation) explains that irrelevant retrieved material can lead to off-topic or incorrect answers. A citation also needs to support the actual claim.

For the Cedar example, check retrieval and generation separately: did the system find both current guides, and did the final answer accurately use them? Test an unanswerable question, an outdated document, and a question needing two sources. These cases reveal failures that a single plausible answer can hide.

## RAG vs fine-tuning

RAG changes the external evidence supplied for a request. [Fine-tuning](/glossary/fine-tuning) changes model parameters through additional training. Decide whether the problem is missing source information or model behavior before choosing an approach. They can be combined, but neither makes incorrect source material reliable.

For deeper explanations, read the [RAG theoretical foundations chapter](/books/rag-kg-master-course/theoretical-foundations). For operational tradeoffs, continue with [why RAG fails in production](/guides/rag-fails-in-production).$body_1$, $sources_1$[{"title":"Retrieval-augmented generation in Azure AI Search (Microsoft)","url":"https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview"},{"title":"What is Retrieval-Augmented Generation? (Google Cloud)","url":"https://cloud.google.com/use-cases/retrieval-augmented-generation"}]$sources_1$::jsonb),
  ('backpropagation', '009198204fef2291c9486e35032b0f19', $summary_2$An algorithm for computing gradients of the loss function with respect to each weight in a neural network by applying the chain rule layer by layer.$summary_2$, $body_2$Backpropagation computes the gradient of a [loss function](/glossary/loss-function) with respect to a neural network's parameters. It applies the chain rule backward through the computation so an optimizer can decide how to change the weights.

## Forward pass, backward pass, and weight update

The forward pass produces a prediction. The loss measures its error against a target. The backward pass computes derivatives of that loss with respect to the parameters. An optimizer then uses those derivatives to update the parameters.

This separates backpropagation from [gradient descent](/glossary/gradient-descent): backpropagation calculates gradients; gradient descent uses them. In [PyTorch's autograd tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html), `loss.backward()` calculates gradients and `optimizer.step()` performs the update.

## A numerical example with two weights

Consider a tiny two-layer linear network without biases or nonlinear activations:

$$h = ax, \qquad \hat{y} = bh, \qquad L = \frac{1}{2}(\hat{y}-y)^2$$

Use input $x=2$, target $y=1$, and initial weights $a=0.5$ and $b=2$. The forward pass gives $h=1$, prediction $\hat{y}=2$, and loss $L=0.5$.

First compute the derivative at the output:

$$\frac{\partial L}{\partial \hat{y}} = \hat{y}-y = 1$$

Then apply the chain rule to each weight:

$$\frac{\partial L}{\partial b} = (\hat{y}-y)h = 1$$

$$\frac{\partial L}{\partial a} = (\hat{y}-y)bx = 4$$

The gradient for $a$ passes through the output weight $b$. That is the backward dependency the example is meant to show.

With a [learning rate](/glossary/learning-rate) of $0.05$, a gradient-descent update gives $a=0.3$ and $b=1.95$. The new prediction is $1.17$ and the new loss is $0.01445$. This one-step improvement is specific to these chosen values, not a guarantee that every training update reduces loss.

## Run the example in plain Python

This standalone calculation requires no deep-learning library:

```python
x, target = 2.0, 1.0
a, b = 0.5, 2.0
learning_rate = 0.05

hidden = a * x
prediction = b * hidden
error = prediction - target
gradient_a = error * b * x
gradient_b = error * hidden

# Compute both gradients before changing either weight.
a -= learning_rate * gradient_a
b -= learning_rate * gradient_b

new_prediction = b * a * x
new_loss = 0.5 * (new_prediction - target) ** 2
print(round(new_prediction, 2), round(new_loss, 5))
# 1.17 0.01445
```

Changing $b$ before calculating the gradient for $a$ would mix values from different forward passes. Both gradients here come from the same original prediction.

## What does PyTorch automate?

PyTorch records differentiable operations and applies the chain rule when `backward()` runs. Gradients are stored in parameter `.grad` attributes. Its [gradient-reset recipe](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html) explains that backward calls accumulate gradients, so an ordinary training loop clears them between updates unless accumulation is intentional.

If loss does not change, check that gradient computation is enabled, the optimizer contains the intended parameters, and an update actually runs. A computed gradient alone does not change a weight.

Continue with the [neural-networks chapter](/books/ai-from-first-principles/neural-networks) for nonlinear networks and the [math foundations chapter](/books/ai-from-first-principles/math-you-cant-escape) for the calculus behind the chain rule.$body_2$, $sources_2$[{"title":"A Gentle Introduction to torch.autograd (PyTorch)","url":"https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html"},{"title":"Zeroing out gradients in PyTorch","url":"https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html"}]$sources_2$::jsonb)
)
UPDATE content_items AS item
SET summary = expanded.summary,
    body = expanded.body,
    blocks = jsonb_set(item.blocks, '{0,content}', to_jsonb(expanded.body)),
    sources = item.sources || COALESCE((
      SELECT jsonb_agg(source)
      FROM jsonb_array_elements(expanded.sources) AS source
      WHERE NOT EXISTS (
        SELECT 1 FROM jsonb_array_elements(item.sources) AS existing
        WHERE existing->>'url' = source->>'url'
      )
    ), '[]'::jsonb),
    updated_at = NOW()
FROM expanded
WHERE item.kind = 'glossary'
  AND item.parent_slug = ''
  AND item.slug = expanded.slug
  AND md5(item.body) IN (expanded.previous_body_md5, md5(expanded.body))
  AND jsonb_array_length(item.blocks) = 1
  AND item.blocks->0->>'type' = 'markdown'
  AND (
    item.body IS DISTINCT FROM expanded.body
    OR item.summary IS DISTINCT FROM expanded.summary
    OR item.blocks->0->>'content' IS DISTINCT FROM expanded.body
    OR EXISTS (
      SELECT 1 FROM jsonb_array_elements(expanded.sources) AS source
      WHERE NOT EXISTS (
        SELECT 1 FROM jsonb_array_elements(item.sources) AS existing
        WHERE existing->>'url' = source->>'url'
      )
    )
  )
RETURNING item.slug;
