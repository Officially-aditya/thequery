-- Answer observed low-memory setup questions and correct serving advice in one living guide.
-- Preserve title, summary, slug, metadata, status, and original publication timestamp.
WITH revised AS (
  SELECT $qwen_body$Qwen3.8-27B is worth the setup time. It's a 27.3-billion-parameter dense model, Apache 2.0 licensed, natively vision-capable, and small enough to run on a single consumer GPU while landing close to what a much larger, once-frontier model needed six months ago. This guide covers every practical route to actually running it: Ollama for the fastest start, llama.cpp for full control, LM Studio if you want a GUI, vLLM if you're serving it to more than yourself, and Apple Silicon specifically, since unified memory changes the math.

## What You're Actually Installing

Confirmed directly from the model itself once pulled: 27.3B parameters, architecture family qwen35 (the same underlying architecture as Qwen3.6-27B, which is why tooling that already supports 3.6 tends to pick up 3.8 with little friction), 262,144 tokens of native context, extensible to 1M with RoPE scaling. It ships with a vision projector, 460.73M parameters, clip architecture, so image input works out of the box, not as a bolted-on add-on. Capabilities reported by the runtime itself: completion, vision, tools, and thinking. It also ships with multi-token prediction enabled by default through Ollama's library build, which is part of why throughput on this model tends to run faster than its parameter count alone would suggest.

The official weights live at `Qwen/Qwen3.8-27B` on Hugging Face, Apache 2.0. Check the publisher name before downloading anything, placeholder and fork repos squatted on this model's name for weeks before release, and some are still indexed.

## Picking Your Route

| Route | Best for | Setup time |
|---|---|---|
| Ollama | Fastest start, sane defaults, minimal config | Minutes |
| llama.cpp | Full control over quantization, sampling, and server flags | 15-30 minutes |
| LM Studio | GUI, no terminal required | Minutes |
| vLLM | Serving to a team or an app, real concurrency | 30-60 minutes |

If you're not sure, start with Ollama. You can always move to llama.cpp later once you know which quantization and context settings you actually need.

## Hardware and Quantization

The memory math for weights alone, before KV cache:

| Precision | VRAM for weights | What runs it |
|---|---|---|
| BF16 | ~54GB | H100, H200, RTX Pro 6000 (96GB) |
| FP8 | ~27GB | L40S, RTX Pro 6000, RTX 5090 (short context) |
| Q8_0 GGUF | ~29GB | RTX 5090, RTX Pro 6000, 32GB+ cards |
| Q4_K_M GGUF | ~16-17GB | RTX 4090, RTX 5090, 24GB cards |

KV cache comes on top and scales with context length and concurrency. Leave room beyond the weights for context, runtime buffers, and vision input when used. First establish whether the weights themselves do not fit or whether the serving context and concurrent requests exceed the remaining memory. A smaller context can address the second problem without changing weight precision; it cannot make an oversized set of weights fit entirely in VRAM.

## Route 1: Ollama

Install Ollama if you don't already have it:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then pull and run the model. This is the actual, verified command, not a guess at a future naming pattern:

```bash
ollama pull qwen3.8
ollama run qwen3.8
```

This requires Ollama 0.32.12 or later. If `ollama pull` fails with a manifest error, that's almost always an out-of-date Ollama install, run `ollama --version` and update first.

To confirm exactly what you pulled and what it's capable of, run:

```bash
ollama show qwen3.8
```

You should see output close to this:

```
Model
  architecture        qwen35
  parameters          27.3B
  context length       262144
  embedding length     5120
  quantization         Q4_K_M

Capabilities
  completion
  vision
  tools
  thinking

Projector
  architecture         clip
  parameters           460.73M
  embedding length     1152
```

The default pull is Q4_K_M. Ollama serves an OpenAI-compatible endpoint on `localhost:11434` immediately, so any existing client code built against OpenAI's API points at it with a one-line base URL change:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by the client, ignored by Ollama
)

response = client.chat.completions.create(
    model="qwen3.8",
    messages=[{"role": "user", "content": "Write a haiku about local inference."}],
)
print(response.choices[0].message.content)
```

For agentic or tool-calling work, thinking is on by default. If you need low-latency tool calls without the reasoning trace, pass `think: false` in the request:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.8",
  "messages": [{"role": "user", "content": "What is 14 * 37?"}],
  "think": false
}'
```

## Route 2: llama.cpp

This is the route for full control, custom sampling parameters, and running the exact quantization you want rather than whatever a library default picked for you.

Build llama.cpp if you haven't already:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # drop -DGGML_CUDA=ON if you're CPU-only or on Apple Silicon
cmake --build build --config Release -j
```

Community GGUF quantizations are already available, including a set built and maintained by the LM Studio community team directly from Qwen's official release:

```bash
./build/bin/llama-server \
  -hf lmstudio-community/Qwen3.8-27B-GGUF:Q4_K_M \
  --jinja \
  -ngl 99 \
  -fa auto \
  -c 32768 \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0 \
  --port 8080
```

The `-hf` flag pulls directly from Hugging Face, no separate download step needed. `-ngl 99` offloads all layers to GPU, `-fa auto` lets the runtime select flash attention, and `-c 32768` sets context length, raise this if your VRAM headroom allows and you actually need the longer context. Swap `Q4_K_M` for `Q8_0` if you have the VRAM and want higher fidelity.

Once running, the server exposes an OpenAI-compatible endpoint at `http://localhost:8080/v1/`, and a web UI at `http://localhost:8080/`.

For a one-off prompt without starting a persistent server:

```bash
./build/bin/llama-cli \
  -hf lmstudio-community/Qwen3.8-27B-GGUF:Q4_K_M \
  -ngl 99 \
  -fa auto \
  --temp 0.6 \
  -p "Explain the difference between BF16 and Q4_K_M in two sentences."
```

One real gotcha worth knowing before you debug it yourself: on some llama.cpp builds, the default reasoning-format parser has mishandled `<think>` tags from Qwen models, truncating or garbling the output some clients receive. If your responses look cut off or empty where reasoning should be, add `--reasoning-format none` to the server command and parse the `<think>` block yourself:

```python
import re

def split_thinking(raw_text: str):
    """Split Qwen output into (thinking, answer) when reasoning-format is set to none."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip(), raw_text[match.end():].strip()
    return "", raw_text.strip()
```

## Route 3: LM Studio

If you'd rather not live in a terminal, LM Studio gives you a GUI over the same llama.cpp engine underneath.

1. Install LM Studio from lmstudio.ai.
2. Open the search tab and search for `Qwen3.8-27B`.
3. Select the `lmstudio-community` build, and pick a quantization based on the VRAM table above.
4. Download, then load the model from the chat tab.

LM Studio also exposes a local OpenAI-compatible server if you go to the Developer tab and start the server, letting you point external tools at it exactly like the Ollama or llama.cpp endpoints above.

## Route 4: vLLM (Real Serving)

For production or high-throughput workloads, [Qwen's model card](https://huggingface.co/Qwen/Qwen3.8-27B) recommends dedicated serving engines such as vLLM. That recommendation is not a claim that local runtimes cannot handle several clients: [Ollama supports concurrent requests](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests), and [llama-server documents parallel decoding and continuous batching](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). Choose an engine using your workload and measured latency, throughput, and memory requirements.

```bash
pip install vllm

vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

This serves an OpenAI-compatible endpoint on port 8000 by default. Raise `--tensor-parallel-size` if you're spreading the model across multiple GPUs, and adjust `--max-model-len` to your context needs and available memory. Test the concurrency and context settings your application actually uses rather than assuming that a successful single-request test establishes serving capacity.

## Running on Apple Silicon

On Apple Silicon, llama.cpp can use Metal. Ollama also provides an MLX variant documented in its [Qwen3.8 support release](https://github.com/ollama/ollama/releases/tag/v0.32.12):

```bash
ollama run qwen3.8:27b-mlx
```

For the llama.cpp route, this example starts with a smaller context for a memory diagnostic. Increase it when the task needs more context and your memory budget allows it.

```bash
brew install llama.cpp
llama-server \
  -hf lmstudio-community/Qwen3.8-27B-GGUF:Q4_K_M \
  -ngl 99 \
  -fa auto \
  -c 4096 \
  --port 8080
```

Unified memory is shared with macOS and other applications. A 24 GB Mac does not automatically have enough free memory for every context size or vision workload, even if the quantized weights fit. Check memory pressure and the runtime's reported allocation, then measure prompt processing and generation on your own configuration. This guide does not establish a tokens-per-second result for a particular Mac.

## Can Qwen3.8-27B Run on a 4 GB GPU?

Not fully in that GPU's VRAM with the Q4_K_M build discussed here. [Qwen identifies this as a dense 27B model](https://huggingface.co/Qwen/Qwen3.8-27B), and [LM Studio's GGUF repository](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF) lists its Q4_K_M file at 16.8 GB. That file size is not a complete runtime memory budget: context, buffers, and the vision projector also need room.

For scale, an idealized 27-billion-weight model at exactly four bits per weight would require `27 billion × 4 ÷ 8 = 13.5 billion bytes` before quantization metadata and other allocations. This arithmetic is an illustration, not a measured memory requirement. Even that simplified figure is larger than 4 GB.

CPU inference or a CPU/GPU split can use system RAM alongside the GPU, if the machine has sufficient memory and a supported runtime. [Ollama documents how to identify that split](https://docs.ollama.com/faq#how-can-i-tell-if-my-model-was-loaded-onto-the-gpu). Offloading changes where weights live; it does not make the whole model fit in 4 GB or establish how fast it will run on your machine.

Running four agents is not a way to select a smaller part of this dense model. Several clients may share one loaded model server, but their request contexts still need memory. Test concurrency after establishing that one request fits.

## How to Check GPU Offloading and Context Memory

While the model is loaded, run:

```bash
ollama ps
```

Check `PROCESSOR` for full GPU placement, CPU placement, or a split. Check the allocated context in `CONTEXT` when your version displays it. [Ollama's context documentation](https://docs.ollama.com/context-length) distinguishes the serving allocation from a model's maximum supported context; increasing it needs more memory. The model card's 262,144-token capacity is not a promise that your hardware can run that allocation.

For a small interactive test in `ollama run qwen3.8`, set the context explicitly at the prompt:

```text
/set parameter num_ctx 4096
```

This is a memory diagnostic starting point, not a recommended context for every coding or agent workload. Increase it only when your task needs the space and your memory budget allows it. The interactive setting is documented in the [Ollama FAQ](https://docs.ollama.com/faq#how-can-i-specify-the-context-window-size).

| Symptom | What to check first |
|---|---|
| Model fails to load | Whether weights plus startup allocations fit; shorten context where relevant, or choose a smaller model/quantization or a supported offload configuration. |
| Short prompts work, longer ones fail | Context allocation and other running applications before changing weight precision. |
| Model runs but GPU use is unexpectedly low | `ollama ps` to establish whether it is using CPU or mixed placement. |
| One request works, several agents exhaust memory | Parallelism and per-request context. Ollama's [concurrency documentation](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests) explains that parallel contexts require additional memory. |

For Linux, follow the Ollama installation route or build llama.cpp with the backend for your actual GPU. The CUDA build flag in this guide is for NVIDIA CUDA; omitting it selects a different build path and does not verify support for a particular AMD, Intel, or older NVIDIA card. Check the runtime's supported hardware before assuming GPU offload.

## Troubleshooting

**Out of memory on load.** Check weights, context allocation, and other loaded models or applications. If the weights fit but the remaining allocations do not, try a smaller context or lower concurrency first. If the weights themselves exceed the budget, choose a smaller model/quantization or a supported CPU/GPU split with enough system RAM.

**Ollama pull fails with a manifest error.** Update Ollama first, this model requires 0.32.12 or later. Older installs will fail to resolve the manifest entirely rather than downloading a broken copy.

**Thinking traces leaking into or vanishing from tool-call output.** This is a known category of issue with reasoning-tag parsing across Qwen models on llama.cpp-based servers. Set `--reasoning-format none` on the server side and parse `<think>` tags yourself, or pass `think: false` in the request if you're using Ollama and don't need the reasoning trace at all.

**Vision input not working.** Confirm your client is actually sending the projector alongside the text tower, some quantization pipelines ship the vision component as a separate `mmproj` file that has to be loaded explicitly rather than bundled automatically.

## Go Deeper

- [Qwen 3.8 27b Is a Revolution in Open Weight Models](/articles/qwen-3-8-27b-opus-4-6-vision-benchmarks) - the full benchmark breakdown behind why this model is worth installing in the first place
- [Every Benchmark Claim Looks Convincing. Here's How to Actually Check One.](/guides/how-to-check-ai-benchmark-claims) - the framework for reading any vendor's numbers, including this one's

**Update (October 13, 2026):** Added the 4 GB GPU and memory diagnostics and explicit flash-attention settings in the llama.cpp examples. The earlier guide described Ollama and llama-server as single-user tools; their current documentation supports concurrent requests, and llama-server documents continuous batching. This update also replaces the blanket Apple Silicon recommendation, the suggestion that 24 GB is comfortably sufficient, and the unsourced general expectation of around 80 tokens per second with configuration-specific memory checks. No new inference benchmark is claimed.$qwen_body$::text AS body,
    $qwen_sources$[{"title":"Qwen3.8-27B official model card","url":"https://huggingface.co/Qwen/Qwen3.8-27B"},{"title":"LM Studio Qwen3.8 GGUF files and sizes","url":"https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF"},{"title":"Ollama FAQ: memory placement, context and concurrency","url":"https://docs.ollama.com/faq"},{"title":"Ollama context-length documentation","url":"https://docs.ollama.com/context-length"},{"title":"Ollama v0.32.12: Qwen3.8 and Apple Silicon support","url":"https://github.com/ollama/ollama/releases/tag/v0.32.12"},{"title":"llama-server documentation: GPU layers, parallel decoding and batching","url":"https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md"}]$qwen_sources$::jsonb AS sources
)
UPDATE content_items AS item
SET body = revised.body,
    blocks = jsonb_set(item.blocks, '{0,content}', to_jsonb(revised.body)),
    sources = item.sources || COALESCE((
      SELECT jsonb_agg(source) FROM jsonb_array_elements(revised.sources) AS source
      WHERE NOT EXISTS (
        SELECT 1 FROM jsonb_array_elements(item.sources) AS existing
        WHERE existing->>'url' = source->>'url'
      )
    ), '[]'::jsonb),
    updated_at = NOW()
FROM revised
WHERE item.kind = 'guide' AND item.parent_slug = ''
  AND item.slug = 'qwen-3-8-27b-install-run'
  AND md5(item.body) IN ('20985e941a8b2d17746c286130a907c3', md5(revised.body))
  AND jsonb_array_length(item.blocks) = 1
  AND item.blocks->0->>'type' = 'markdown'
  AND (
    item.body IS DISTINCT FROM revised.body
    OR item.blocks->0->>'content' IS DISTINCT FROM revised.body
    OR EXISTS (
      SELECT 1 FROM jsonb_array_elements(revised.sources) AS source
      WHERE NOT EXISTS (
        SELECT 1 FROM jsonb_array_elements(item.sources) AS existing
        WHERE existing->>'url' = source->>'url'
      )
    )
  )
RETURNING item.slug;
