-- Publish guide: How to Use Other Models in Codex: A Provider Setup Guide.
WITH guide_body AS (
  SELECT $body$Codex defaults to OpenAI models, but it can also work with models served by Ollama, LM Studio, OpenRouter, and other compatible providers. The important part is understanding the relationship between `model`, which identifies the model you want to use, and `model_provider`, which tells Codex where that model should be reached. These settings normally live in `~/.codex/config.toml`, while separate profile files let you keep several configurations and switch between them without repeatedly editing your main configuration.

## 1. Use a Local Model with Ollama or LM Studio

For local models, Codex provides built-in support for Ollama and LM Studio, so you do not normally need to create a custom `model_providers` entry just to get started. The simplest option is the `--oss` flag, which tells Codex to use a local provider. You can select the provider explicitly and pass the model you already have installed, making this the easiest route when your goal is to run Codex against a model on your own machine rather than sending requests to a hosted API.

```bash id="7m3h1c"
codex --oss
codex --oss --local-provider ollama -m <model>
codex --oss --local-provider lmstudio -m <model>
```

When you use Ollama or LM Studio regularly, you can set a default provider so you do not have to specify it every time:

```toml id="w4f8d2"
oss_provider = "ollama"
```

You can then launch Codex with:

```bash id="p6n2qa"
codex --oss -m <model>
```

This route is particularly convenient for local experimentation because authentication is handled by the local server rather than by an external API key. Just make sure the model is actually available in Ollama or loaded in LM Studio before troubleshooting the Codex configuration.

## 2. Connect Codex to OpenRouter

OpenRouter is useful when you want to use models from multiple providers through a single endpoint and API key. In Codex, OpenRouter is configured as a custom provider: the `model` value contains the exact OpenRouter model slug, `model_provider` points to your provider definition, and the provider block contains the OpenRouter endpoint and authentication settings. Because current Codex custom providers use the Responses API, the provider should be configured with `wire_api = "responses"`.

```toml id="n9t2kx"
model = "<openrouter-model-slug>"
model_provider = "openrouter"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"
env_key = "OPENROUTER_API_KEY"
```

Then provide the API key in your environment:

```bash id="d1v6rp"
export OPENROUTER_API_KEY="your-key"
codex
```

The model name matters here. OpenRouter uses provider-qualified slugs, so you should copy the model identifier directly from OpenRouter instead of guessing or shortening it. Once the provider and model are configured, Codex sends requests through OpenRouter in the same way that it would normally send them through its default provider.

For setups where you want Codex to retrieve provider-specific model information, OpenRouter also documents a command-based authentication approach. That can be useful when working with a large collection of models because Codex can obtain more complete metadata instead of relying entirely on fallback information.

## 3. Connect Another Responses-Compatible Provider

The same configuration pattern works beyond OpenRouter. A hosted inference service, company gateway, self-hosted proxy, or another provider can be added with its own identifier and endpoint. The general idea is always the same: give Codex a model name, assign a provider identifier to it, and define the endpoint and credentials under `model_providers`.

```toml id="r5c8uz"
model = "<model-id>"
model_provider = "my-provider"

[model_providers.my-provider]
name = "My Provider"
base_url = "https://example.com/v1"
wire_api = "responses"
env_key = "MY_PROVIDER_API_KEY"
```

The important compatibility requirement is the protocol. Current Codex expects the Responses API for custom providers. If a service only exposes Chat Completions, changing `wire_api` will not make it compatible; you need a translation layer or gateway that presents a Responses-compatible endpoint to Codex.

This is also where the other provider options become useful. Static HTTP headers can be supplied with `http_headers`, environment-backed headers with `env_http_headers`, and more advanced authentication can use the `auth.command` mechanism instead of a fixed environment variable. These settings let the same basic provider model work for everything from a simple personal gateway to a more involved enterprise setup.

## 4. Configure Azure OpenAI

Azure OpenAI follows the same general provider model, but the endpoint and authentication details are specific to Azure deployments. Instead of pointing Codex at the public OpenAI endpoint, you provide your Azure OpenAI resource URL and the API version expected by that deployment. This makes `query_params` particularly useful because Azure commonly expects the API version to be included in the request URL.

```toml id="j7s4ve"
[model_providers.azure]
name = "Azure"
base_url = "https://YOUR_RESOURCE.openai.azure.com/openai"
env_key = "AZURE_OPENAI_API_KEY"
query_params = { api-version = "YOUR_SUPPORTED_API_VERSION" }
wire_api = "responses"
```

The exact API version should be chosen according to the Azure API surface available to your deployment rather than copied blindly from an old example. The important part for the Codex configuration is the pattern: Azure is treated as a provider, the credentials come from your environment, and the API version is supplied as a query parameter.

## 5. Use Amazon Bedrock

Amazon Bedrock is different from an ordinary custom HTTP provider because Codex includes built-in handling for it. Instead of creating a normal provider block with a `base_url` and API key, you select the Bedrock provider and configure the AWS-specific information such as the model identifier, region, and, when needed, an AWS profile.

```toml id="m2q9xf"
model_provider = "amazon-bedrock"
model = "<bedrock-model-id>"

[model_providers.amazon-bedrock.aws]
profile = "default"
region = "eu-central-1"
```

The `profile` setting is optional. When it is omitted, Codex can use the AWS credentials already available through the normal AWS credential configuration. This makes Bedrock useful for organizations that already have their AWS authentication and deployment structure in place and want Codex to operate through that environment rather than another standalone API key.

## 6. Keep Multiple Configurations with Profiles

Once you use more than one provider, editing `~/.codex/config.toml` every time becomes inconvenient. Profiles solve that by giving each setup its own configuration file. A profile file sits inside your Codex home directory and contains the values that differ from your normal configuration. You can therefore keep a local setup, a cloud setup, and a specialized review setup side by side without constantly rewriting your main config.

For example:

```toml id="e8k3wd"
# ~/.codex/local.config.toml
model = "<local-model>"
model_provider = "ollama"
```

Then run it with:

```bash id="c5r1yn"
codex --profile local
```

A second profile could point at a hosted provider:

```toml id="u6p4mb"
# ~/.codex/cloud.config.toml
model = "<cloud-model>"
model_provider = "openrouter"
```

and be selected with:

```bash id="v3k7qa"
codex --profile cloud
```

Profiles are especially useful when the rest of your Codex configuration—such as MCP servers, approval behavior, or sandbox settings—should remain unchanged while only the model provider changes.

## 7. Put Provider Configuration in the Right Place

For security and machine-local configuration reasons, provider selection belongs in your user configuration or a profile rather than a project's `.codex/config.toml`. That means a repository can have its own project configuration without silently changing which external service your Codex installation sends requests to.

Your normal setup therefore lives in:

```text
~/.codex/config.toml
```

while named alternatives live alongside it:

```text
~/.codex/local.config.toml
~/.codex/cloud.config.toml
~/.codex/review.config.toml
```

This separation is useful because a project can specify its own operational settings while your provider credentials and model routing remain local to your machine.

## 8. The One Rule to Remember

Regardless of which provider you choose, the core configuration is always the same: `model` says **what** Codex should use, `model_provider` says **where** it should send the request, and the provider block defines **how** to reach that service. Once that mental model is clear, Ollama, OpenRouter, Azure, Bedrock, and other compatible endpoints are variations on the same configuration pattern rather than completely different systems.

> **Caveats:** Current Codex versions require the Responses API for custom providers; older configurations using `wire_api = "chat"` no longer work. Provider settings such as `model_provider` and `model_providers` are not accepted from project-level `.codex/config.toml`. Custom-provider behavior can also vary between Codex releases and authentication modes, and current issues have been reported around the `/model` picker, changing providers inside an existing session, custom model catalogs, and some advanced features such as MCP. Provider compatibility should therefore be checked against both the current Codex documentation and the provider's own API documentation before relying on a specific configuration.$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'guide:how-to-use-other-models-in-codex',
  'guide',
  'how-to-use-other-models-in-codex',
  '',
  'guides/how-to-use-other-models-in-codex',
  'How to Use Other Models in Codex: A Provider Setup Guide',
  'Codex defaults to OpenAI models, but it also runs Ollama, LM Studio, OpenRouter, Azure, and Bedrock. How model, model_provider, and profiles route requests.',
  guide_body.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', guide_body.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Config basics – Codex | OpenAI Developers', 'url', 'https://developers.openai.com/codex/local-config'),
    jsonb_build_object('title', 'Advanced Configuration – Codex | OpenAI Developers', 'url', 'https://developers.openai.com/codex/config-advanced'),
    jsonb_build_object('title', 'Config basics | ChatGPT Learn', 'url', 'https://developers.openai.com/codex/config-basic'),
    jsonb_build_object('title', 'Integration with Codex CLI | OpenRouter', 'url', 'https://openrouter.ai/docs/guides/coding-agents/codex-cli'),
    jsonb_build_object('title', 'Use Codex with Amazon Bedrock | OpenAI Developers', 'url', 'https://developers.openai.com/codex/amazon-bedrock'),
    jsonb_build_object('title', 'openai/codex · GitHub', 'url', 'https://github.com/openai/codex')
  ),
  '{}'::jsonb,
  NULL,
  NULL,
  'published',
  DATE '2026-09-16',
  0
FROM guide_body
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path,
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  body = EXCLUDED.body,
  blocks = EXCLUDED.blocks,
  sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata,
  cover_image_url = EXCLUDED.cover_image_url,
  cover_image_alt = EXCLUDED.cover_image_alt,
  status = EXCLUDED.status,
  published_at = EXCLUDED.published_at,
  sort_order = EXCLUDED.sort_order,
  updated_at = NOW();
