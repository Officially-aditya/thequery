-- Expand the decisions API category with the providers covered by TheQuery's article.
-- Include current API contracts, access status and local serving distinctions.
UPDATE content_items AS content
SET summary = entry.summary, body = entry.body,
    blocks = jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', entry.body)),
    sources = entry.sources, metadata = entry.metadata,
    updated_at = TIMESTAMPTZ '2026-10-12 00:00:00+00'
FROM jsonb_to_record($entry$
{
  "id": "glossary:decisions-api",
  "slug": "decisions-api",
  "title": "Decisions API",
  "summary": "An API category that turns evidence into bounded judgments, such as labels, predicate probabilities and rubric scores, for classification, routing and prioritization.",
  "body": "A **decisions API**, also called a **decision API**, is an [API](/glossary/api) category for turning evidence into bounded judgments that software can use directly. In an AI workflow, the caller supplies context, a question and a defined answer space. The service returns a label, a true-or-false assessment, or a rubric score, often accompanied by probabilities or confidence information. The application uses that result to classify, route, filter or prioritize work.\n\nThe category spans hosted APIs, locally served open models and database integrations. Companies with decision offerings include TypeSafe, OpenAI, Microsoft, Perplexity, Cloudflare, Amazon, Snowflake and Upstage. Their products share bounded judgments as a task, while differing in request formats, supported inputs and deployment options. [Pydantic AI](https://pydantic.dev/docs/ai/models/decision/) documents several providers under a shared decision-model abstraction, with provider-specific configuration and limits.\n\nThink of a switchboard operator: the destinations already exist, and the task is to choose where an incoming request belongs. A decisions API makes that choice available to application code. The category describes this interface pattern; it does not establish one universal endpoint, request schema, model architecture or pricing policy.\n\n## How a decisions API works\n\nA request separates the **evidence** from the **judgment to make**. The evidence might be a support ticket, a document passage, the state of an agent workflow, or an image when the implementation supports images. The question states what to assess, and the options or rubric describe the permitted outcomes.\n\nFor example, an application can submit a customer message with the question, \"Which team should handle this?\" It supplies billing, technical support and other as the available categories, with descriptions that distinguish them. The service returns a selected category and whatever uncertainty information its contract exposes. Application code then places the ticket in that queue.\n\nSeveral independent questions can assess the same evidence, such as the responsible team and whether a customer explicitly requested cancellation. Some implementations answer them together. When a later question depends on an earlier answer, the application must establish the sequence rather than assume that a batch of questions creates it.\n\n## Common output forms\n\nTyped decision services commonly expose three patterns. Their exact names and returned fields differ between implementations.\n\n| Decision pattern | Meaning | Example | Implementation detail to check |\n| --- | --- | --- | --- |\n| Predicate or yes/no assessment | Whether a stated condition holds, or its estimated probability | Does the message request cancellation? | OpenAI calls this `predicate`; SystemOne uses `noul`. An API or client may return a probability or convert it to a Boolean. |\n| Fixed-option choice | Select one of the supplied alternatives | Billing, technical support or other | Check option limits, descriptions, probability fields and whether an explicit fallback option is supported. |\n| Ordered rubric score | Assess evidence against described levels | Routine, urgent or critical | A score may fall between level indices. Verify the scale and whether the client rounds it. |\n\nA rubric works like a teacher's marking guide: each level describes an observable situation. \"The customer cannot use a paid feature\" gives a model more specific evidence to assess than \"very urgent.\" Categories need equally clear boundaries, particularly when an input could reasonably fit more than one.\n\nA weighted rubric score and a discrete label are different outputs. In OpenAI's implementation, probabilities of 0.1, 0.6 and 0.3 across levels 0, 1 and 2 yield a score of 1.2. That calculation summarizes the distribution. An application requiring one category should use a choice or define an explicit conversion policy.\n\n## The API, the model and the application\n\nThe API is the contract between software components. The **decision model** is the system producing the judgment behind that contract. A local serving layer can expose different models through the same interface, while the same model can be exposed through more than one request format. [LocalAI](https://localai.io/docs/features/decisions/), for example, documents both SystemOne and OpenAI-compatible decisions endpoints sharing the same execution layer.\n\nAn API name alone does not reveal how a model computes an answer. Dedicated decision models can score bounded outcomes without generating a written reply. General language models can also perform classification through constrained structured output. Compare the actual implementation and workload before assuming that every decisions API uses the same architecture or inference process.\n\nThe application defines what happens after the judgment. Choosing \"billing\" does not execute a refund. Choosing an [AI agent](/glossary/ai-agent)'s next step does not supply every argument or permission needed to carry it out. The decision result is one input to the surrounding workflow, alongside account state and explicit application rules.\n\n## Related interfaces compared\n\n| Interface | Core task | Typical output | Appropriate use |\n| --- | --- | --- | --- |\n| Decisions API | Judge evidence against bounded outcomes | Label, probability or rubric score | Triage, relevance filtering and [query routing](/glossary/query-routing) |\n| General model with structured output | Generate content within a schema | Custom object with extracted or generated fields | Document extraction, structured reports and tasks needing free-form fields |\n| [Tool Calling](/glossary/tool-calling) | Request an application operation | Function name and arguments | Select and parameterize a tool for the application to execute |\n| Deterministic rules service | Apply explicit conditions | Rule result or prescribed branch | Enforce known limits and business rules |\n\nStructured output resembles a form: it constrains the shape of the answer. Tool calling resembles a work order: it names an operation and supplies its parameters. These interfaces overlap with decision workflows. A schema containing a fixed enum can represent a classification, and a decision client can turn selected routes into tool calls. The useful distinction is the work required and the contract exposed to the application, rather than an assumption that only one interface can classify text.\n\n## Companies and implementations\n\nDecision interfaces are available from several companies, with different access methods and deployment options. The following comparison reflects documentation reviewed on October 12, 2026.\n\n| Company | Offering | Access and availability | Distinguishing details |\n| --- | --- | --- | --- |\n| [TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Jev | Hosted TypeSafe API | Typed judgments over text, with probabilities; the SystemOne interface is also used by other decision-model servers. |\n| [OpenAI](https://developers.openai.com/api/docs/guides/decisions) | Decisions API | Hosted `POST /v1/decisions`, public beta | [GPT-6 Luna](/glossary/gpt-6-luna); text and image inputs; `predicate`, `choice` and `score` questions. |\n| [Microsoft](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/) | Microsoft-Decision-1 | Microsoft Foundry and OpenRouter | A decision model post-trained from Qwen3.5-9B, supporting yes/no, multiple-choice and rating judgments for routing, classification and workflow control. |\n| [Perplexity](https://docs.perplexity.ai/docs/decisions/quickstart) | Decisions API, using `pplx-decider-v1.1-27b` | Hosted `POST /v1/decisions` and [published model weights](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | Text, JSON and image state; `noul`, `choice` and `score`. Its request shape differs from OpenAI's despite sharing an endpoint path. |\n| [Cloudflare](https://blog.cloudflare.com/clef-faster-cheaper-multimodal/) | Clef, Clef-flash and Clef-omni | Workers AI and open weights | Clef-omni extends the family to text, image, audio and video decisions. Model selection determines the supported modalities. |\n| [Amazon / AWS Strands](https://github.com/strands-labs/strands-decider) | Strands Decider 2B | Open model and tooling; local `/v1/systemone` serving | A Qwen3.5-2B-based model for local CPU/GPU use. Current tooling supports choice, `noul` and rubric-score questions. |\n| [Snowflake](https://docs.snowflake.com/en/release-notes/2026/other/2026-10-07-ai-complete-snowflake-decision-preview) | Snowflake Decision | `AI_COMPLETE` in Cortex AI Functions, private preview | Classification, rubric scoring and true-or-false assessment from SQL, including decisions over table rows. |\n| [Upstage](https://console.upstage.ai/api/systemone) | Solar Decide | Hosted `POST /v1/systemone`, beta | Served on Solar Mini 4; choice, score and `noul` questions over text or serialized JSON state, with a documented 512K context window. |\n\nThese offerings expose the same broad pattern through different products. Microsoft's offering is a model accessed through hosted platforms. Amazon supplies a model and serving tools for local deployment. Snowflake embeds decisions in a SQL function. Treating all three as identical HTTP endpoints would hide the integration work that distinguishes them.\n\nThe interface names also require care. OpenAI uses `input` and an ordered `questions` array. Perplexity uses `state` and named questions, even though its endpoint is also `/v1/decisions`. Upstage documents the SystemOne-shaped `state` and question map. Check the complete request and response contract when moving between providers.\n\nAvailability and capabilities vary independently. OpenAI's product entered [public beta on October 6, 2026](https://developers.openai.com/api/docs/changelog). Snowflake's release note dates its private preview to October 7, and Upstage labels Solar Decide beta with a schema that may change before general availability. Cloudflare's October 9 update adds Clef-omni's audio and video inputs; those capabilities should not be assumed for every decision model.\n\n## Open models and serving tools\n\nThe ecosystem also includes open decision models such as Laya and Tev1 and serving tools such as LocalAI and Ollama. These play different roles: a model supplies the judgment, while the server exposes it to application code. Amazon's Strands Decider similarly includes a local server alongside its model and development tools.\n\n[LocalAI](https://localai.io/docs/features/decisions/) documents both `/v1/systemone` and OpenAI-compatible `/v1/decisions` routes for installed decision models. Its guide lists model and backend requirements, and warns that compatible output fields do not establish equivalent calibration or model quality. Image support depends on the installed model and backend.\n\n[Pydantic AI's System One integration](https://pydantic.dev/docs/ai/models/system-one/) supports decision models behind that interface, including models served locally by Ollama. Its broader decision-model abstraction also supports TypeSafe Jev and OpenAI Decisions. A shared client can simplify integration, while provider and model profiles retain differences in limits and capabilities.\n\nFor the launch history, practical applications and caveats behind vendor performance claims, read TheQuery's [Decisions API Use Cases Are Real. Scores Are Self-Graded.](/articles/decisions-api-use-cases-self-graded-scores).\n\n## Probabilities, confidence and thresholds\n\nAn option probability, a rubric score and a confidence field can mean different things. Read the provider's definitions before using any of them as a routing threshold. Pydantic AI, for example, documents its per-field confidence as a margin rather than the probability that the answer is correct. Numeric fields with the same name need not be comparable across clients or models.\n\nCalibration is the relationship between reported probability and observed outcomes. It resembles checking a weather forecast over many days: events assigned an 80% probability should occur about 80% of the time in comparable cases. A number between zero and one does not establish that relationship by itself.\n\nSelect thresholds using labeled examples from the application's workload and the cost of mistakes. A relevance filter and an automatic escalation rule may need different thresholds even when they use the same model. Test the fallback path as well as the ordinary path. An explicit \"none of these\" option, an uncertain answer, a refusal and an API failure can require different handling.\n\n## Evaluating a decision workflow\n\nEvaluate the complete workflow against representative examples. Measure classification errors, how often cases require review, latency and cost per completed task. Include ambiguous inputs and cases outside the supplied categories. A valid label demonstrates conformance to an answer format; task accuracy requires separate evidence.\n\nCompare providers using the same evidence, questions and outcome definitions. Recheck thresholds when changing model versions or providers. Validate supported modalities and request limits before relying on a shared schema. For local deployments, account for the model, serving backend and server logging; for hosted services, check the provider's processing and retention terms.\n\nA decisions API fits a repeatable judgment whose possible outcomes can be described in advance. Writing a reply, explaining an assessment or planning a new sequence of steps may require a language model elsewhere in the workflow. Combining these components lets each stage expose the output the next stage actually needs.\n\nDefine the judgment. Test the resulting action.",
  "sources": [
    {
      "title": "TypeSafe AI: Introducing System One Models and Jev",
      "url": "https://typesafe.ai/blog/introducing-system-one-models-and-jev"
    },
    {
      "title": "OpenAI: Decisions API guide",
      "url": "https://developers.openai.com/api/docs/guides/decisions"
    },
    {
      "title": "Microsoft: Introducing Microsoft-Decision-1",
      "url": "https://commandline.microsoft.com/microsoft-decision-1-model-foundry/"
    },
    {
      "title": "Perplexity: Decisions API quickstart",
      "url": "https://docs.perplexity.ai/docs/decisions/quickstart"
    },
    {
      "title": "Cloudflare: Clef-omni and the updated Clef family",
      "url": "https://blog.cloudflare.com/clef-faster-cheaper-multimodal/"
    },
    {
      "title": "AWS Strands: Strands Decider repository and serving guide",
      "url": "https://github.com/strands-labs/strands-decider"
    },
    {
      "title": "Snowflake: Snowflake Decision private preview release note",
      "url": "https://docs.snowflake.com/en/release-notes/2026/other/2026-10-07-ai-complete-snowflake-decision-preview"
    },
    {
      "title": "Upstage: Solar Decide System One API reference",
      "url": "https://console.upstage.ai/api/systemone"
    },
    {
      "title": "Pydantic AI: Decision models",
      "url": "https://pydantic.dev/docs/ai/models/decision/"
    },
    {
      "title": "LocalAI: Decisions API guide",
      "url": "https://localai.io/docs/features/decisions/"
    }
  ],
  "metadata": {
    "category": "Systems, Tools & Safety",
    "relatedTerms": [
      "api",
      "text-classification",
      "query-routing",
      "tool-calling",
      "ai-agent"
    ],
    "seoDescription": "Decisions APIs explained: bounded judgments, scoring and routing, with Microsoft, OpenAI, Perplexity, Cloudflare, Amazon, TypeSafe, Snowflake and Upstage.",
    "seoKeywords": [
      "Decisions API",
      "decision API category",
      "AI decision APIs",
      "decision models",
      "typed decisions",
      "decision API routing",
      "decision API confidence",
      "SystemOne API",
      "decisions-api",
      "Microsoft-Decision-1",
      "Perplexity Decisions API",
      "Cloudflare Clef",
      "Strands Decider",
      "Snowflake Decision",
      "Solar Decide"
    ]
  }
}
$entry$::jsonb) AS entry(
  id TEXT, slug TEXT, title TEXT, summary TEXT, body TEXT, sources JSONB, metadata JSONB
)
WHERE content.kind = 'glossary' AND content.slug = entry.slug
  AND content.parent_slug = '' AND content.id = entry.id;
