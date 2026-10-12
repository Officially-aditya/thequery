-- Define decisions APIs as an interface category, with multiple implementations.
-- Preserve the existing URL and publication date.
UPDATE content_items AS content
SET title = entry.title, summary = entry.summary, body = entry.body,
    blocks = jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', entry.body)),
    sources = entry.sources, metadata = entry.metadata,
    updated_at = TIMESTAMPTZ '2026-10-12 00:00:00+00'
FROM jsonb_to_record($entry$
{
  "id": "glossary:decisions-api",
  "slug": "decisions-api",
  "title": "Decisions API",
  "summary": "An API category that turns evidence into bounded judgments, such as labels, predicate probabilities and rubric scores, for classification, routing and prioritization.",
  "body": "A **decisions API**, also called a **decision API**, is an [API](/glossary/api) category for turning evidence into bounded judgments that software can use directly. In an AI workflow, the caller supplies context, a question and a defined answer space. The service returns a label, a true-or-false assessment, or a rubric score, often accompanied by probabilities or confidence information. The application uses that result to classify, route, filter or prioritize work.\n\nThe category includes hosted services and locally served decision models. OpenAI's **Decisions API** is one implementation. TypeSafe's Jev and open models served through SystemOne-compatible interfaces provide other implementations. [Pydantic AI documents](https://pydantic.dev/docs/ai/models/decision/) these under a shared decision-model abstraction, while keeping their provider-specific configuration and limits separate.\n\nThink of a switchboard operator: the destinations already exist, and the task is to choose where an incoming request belongs. A decisions API makes that choice available to application code. The category describes this interface pattern; it does not establish one universal endpoint, request schema, model architecture or pricing policy.\n\n## How a decisions API works\n\nA request separates the **evidence** from the **judgment to make**. The evidence might be a support ticket, a document passage, the state of an agent workflow, or an image when the implementation supports images. The question states what to assess, and the options or rubric describe the permitted outcomes.\n\nFor example, an application can submit a customer message with the question, \"Which team should handle this?\" It supplies billing, technical support and other as the available categories, with descriptions that distinguish them. The service returns a selected category and whatever uncertainty information its contract exposes. Application code then places the ticket in that queue.\n\nSeveral independent questions can assess the same evidence, such as the responsible team and whether a customer explicitly requested cancellation. Some implementations answer them together. When a later question depends on an earlier answer, the application must establish the sequence rather than assume that a batch of questions creates it.\n\n## Common output forms\n\nTyped decision services commonly expose three patterns. Their exact names and returned fields differ between implementations.\n\n| Decision pattern | Meaning | Example | Implementation detail to check |\n| --- | --- | --- | --- |\n| Predicate or yes/no assessment | Whether a stated condition holds, or its estimated probability | Does the message request cancellation? | OpenAI calls this `predicate`; SystemOne uses `noul`. An API or client may return a probability or convert it to a Boolean. |\n| Fixed-option choice | Select one of the supplied alternatives | Billing, technical support or other | Check option limits, descriptions, probability fields and whether an explicit fallback option is supported. |\n| Ordered rubric score | Assess evidence against described levels | Routine, urgent or critical | A score may fall between level indices. Verify the scale and whether the client rounds it. |\n\nA rubric works like a teacher's marking guide: each level describes an observable situation. \"The customer cannot use a paid feature\" gives a model more specific evidence to assess than \"very urgent.\" Categories need equally clear boundaries, particularly when an input could reasonably fit more than one.\n\nA weighted rubric score and a discrete label are different outputs. In OpenAI's implementation, probabilities of 0.1, 0.6 and 0.3 across levels 0, 1 and 2 yield a score of 1.2. That calculation summarizes the distribution. An application requiring one category should use a choice or define an explicit conversion policy.\n\n## The API, the model and the application\n\nThe API is the contract between software components. The **decision model** is the system producing the judgment behind that contract. A local serving layer can expose different models through the same interface, while the same model can be exposed through more than one request format. [LocalAI](https://localai.io/docs/features/decisions/), for example, documents both SystemOne and OpenAI-compatible decisions endpoints sharing the same execution layer.\n\nAn API name alone does not reveal how a model computes an answer. Dedicated decision models can score bounded outcomes without generating a written reply. General language models can also perform classification through constrained structured output. Compare the actual implementation and workload before assuming that every decisions API uses the same architecture or inference process.\n\nThe application defines what happens after the judgment. Choosing \"billing\" does not execute a refund. Choosing an [AI agent](/glossary/ai-agent)'s next step does not supply every argument or permission needed to carry it out. The decision result is one input to the surrounding workflow, alongside account state and explicit application rules.\n\n## Related interfaces compared\n\n| Interface | Core task | Typical output | Appropriate use |\n| --- | --- | --- | --- |\n| Decisions API | Judge evidence against bounded outcomes | Label, probability or rubric score | Triage, relevance filtering and [query routing](/glossary/query-routing) |\n| General model with structured output | Generate content within a schema | Custom object with extracted or generated fields | Document extraction, structured reports and tasks needing free-form fields |\n| [Tool Calling](/glossary/tool-calling) | Request an application operation | Function name and arguments | Select and parameterize a tool for the application to execute |\n| Deterministic rules service | Apply explicit conditions | Rule result or prescribed branch | Enforce known limits and business rules |\n\nStructured output resembles a form: it constrains the shape of the answer. Tool calling resembles a work order: it names an operation and supplies its parameters. These interfaces overlap with decision workflows. A schema containing a fixed enum can represent a classification, and a decision client can turn selected routes into tool calls. The useful distinction is the work required and the contract exposed to the application, rather than an assumption that only one interface can classify text.\n\n## Implementations in the category\n\nThe following examples illustrate deployment and interface differences, based on documentation reviewed on October 12, 2026. They are not a benchmark ranking.\n\n| Implementation | Deployment and interface | Documented characteristics |\n| --- | --- | --- |\n| OpenAI Decisions API | Hosted `POST /v1/decisions` | [GPT-6 Luna](/glossary/gpt-6-luna); text and image inputs; `predicate`, `choice` and `score` questions with typed answers. |\n| TypeSafe Jev | Hosted TypeSafe service; SystemOne decision interface | Typed judgments over text with probability information; supported by Pydantic AI's TypeSafe integration. |\n| LocalAI decision serving | Local or self-hosted `/v1/systemone` and OpenAI-compatible `/v1/decisions` | Serves installed decision models, including Laya and Tev1. Capabilities and limits depend on the model and backend. |\n\nOpenAI's product entered public beta on October 6, 2026. Its documented base rate is $0.10 per million input tokens, with no output, cache-read or cache-write token charges for this endpoint; applicable regional and long-context adjustments remain. These are OpenAI-specific terms, not category-wide pricing rules. Its supported model, input limits and retention policies also belong to that service.\n\nPydantic AI's decision-model abstraction supports TypeSafe Jev, SystemOne-compatible providers and OpenAI Decisions. A shared client abstraction can simplify integration, but changing the provider can change supported inputs, output semantics and model quality. LocalAI explicitly warns that OpenAI-compatible fields do not establish OpenAI-equivalent calibration or behavior.\n\n## Probabilities, confidence and thresholds\n\nAn option probability, a rubric score and a confidence field can mean different things. Read the provider's definitions before using any of them as a routing threshold. Pydantic AI, for example, documents its per-field confidence as a margin rather than the probability that the answer is correct. Numeric fields with the same name need not be comparable across clients or models.\n\nCalibration is the relationship between reported probability and observed outcomes. It resembles checking a weather forecast over many days: events assigned an 80% probability should occur about 80% of the time in comparable cases. A number between zero and one does not establish that relationship by itself.\n\nSelect thresholds using labeled examples from the application's workload and the cost of mistakes. A relevance filter and an automatic escalation rule may need different thresholds even when they use the same model. Test the fallback path as well as the ordinary path. An explicit \"none of these\" option, an uncertain answer, a refusal and an API failure can require different handling.\n\n## Evaluating a decision workflow\n\nEvaluate the complete workflow against representative examples. Measure classification errors, how often cases require review, latency and cost per completed task. Include ambiguous inputs and cases outside the supplied categories. A valid label demonstrates conformance to an answer format; task accuracy requires separate evidence.\n\nCompare providers using the same evidence, questions and outcome definitions. Recheck thresholds when changing model versions or providers. Validate supported modalities and request limits before relying on a shared schema. For local deployments, account for the model, serving backend and server logging; for hosted services, check the provider's processing and retention terms.\n\nA decisions API fits a repeatable judgment whose possible outcomes can be described in advance. Writing a reply, explaining an assessment or planning a new sequence of steps may require a language model elsewhere in the workflow. Combining these components lets each stage expose the output the next stage actually needs.\n\nDefine the judgment. Test the resulting action.",
  "sources": [
    {
      "title": "Pydantic AI: Decision models",
      "url": "https://pydantic.dev/docs/ai/models/decision/"
    },
    {
      "title": "Pydantic AI: System One API",
      "url": "https://pydantic.dev/docs/ai/models/system-one/"
    },
    {
      "title": "LocalAI: Decisions API guide",
      "url": "https://localai.io/docs/features/decisions/"
    },
    {
      "title": "OpenAI: Decisions API guide",
      "url": "https://developers.openai.com/api/docs/guides/decisions"
    },
    {
      "title": "OpenAI: Create a decision endpoint reference",
      "url": "https://developers.openai.com/api/reference/resources/decisions/methods/create"
    },
    {
      "title": "OpenAI API changelog: Decisions beta release",
      "url": "https://developers.openai.com/api/docs/changelog"
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
    "seoDescription": "Decisions APIs explained as a category: bounded judgments, routing, rubric scores, confidence, and hosted or local implementations across providers.",
    "seoKeywords": [
      "Decisions API",
      "decision API category",
      "AI decision APIs",
      "decision models",
      "typed decisions",
      "decision API routing",
      "decision API confidence",
      "SystemOne API",
      "decisions-api"
    ]
  }
}
$entry$::jsonb) AS entry(
  id TEXT, slug TEXT, title TEXT, summary TEXT, body TEXT, sources JSONB, metadata JSONB
)
WHERE content.kind = 'glossary' AND content.slug = entry.slug
  AND content.parent_slug = '' AND content.id = entry.id;
