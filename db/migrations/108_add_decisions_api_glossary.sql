-- Publish the Decisions API glossary entry from the official guide,
-- endpoint reference, changelog and data-control policy.
-- Image-URL documentation differences and public-beta status stay explicit.
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order, updated_at
)
SELECT
  entry.id, 'glossary', entry.slug, '', 'glossary/' || entry.slug,
  entry.title, entry.summary, entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', entry.body)),
  entry.sources, entry.metadata, NULL, NULL, 'published', DATE '2026-10-12', 0,
  TIMESTAMPTZ '2026-10-12 00:00:00+00'
FROM jsonb_to_recordset($glossary$
[
  {
    "id": "glossary:decisions-api",
    "slug": "decisions-api",
    "title": "Decisions API",
    "summary": "OpenAI's GPT-6 Luna-powered public-beta endpoint that turns text and images into predicate probabilities, fixed-option choices and rubric scores, with input-only billing.",
    "body": "The **Decisions API** is OpenAI's hosted [API](/glossary/api) for turning text and images into bounded judgments that an application can use directly. It returns an estimated probability, a selection from predefined options, or a score against an ordered rubric. OpenAI's changelog records its public-beta release on October 6, 2026. The documentation reviewed on October 12 lists [GPT-6 Luna](/glossary/gpt-6-luna), requested as `gpt-6-luna`, as the only supported model.\n\nThe dedicated endpoint is `POST https://api.openai.com/v1/decisions`. A useful analogy is a switchboard operator: the application supplies the available destinations, and the service helps decide where an incoming request should go. Typical uses include document classification, support routing, relevance checks and choosing an [AI agent](/glossary/ai-agent)'s next step. The application still defines what each returned answer causes it to do.\n\n## The three question types\n\nA request supplies shared evidence in `input` and the evaluations to perform in `questions`. Each question contains instructions and one of three types.\n\n| Question type | Returned judgment | Appropriate use |\n| --- | --- | --- |\n| `predicate` | `probability`, from 0 to 1, that a condition is true | Whether a photo shows damage or a passage is relevant |\n| `choice` | A supplied `choice`, a probability distribution and separate `confidence` | Which support queue or document category applies |\n| `score` | A probability-weighted `score` across ordered level indices, plus probabilities and confidence | Severity, urgency or another graded assessment |\n\nA predicate resembles a weather forecast: a probability describes how strongly the evidence supports an outcome. The application chooses the threshold that turns that estimate into a flag. It should not assume that a predicate returns a Boolean answer.\n\nA choice uses categories without an inherent order. The endpoint reference allows 2 to 255 unique choices, with string or Boolean values. Clear descriptions help distinguish overlapping labels. A catch-all option can represent requests outside the named categories instead of making every input fit a narrow set of destinations.\n\nA score resembles a teacher's rubric. Its ordered levels receive indices starting at zero, and the result can fall between those indices. For an illustrative three-level rubric, probabilities of 0.1, 0.6 and 0.3 across levels 0, 1 and 2 produce a score of 1.2. That summarizes the distribution; it does not select a single category. Use `choice` when the application needs one discrete option.\n\n## Requests and typed answers\n\nThe main request fields are `model`, `input` and `questions`. Naming each question makes its answer easier to identify. Names are optional in the endpoint schema, and returned answers can have a null name when none was supplied. Answers are returned in the question order.\n\nThis illustrative JSON request routes a product-support report:\n\n```json\n{\n  \"model\": \"gpt-6-luna\",\n  \"input\": \"The export button stops working after I select PDF.\",\n  \"questions\": [\n    {\n      \"type\": \"choice\",\n      \"name\": \"support_queue\",\n      \"instructions\": \"Choose the support queue that should investigate this report.\",\n      \"choices\": [\n        {\"value\": \"product\", \"description\": \"Broken application features.\"},\n        {\"value\": \"billing\", \"description\": \"Charges and invoice questions.\"},\n        {\"value\": \"other\", \"description\": \"Requests outside these areas.\"}\n      ]\n    }\n  ]\n}\n```\n\nThe response contains an `answers` array. Software should inspect each answer's `type` before using type-specific fields. A question can return `refusal`, which is a separate answer variant rather than a normal choice with low confidence. A refusal handler and a low-confidence fallback therefore address different outcomes.\n\nSeveral independent questions can share the same input in one request, much as several inspectors can examine the same photograph for different properties. If a later question depends on an earlier answer, the guide recommends separate requests. Putting dependent questions together does not establish the application-level sequence they require.\n\n## Decisions versus structured generation and tool calling\n\nThe choice of interface follows the output the application needs.\n\n| Interface | Output needed | Typical role |\n| --- | --- | --- |\n| Decisions API | Predicate probability, fixed choice or ordered rubric score | Classify, route, filter or prioritize |\n| Responses API with Structured Outputs | An object matching a custom JSON schema | Extract named fields or produce a structured report |\n| [Tool Calling](/glossary/tool-calling) | A requested function and its arguments | Ask an application to execute a tool operation |\n\nStructured Outputs resemble a form with fields that must be filled in. Tool calling resembles a work order naming an operation and its parameters. A Decisions choice can select a destination or action label, while the surrounding application supplies the code, arguments and permissions for the next operation. The Decisions endpoint itself does not accept tools or function-call items as input.\n\nFor example, a service can classify an incoming request, then send it to a specialist model or a human queue. If the task requires an explanation, a complete extraction schema, or generated tool arguments, the Responses-based interfaces remain appropriate. A bounded decision output does not replace every step in an agent workflow.\n\n## Inputs and documented limits\n\nText can be supplied as a string. Multimodal input uses user messages containing `input_text` and `input_image` parts. The endpoint reference lists a maximum of 128 images across the request and excludes file IDs, file inputs, audio, other message roles, tools and item references.\n\nThere is a documentation difference worth keeping visible. The retrieved guide describes inline base64 image data URLs only, while the endpoint reference allows base64 data URLs or publicly accessible HTTP(S) image URLs. The reference is the source for the field constraints described here; developers should check its current contract and verify image handling in their integration.\n\nVoice-control integrations are separate from native audio input to this endpoint. The guide links a Live API client-delegation workflow for voice-driven actions, but that does not change the Decisions request schema into an audio-input API.\n\n## Confidence and routing policies\n\nPredicates expose the estimated probability that a condition is true. Choices and scores expose probabilities over the supplied options and a separate `confidence` value. The score, confidence and largest option probability are distinct fields and should not be treated as interchangeable.\n\nA confidence threshold acts like a quality-control checkpoint: straightforward cases pass through, while uncertain cases go to another process. OpenAI recommends using labeled examples from the application's own workload to select thresholds according to the cost of false positives and false negatives. A routing rule can send uncertain requests for review without making the service's numeric output a guarantee of correctness.\n\nThe question design also matters. Observable criteria, distinct categories and clearly separated rubric levels reduce avoidable ambiguity. A formally valid returned label can still be the wrong label for the task. Evaluate the complete routing policy, including catch-all choices, refusal handling and review paths, rather than judging the system only by whether it returns valid JSON.\n\n## Pricing, speed and availability\n\nThe documented base price for `gpt-6-luna` on `/v1/decisions` is $0.10 per million input tokens. There are no cache-read, cache-write or output-token charges for this endpoint. A request using 10,000 billed input tokens therefore has a base token cost of $0.001, before applicable adjustments. Regional processing premiums and long-context input multipliers still apply. Other request types using GPT-6 Luna follow their applicable model and processing-tier rates.\n\nOpenAI describes Decisions as about ten times faster than using the Responses API for these typed judgments. This is the provider's performance claim. The sources reviewed here do not establish a universal fixed latency for every input, question set or deployment region. Public beta status also remains distinct from general availability.\n\n## Data controls and deployment fit\n\nOpenAI's data-control documentation states that default abuse-monitoring logs for this endpoint can be retained for up to 30 days. Eligible customers can use Zero Data Retention, subject to documented limitations, including prompt-cache application state and image-safety retention exceptions. Billing rules and retention rules are separate policies.\n\nThe endpoint is available in supported API regions, with regional processing documented in the United States and Europe, including the EEA and Switzerland. HIPAA use requires the executed OpenAI agreement and the applicable account configuration. Those capabilities depend on eligibility and setup; adding an endpoint to an application does not establish them by itself.\n\nThe useful deployment boundary is a clearly specified judgment with a defined application policy. Decide which outputs are allowed, test them against representative labeled examples, and make the follow-on action explicit.\n\nDefine the choices. Validate the decisions.",
    "sources": [
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
      },
      {
        "title": "OpenAI platform data controls",
        "url": "https://developers.openai.com/api/docs/guides/your-data"
      }
    ],
    "metadata": {
      "category": "Systems, Tools & Safety",
      "relatedTerms": [
        "api",
        "gpt-6-luna",
        "tool-calling",
        "ai-agent"
      ],
      "seoDescription": "Decisions API explained: OpenAI's GPT-6 Luna endpoint for predicates, fixed choices and rubric scores, confidence outputs, input-only pricing and limits.",
      "seoKeywords": [
        "Decisions API",
        "OpenAI Decisions API",
        "GPT-6 Luna Decisions API",
        "Decisions API pricing",
        "Decisions API confidence",
        "predicate choice score",
        "decisions-api"
      ]
    }
  }
]
$glossary$::jsonb) AS entry(
  id TEXT, slug TEXT, title TEXT, summary TEXT, body TEXT, sources JSONB, metadata JSONB
)
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path, title = EXCLUDED.title, summary = EXCLUDED.summary,
  body = EXCLUDED.body, blocks = EXCLUDED.blocks, sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata, status = EXCLUDED.status,
  published_at = EXCLUDED.published_at, updated_at = EXCLUDED.updated_at;
