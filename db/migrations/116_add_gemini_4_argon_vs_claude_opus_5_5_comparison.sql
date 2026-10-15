-- Publish a minimal authored comparison entry for Gemini 4 Argon vs Claude
-- Opus 5.5 so the pair is listed among authored comparisons and participates
-- in internal backlinks. The detailed spec tables are rendered at runtime from
-- the model catalog because this row carries only a markdown overview.
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
) VALUES (
  'comparison:gemini-4-argon-vs-opus-5-5',
  'comparison',
  'gemini-4-argon-vs-opus-5-5',
  '',
  'comparisons/gemini-4-argon-vs-opus-5-5',
  'Gemini 4 Argon vs Claude Opus 5.5',
  'Gemini 4 Argon vs Claude Opus 5.5: Google''s restricted-access flagship for long-horizon coding and enterprise work against Anthropic''s September 2026 agentic flagship. Side-by-side specifications and benchmark tables are generated from TheQuery''s verified model catalog.',
  'Gemini 4 Argon is Google DeepMind''s September 30, 2026 frontier model for long-horizon software engineering, enterprise knowledge work, and cybersecurity defense. It is available only through Google''s trusted-defender and internal programs at launch, with no public API model ID or price published yet. Claude Opus 5.5 is Anthropic''s September 22, 2026 flagship for long-running agentic coding and professional knowledge work. This page compares the two from TheQuery''s verified model catalog; see the [Gemini 4 Argon](/glossary/gemini-4-argon) and [Claude Opus 5.5](/glossary/claude-opus-5-5) model cards for the full caveats.',
  '[{"id":"gemini4argon-opus55-overview","type":"markdown","content":"## Overview\n\nThis comparison places [Gemini 4 Argon](/glossary/gemini-4-argon) beside [Claude Opus 5.5](/glossary/claude-opus-5-5). Argon is Google DeepMind''s restricted-access flagship for long-horizon coding and enterprise work; Opus 5.5 is Anthropic''s flagship for long-running agentic coding. The specification and benchmark tables below are generated from TheQuery''s verified model catalog."}]'::jsonb,
  '[
    {"title": "Introducing Gemini 4 Argon", "url": "https://blog.google/technology/ai/gemini-4-argon/"},
    {"title": "Claude Opus 5.5", "url": "https://www.anthropic.com/claude-opus-5-5"}
  ]'::jsonb,
  '{"modelA":"gemini-4-argon","modelB":"claude-opus-5-5"}'::jsonb,
  NULL,
  NULL,
  'published',
  DATE '2026-09-30',
  0
)
ON CONFLICT (id) DO UPDATE SET
  kind = EXCLUDED.kind,
  slug = EXCLUDED.slug,
  parent_slug = EXCLUDED.parent_slug,
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
  sort_order = EXCLUDED.sort_order;
