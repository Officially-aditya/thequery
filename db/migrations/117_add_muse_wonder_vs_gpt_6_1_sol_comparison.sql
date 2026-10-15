-- Publish a minimal authored comparison entry for Muse Wonder vs GPT 6.1 Sol
-- so the pair is listed among authored comparisons and participates in
-- internal backlinks. The detailed spec tables are rendered at runtime from
-- the model catalog because this row carries only a markdown overview.
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
) VALUES (
  'comparison:muse-wonder-vs-gpt-6-1-sol',
  'comparison',
  'muse-wonder-vs-gpt-6-1-sol',
  '',
  'comparisons/muse-wonder-vs-gpt-6-1-sol',
  'Muse Wonder vs GPT 6.1 Sol',
  'Muse Wonder vs GPT 6.1 Sol: Meta Superintelligence Labs'' frontier model against OpenAI''s complex everyday-work flagship. Side-by-side specifications and benchmark tables are generated from TheQuery''s verified model catalog.',
  'Muse Wonder is Meta Superintelligence Labs'' October 12, 2026 frontier model for personal and professional agentic tasks, available through the Meta Model API preview and OpenRouter. GPT-6.1 Sol is OpenAI''s September 29, 2026 model built for coding agents, computer use, and document-heavy work. This page compares the two from TheQuery''s verified model catalog; see the [Muse Wonder](/glossary/muse-wonder) and [GPT-6.1 Sol](/glossary/gpt-6-1-sol) model cards for the full caveats.',
  '[{"id":"musewonder-gpt61sol-overview","type":"markdown","content":"## Overview\n\nThis comparison places [Muse Wonder](/glossary/muse-wonder) beside [GPT-6.1 Sol](/glossary/gpt-6-1-sol). Muse Wonder is Meta Superintelligence Labs'' frontier model for personal and professional agentic tasks; GPT-6.1 Sol is OpenAI''s flagship for complex everyday work. The specification and benchmark tables below are generated from TheQuery''s verified model catalog."}]'::jsonb,
  '[
    {"title": "Introducing Muse Wonder", "url": "https://research.meta.ai/blog/muse-wonder"},
    {"title": "Introducing GPT-6.1 Sol", "url": "https://openai.com/index/introducing-gpt-6-1-sol/"}
  ]'::jsonb,
  '{"modelA":"muse-wonder","modelB":"gpt-6-1-sol"}'::jsonb,
  NULL,
  NULL,
  'published',
  DATE '2026-10-12',
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
