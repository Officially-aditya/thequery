UPDATE models
SET
  slug = 'gemini-omni-flash',
  name = 'Gemini Omni Flash',
  release_date = DATE '2026-05-19',
  comparison_data = jsonb_build_object(
    'Developer', 'Google DeepMind',
    'Release date', '2026-05-19',
    'API model ID', 'gemini-omni-flash-preview (API preview from June 30)',
    'Text input', 'Yes',
    'Image / vision input', 'Yes',
    'Audio input', 'Yes',
    'Video input', 'Yes',
    'Text output', 'No',
    'Image output', 'No',
    'Audio output', 'Generated with video',
    'Video output', 'Yes',
    'API access', 'Preview from June 30',
    'Product access', 'Gemini app + Flow + YouTube + API preview',
    'Weights / license', 'Proprietary'
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Introducing Gemini Omni', 'url', 'https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-omni/'),
    jsonb_build_object('title', 'Google I/O 2026 announcements', 'url', 'https://blog.google/innovation-and-ai/technology/ai/google-io-2026-all-our-announcements/'),
    jsonb_build_object('title', 'Gemini Omni Flash API docs', 'url', 'https://ai.google.dev/gemini-api/docs/models/gemini-omni-flash')
  ),
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'gemini-omni-flash-preview';

INSERT INTO models (slug, name, developer, release_date, access, comparison_data, sources, verified_at)
VALUES (
  'gemini-omni-1-1-flash',
  'Gemini Omni 1.1 Flash',
  'Google DeepMind',
  DATE '2026-08-27',
  'proprietary',
  jsonb_build_object(
    'Developer', 'Google DeepMind',
    'Release date', '2026-08-27',
    'API model ID', 'gemini-omni-1.1-flash',
    'Context window', '1,048,576 tokens',
    'Text input', 'Yes',
    'Image / vision input', 'Yes',
    'Audio input', 'Not listed in the model-card supported data types',
    'Video input', 'Yes',
    'Text output', 'No',
    'Image output', 'No',
    'Audio output', 'Generated with video',
    'Video output', 'Yes',
    'API access', 'Yes',
    'Product access', 'Google AI Studio + Gemini Enterprise Agent Platform + Gemini app + Flow',
    'Weights / license', 'Proprietary'
  ),
  jsonb_build_array(
    jsonb_build_object('title', 'Gemini Omni 1.1 Flash announcement', 'url', 'https://blog.google/innovation-and-ai/technology/developers-tools/build-with-gemini-omni-1-1-flash/'),
    jsonb_build_object('title', 'Gemini Omni Flash API docs', 'url', 'https://ai.google.dev/gemini-api/docs/models/gemini-omni-flash'),
    jsonb_build_object('title', 'Gemini API deprecations', 'url', 'https://ai.google.dev/gemini-api/docs/deprecations')
  ),
  NOW()
)
ON CONFLICT (slug) DO UPDATE SET
  name = EXCLUDED.name,
  developer = EXCLUDED.developer,
  release_date = EXCLUDED.release_date,
  access = EXCLUDED.access,
  comparison_data = EXCLUDED.comparison_data,
  sources = EXCLUDED.sources,
  verified_at = EXCLUDED.verified_at,
  updated_at = NOW();

INSERT INTO models (slug, name, developer, release_date, access, comparison_data, sources, verified_at)
VALUES (
  'qwen3-8-27b',
  'Qwen3.8-27B',
  'Alibaba / Qwen',
  DATE '2026-08-17',
  'open_source',
  jsonb_build_object(
    'Developer', 'Alibaba / Qwen',
    'Release date', '2026-08-17',
    'Context window', '1M tokens on hosted Qwen service',
    'Text input', 'Yes',
    'Image / vision input', 'Yes',
    'Text output', 'Yes',
    'API access', 'Model Studio / hosted service',
    'Product access', 'Downloadable weights + hosted service',
    'Weights / license', 'Open source — Apache 2.0'
  ),
  jsonb_build_array(
    jsonb_build_object('title', 'Alibaba Cloud model releases', 'url', 'https://www.alibabacloud.com/help/en/model-studio/newly-released-models'),
    jsonb_build_object('title', 'Qwen3.8-27B official weights', 'url', 'https://huggingface.co/Qwen/Qwen3.8-27B')
  ),
  NOW()
)
ON CONFLICT (slug) DO UPDATE SET
  name = EXCLUDED.name,
  developer = EXCLUDED.developer,
  release_date = EXCLUDED.release_date,
  access = EXCLUDED.access,
  comparison_data = EXCLUDED.comparison_data,
  sources = EXCLUDED.sources,
  verified_at = EXCLUDED.verified_at,
  updated_at = NOW();

UPDATE content_items AS item
SET
  blocks = (
    SELECT jsonb_agg(
      CASE
        WHEN block->>'type' = 'spec_table'
          AND lower(COALESCE(block->>'title', '')) = 'capabilities & access'
        THEN jsonb_set(
          block,
          '{rows}',
          COALESCE(block->'rows', '[]'::jsonb)
          || CASE
               WHEN NOT EXISTS (
                 SELECT 1 FROM jsonb_array_elements(COALESCE(block->'rows', '[]'::jsonb)) AS row
                 WHERE row->>0 = 'Image output'
               )
               THEN jsonb_build_array(jsonb_build_array('Image output', '', ''))
               ELSE '[]'::jsonb
             END
          || CASE
               WHEN NOT EXISTS (
                 SELECT 1 FROM jsonb_array_elements(COALESCE(block->'rows', '[]'::jsonb)) AS row
                 WHERE row->>0 = 'Video output'
               )
               THEN jsonb_build_array(jsonb_build_array('Video output', '', ''))
               ELSE '[]'::jsonb
             END,
          true
        )
        ELSE block
      END
      ORDER BY ordinality
    )
    FROM jsonb_array_elements(item.blocks) WITH ORDINALITY AS blocks(block, ordinality)
  ),
  updated_at = NOW()
WHERE item.kind = 'comparison';
