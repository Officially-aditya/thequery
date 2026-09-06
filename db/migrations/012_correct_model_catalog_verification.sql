UPDATE models
SET
  release_date = DATE '2026-04-28',
  access = 'open_weights',
  comparison_data = comparison_data || jsonb_build_object(
    'Release date', '2026-04-28',
    'API model ID', 'mistral-medium-3-5',
    'Context window', '256K tokens',
    'Reasoning / effort', 'Adjustable reasoning',
    'Weights / license', 'Open weights — Modified MIT'
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Mistral Medium 3.5 model docs', 'url', 'https://docs.mistral.ai/models/mistral-medium-3-5-26-04'),
    jsonb_build_object('title', 'Mistral changelog', 'url', 'https://docs.mistral.ai/resources/changelogs')
  ),
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'mistral-medium-3-5';

UPDATE models
SET
  access = 'open_source',
  comparison_data = comparison_data || jsonb_build_object(
    'Product access', 'API + downloadable weights',
    'Weights / license', 'Open source weights — MIT'
  ),
  sources = sources || jsonb_build_array(
    jsonb_build_object('title', 'GLM-5.1 official weights', 'url', 'https://huggingface.co/zai-org/GLM-5.1')
  ),
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'glm-5-1';

UPDATE models
SET
  access = 'open_source',
  comparison_data = comparison_data || jsonb_build_object(
    'Product access', 'API + downloadable weights',
    'Weights / license', 'Open source weights — MIT'
  ),
  sources = sources || jsonb_build_array(
    jsonb_build_object('title', 'GLM-5.2 official weights', 'url', 'https://huggingface.co/zai-org/GLM-5.2')
  ),
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'glm-5-2';

UPDATE models
SET
  access = 'open_weights',
  comparison_data = comparison_data || jsonb_build_object(
    'Product access', 'API + downloadable weights',
    'Weights / license', 'Open weights — GLM-5.3 License'
  ),
  sources = sources || jsonb_build_array(
    jsonb_build_object('title', 'GLM-5.3 official weights', 'url', 'https://huggingface.co/zai-org/GLM-5.3')
  ),
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'glm-5-3';

UPDATE models
SET
  access = 'open_source',
  comparison_data = comparison_data || jsonb_build_object(
    'Image / vision input', 'Yes',
    'Text output', 'Yes',
    'Product access', 'API + downloadable weights',
    'Weights / license', 'Open source weights — MIT'
  ),
  sources = sources || jsonb_build_array(
    jsonb_build_object('title', 'GLM-5.3-Flash official weights', 'url', 'https://huggingface.co/zai-org/GLM-5.3-Flash')
  ),
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'glm-5-3-flash';
