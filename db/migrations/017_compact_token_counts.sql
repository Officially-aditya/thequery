-- Normalize large token counts to compact comparison-friendly notation.
-- Examples: 1,050,000 -> 1.05M, 1,000,000 -> 1M, 131,072 -> 128K, 128,000 -> 128K.
-- This updates canonical model snapshots and already-materialized comparison content.

UPDATE models
SET
  comparison_data = replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(
    comparison_data::text,
    '1,050,000 tokens', '1.05M tokens'),
    '1,048,576 tokens', '1M tokens'),
    '1,000,000 tokens', '1M tokens'),
    '400,000 tokens', '400K tokens'),
    '256,000 tokens', '256K tokens'),
    '200,000 tokens', '200K tokens'),
    '131,072 input tokens', '128K input tokens'),
    '131,072 tokens', '128K tokens'),
    '128,000 tokens', '128K tokens'),
    '96,000 tokens', '96K tokens'),
    '65,536 input tokens', '64K input tokens'),
    '65,536 tokens', '64K tokens'),
    '64,000 tokens', '64K tokens'),
    '32,768 tokens', '32K tokens'),
    '32,000 tokens', '32K tokens'),
    '16,384 tokens', '16K tokens'),
    '8,192 input tokens', '8K input tokens'),
    '8,192 tokens', '8K tokens'),
    '4,096 tokens', '4K tokens'),
    '2,048 tokens', '2K tokens')::jsonb,
  updated_at = NOW()
WHERE comparison_data::text ~ '[0-9],[0-9]{3}[^\"]*tokens';

UPDATE content_items
SET
  blocks = replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(
    blocks::text,
    '1,050,000 tokens', '1.05M tokens'),
    '1,048,576 tokens', '1M tokens'),
    '1,000,000 tokens', '1M tokens'),
    '400,000 tokens', '400K tokens'),
    '256,000 tokens', '256K tokens'),
    '200,000 tokens', '200K tokens'),
    '131,072 input tokens', '128K input tokens'),
    '131,072 tokens', '128K tokens'),
    '128,000 tokens', '128K tokens'),
    '96,000 tokens', '96K tokens'),
    '65,536 input tokens', '64K input tokens'),
    '65,536 tokens', '64K tokens'),
    '64,000 tokens', '64K tokens'),
    '32,768 tokens', '32K tokens'),
    '32,000 tokens', '32K tokens'),
    '16,384 tokens', '16K tokens'),
    '8,192 input tokens', '8K input tokens'),
    '8,192 tokens', '8K tokens'),
    '4,096 tokens', '4K tokens'),
    '2,048 tokens', '2K tokens')::jsonb,
  body = replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(
    body,
    '1,050,000 tokens', '1.05M tokens'),
    '1,048,576 tokens', '1M tokens'),
    '1,000,000 tokens', '1M tokens'),
    '400,000 tokens', '400K tokens'),
    '256,000 tokens', '256K tokens'),
    '200,000 tokens', '200K tokens'),
    '131,072 input tokens', '128K input tokens'),
    '131,072 tokens', '128K tokens'),
    '128,000 tokens', '128K tokens'),
    '96,000 tokens', '96K tokens'),
    '65,536 input tokens', '64K input tokens'),
    '65,536 tokens', '64K tokens'),
    '64,000 tokens', '64K tokens'),
    '32,768 tokens', '32K tokens'),
    '32,000 tokens', '32K tokens'),
    '16,384 tokens', '16K tokens'),
    '8,192 input tokens', '8K input tokens'),
    '8,192 tokens', '8K tokens'),
    '4,096 tokens', '4K tokens'),
    '2,048 tokens', '2K tokens'),
  updated_at = NOW()
WHERE kind = 'comparison'
  AND (blocks::text ~ '[0-9],[0-9]{3}[^\"]*tokens' OR body ~ '[0-9],[0-9]{3}[^\n]*tokens');
