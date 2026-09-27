-- Update Muse Spark 1.3 glossary AA Intelligence Index score 62 -> 48.
--
-- Verified 2026-09-27 against https://artificialanalysis.ai/models/muse-spark-1-3 :
-- - Intelligence Index v4.3.2 = 48 (was 62 at limited-preview write time)
-- - Rank #17 / 211 in class (was 6th / 636)
-- - Verbosity 170M output tokens, median ~88M (was 120M / ~72M)
-- - AA now lists text, image, video input + 1M context, $1.25/$4.25, 214 tok/s,
--   ~$1.60 per task (resolves the old text-only conflict note)
-- Body and blocks markdown are kept in sync because the glossary renderer
-- reads blocks. updated_at is bumped so "Last updated" reflects this edit.

UPDATE content_items
SET body = REPLACE(REPLACE(REPLACE(REPLACE(body,
  'Artificial Analysis lists Muse Spark 1.3 (max) at **62** on its Intelligence Index, ranking it 6th out of 636 models it tracks, ahead of every OpenAI model in its index and behind only Claude Fable 5.1 and Claude Opus 5. Artificial Analysis itself noted that this max-reasoning variant is in limited preview for Meta''s partners, distinct from the reasoning modes rolling out broadly today. On verbosity, the model used 120 million output tokens to complete Artificial Analysis''s Intelligence Index tasks, above the roughly 72 million median for models in its price class.',
  'Artificial Analysis lists Muse Spark 1.3 (max) at **48** on its Intelligence Index (v4.3.2), ranking it #17 out of 211 models in its class, well above the median of 26 for comparable models. On verbosity, the model generated 170 million output tokens across Intelligence Index tasks, well above the roughly 88 million median.'),
  '| Muse Spark 1.3 (max, limited preview) | September 2, 2026 | 62 |',
  '| Muse Spark 1.3 (max) | September 2, 2026 | 48 |'),
  'Second, Artificial Analysis''s own model card for Muse Spark 1.3 (max) lists text-only input support, which conflicts with Meta''s broader description of the Spark line as accepting text, images, video, audio, and PDF documents. That''s most likely a reflection of what Artificial Analysis specifically tested in this early preview rather than a real cut to multimodal support, but it hasn''t been resolved as of this writing.',
  'Second, Artificial Analysis now lists text, image, and video input support for Muse Spark 1.3 (max) with a 1M-token context window, matching Meta''s multimodal description, alongside an output speed of 214 tokens per second and a cost of about $1.60 per Intelligence Index task at $1.25/$4.25 per million input/output tokens.'),
  'Muse Spark 1.3 is Meta''s most credible frontier-tier release so far: independently placed at 62 on the Artificial Analysis Intelligence Index, ahead of every tracked OpenAI model and behind only Claude Fable 5.1 and Claude Opus 5. The catch is that this score, like Meta''s own launch comparison, describes a max-reasoning tier still in limited partner preview rather than the model generally available in Muse Code and the API today.',
  'Muse Spark 1.3 is Meta''s most credible frontier-tier release so far: independently placed at 48 on the Artificial Analysis Intelligence Index (v4.3.2, #17 of 211). The catch is that this score, like Meta''s own launch comparison, describes the max-reasoning tier rather than every reasoning mode available in Muse Code and the API today.'),
  updated_at = NOW()
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug = 'muse-spark-1-3';

UPDATE content_items
SET blocks = jsonb_set(blocks, '{0,content}', to_jsonb(body)),
    updated_at = NOW()
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug = 'muse-spark-1-3'
  AND jsonb_typeof(blocks) = 'array';
