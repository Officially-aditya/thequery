-- Remove the opening analogy line from the Muse Spark 1.3 glossary model card.
--
-- The body opened with a "senior engineer ... oversized desk" analogy paragraph.
-- It is removed here; the card now opens directly with the release paragraph.
-- Body and blocks markdown are kept in sync because the glossary renderer
-- reads blocks. updated_at is bumped so "Last updated" reflects this edit.

UPDATE content_items
SET body = REPLACE(body,
  'If Muse Spark 1.2 was a senior engineer who could hold an entire project in view from an oversized desk, Muse Spark 1.3 is that same engineer a few weeks further into the job: making fewer unnecessary trips back and forth, wasting less material, and now pausing to check in before touching anything that cannot be undone.' || chr(10) || chr(10),
  '')
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
