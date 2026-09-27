-- Fill the empty Muse Spark 1.2 cells in the Muse Spark 1.3 glossary benchmark table.
--
-- The coding table on /glossary/muse-spark-1-3 showed "not reported" in the 1.2
-- column on two rows, even though both values are already normalized in
-- model_benchmarks:
-- - SWE-Atlas Codebase QnA 1.2 = 46.2% at xhigh, from Meta's 1.3 launch
--   scorecard (id meta-20260902-muse12-swe-atlas). Same harness as the 1.3 cell
--   (59.4 at max), so directly comparable under the table's max/xhigh header.
-- - Terminal-Bench 2.1 1.2 = 82.9% (id gdm-20260813-muse12-terminal21, Google
--   DeepMind model-card comparison). Meta's 1.3 chart never published a 1.2
--   Terminal-Bench figure, so the cell carries the dagger mark. The paragraph
--   below the table already explains that 82.9 comes from 1.2's own August
--   chart at a different setting and is not directly comparable to the 88.8.
--
-- The DeepSWE v1.1 1.2 cell (55.0) is already filled and is left as-is.
-- Body and blocks markdown are kept in sync because the glossary renderer
-- reads blocks. updated_at is bumped so "Last updated" reflects this edit.

UPDATE content_items
SET body = REPLACE(REPLACE(body,
  '| SWEAtlas CodeBase QnA | 59.4 | not reported | 53.5 | 52.7 |',
  '| SWEAtlas CodeBase QnA | 59.4 | 46.2 | 53.5 | 52.7 |'),
  '| Terminal-Bench 2.1 | 88.8 | not reported | 88.8 (tie) | 86.7 |',
  '| Terminal-Bench 2.1 | 88.8 | 82.9† | 88.8 (tie) | 86.7 |')
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
