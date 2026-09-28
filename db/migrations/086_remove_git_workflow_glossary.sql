-- Remove off-topic git-workflow glossary pages (merge-queue, squash-merge).
--
-- These two pages describe generic Git/GitHub merge mechanics, not AI concepts.
-- They existed only to support inline links in the April 30 2026 article
-- 'vibe-coding-broke-github-that-is-not-the-surprising-part'. The article
-- keeps its explanation in prose; the two inline glossary links become plain
-- text. No other glossary relatedTerms point at these slugs (merge-queue
-- pointed at squash-merge, which is deleted with it; vibe-coding does not
-- point back), so the delete is clean.

UPDATE content_items
SET body = REPLACE(REPLACE(body,
  '[merge queue](/glossary/merge-queue)',
  'merge queue'),
  '[squash merges](/glossary/squash-merge)',
  'squash merges'),
  updated_at = NOW()
WHERE kind = 'article'
  AND parent_slug = ''
  AND slug = 'vibe-coding-broke-github-that-is-not-the-surprising-part';

UPDATE content_items
SET blocks = jsonb_set(blocks, '{0,content}', to_jsonb(body)),
    updated_at = NOW()
WHERE kind = 'article'
  AND parent_slug = ''
  AND slug = 'vibe-coding-broke-github-that-is-not-the-surprising-part'
  AND jsonb_typeof(blocks) = 'array';

DELETE FROM content_items
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug IN ('merge-queue', 'squash-merge');
