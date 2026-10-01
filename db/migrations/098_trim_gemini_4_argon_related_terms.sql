-- Trim the Gemini 4 Argon relatedTerms array written by 096 from 28 entries to
-- 5. The oversized list came from every concept mentioned in the body; the
-- Related Terms block on a term page is a navigation aid, not an index, and
-- the wider concept set is already reachable through inline glossary links.
--
-- Kept: the family (gemini), the maker (google-deepmind), and the three
-- rivals Google's own launch table compares against (claude-opus-5-5,
-- claude-sonnet-5-5, gpt-6-astra).

UPDATE content_items
SET
  metadata = jsonb_set(
    metadata,
    '{relatedTerms}',
    '["gemini", "google-deepmind", "claude-opus-5-5", "claude-sonnet-5-5", "gpt-6-astra"]'::jsonb,
    true
  ),
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug = 'gemini-4-argon'
  AND parent_slug = '';