-- Rebuild the Codex provider guide's blocks from its body so the
-- NOTE callout restyle (043) is reflected in the rendered output.
-- (043 updated body; blocks store quotes JSON-escaped so the match missed.)
UPDATE content_items
SET
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', body)
  ),
  updated_at = NOW()
WHERE kind = 'guide'
  AND slug = 'how-to-use-other-models-in-codex'
  AND parent_slug = '';
