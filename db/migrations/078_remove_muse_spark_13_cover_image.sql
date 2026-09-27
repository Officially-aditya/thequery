-- Remove the cover image from the Muse Spark 1.3 glossary model card.
--
-- Migration 077 attached /muse-spark-13-cover.webp to this row. The artwork was
-- not wanted, so the row goes back to rendering without a cover like the other
-- glossary model cards.
--
-- cover_image_url and cover_image_alt are set to NULL rather than an empty
-- string so CoverImage's "no cover" check keeps working.
--
-- published_at is left as 077 set it (2026-09-02). That date is the model's
-- release date, not an artifact of the cover image, and the page needs it to
-- render a publish date. updated_at is deliberately untouched so the "Last
-- updated" line still reads 2026-09-03, the honest date of the last body edit.

UPDATE content_items
SET
  cover_image_url = NULL,
  cover_image_alt = NULL
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug = 'muse-spark-1-3';
