-- Attach the generated cover image to the Muse Spark 1.3 glossary model card.
--
-- The artwork was produced for this page in an earlier session and committed to
-- public/ as muse-spark-13-cover.webp, but the glossary row was never wired to
-- it. The file shipped without a single reference, and the page rendered with no
-- cover while the other four covers in public/ were in use.
--
-- cover_image_url is stored as a root-relative path, matching the convention the
-- article covers already use, because CoverImage passes the value straight to an
-- img src. The alt text describes the artwork rather than restating the page
-- title.
--
-- published_at was null on this row, which left the date logic in lib/glossary.ts
-- with nothing but updated_at to fall back on. It is filled here with the date
-- Meta released the model, 2026-09-02.
--
-- updated_at is deliberately left at its existing value. lastUpdatedDate prefers
-- updatedAt over publishedAt, so touching it would move the "Last updated" line
-- on the page to the date of this migration and imply the model card text had
-- been revised when only an image was added. 2026-09-03 is the honest date for
-- the last edit to the body.

UPDATE content_items
SET
  cover_image_url = '/muse-spark-13-cover.webp',
  cover_image_alt = 'Abstract blue and violet gradient field for the Muse Spark 1.3 model card',
  published_at = COALESCE(published_at, DATE '2026-09-02')
WHERE kind = 'glossary'
  AND parent_slug = ''
  AND slug = 'muse-spark-1-3';
