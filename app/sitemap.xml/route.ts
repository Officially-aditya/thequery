
const BASE_URL = "https://www.thequery.in";

export const revalidate = false;

const SITEMAP_SECTIONS = [
  "pages",
  "books",
  "guides",
  "comparisons",
  "articles",
  "glossary",
];

export async function GET() {
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${SITEMAP_SECTIONS.map(
  (section) => `  <sitemap>
    <loc>${BASE_URL}/sitemap/${section}.xml</loc>
  </sitemap>`
).join("\n")}
</sitemapindex>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
    },
  });
}
