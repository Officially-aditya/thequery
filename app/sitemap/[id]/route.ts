import getSitemapEntries from "../../sitemap.ts";

export const revalidate = false;

export function generateStaticParams() {
  return [
    { id: "pages.xml" },
    { id: "books.xml" },
    { id: "guides.xml" },
    { id: "comparisons.xml" },
    { id: "articles.xml" },
    { id: "glossary.xml" },
  ];
}

interface Props {
  params: Promise<{ id: string }>;
}

export async function GET(_req: Request, props: Props) {
  const { id } = await props.params;
  const section = id.replace(/\.xml$/, "");

  const allEntries = await getSitemapEntries();

  let filtered = allEntries;
  if (section === "books") {
    filtered = allEntries.filter((e) => e.url.startsWith("https://www.thequery.in/books"));
  } else if (section === "guides") {
    filtered = allEntries.filter((e) => e.url.startsWith("https://www.thequery.in/guides"));
  } else if (section === "articles") {
    filtered = allEntries.filter((e) => e.url.startsWith("https://www.thequery.in/articles"));
  } else if (section === "comparisons") {
    filtered = allEntries.filter((e) => e.url.startsWith("https://www.thequery.in/comparisons"));
  } else if (section === "glossary") {
    filtered = allEntries.filter((e) => e.url.startsWith("https://www.thequery.in/glossary"));
  } else if (section === "pages") {
    filtered = allEntries.filter(
      (e) =>
        !e.url.startsWith("https://www.thequery.in/books/") &&
        !e.url.startsWith("https://www.thequery.in/guides/") &&
        !e.url.startsWith("https://www.thequery.in/articles/") &&
        !e.url.startsWith("https://www.thequery.in/comparisons/") &&
        !e.url.startsWith("https://www.thequery.in/glossary/")
    );
  } else {
    return new Response("Not Found", { status: 404 });
  }

  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${filtered
    .map(
      (e) =>
        `  <url>\n    <loc>${e.url}</loc>${
          e.lastModified
            ? `\n    <lastmod>${new Date(e.lastModified).toISOString()}</lastmod>`
            : ""
        }\n  </url>`
    )
    .join("\n")}\n</urlset>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
    },
  });
}
