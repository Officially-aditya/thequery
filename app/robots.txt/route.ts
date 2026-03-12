import { NextResponse } from "next/server";

export function GET() {
  const robots = `User-agent: *
Allow: /
Disallow: /admin
Disallow: /api

User-agent: GPTBot
Allow: /

User-agent: ClaudeBot
Allow: /

User-agent: PerplexityBot
Allow: /

User-agent: Google-Extended
Allow: /

Sitemap: https://www.thequery.in/sitemap.xml
ai-generated: true
`;

  return new NextResponse(robots, {
    headers: {
      "Content-Type": "text/plain",
    },
  });
}
