import fs from "fs";
import path from "path";

const dataPath = path.join(process.cwd(), "data", "guides.json");

export interface Guide {
  title: string;
  slug: string;
  date: string;
  summary: string;
  content: string;
}

export function getAllGuides(): Guide[] {
  const raw = fs.readFileSync(dataPath, "utf-8");
  const guides: Guide[] = JSON.parse(raw);
  return guides.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getGuideBySlug(slug: string): Guide | null {
  const guides = getAllGuides();
  return guides.find((g) => g.slug === slug) ?? null;
}

export function saveAllGuides(guides: Guide[]): void {
  fs.writeFileSync(dataPath, JSON.stringify(guides, null, 2), "utf-8");
}
