import fs from "fs";
import path from "path";

const dataPath = path.join(process.cwd(), "data", "articles.json");

export interface Article {
  title: string;
  slug: string;
  date: string;
  summary: string;
  content: string;
}

export function getAllIssues(): Article[] {
  const raw = fs.readFileSync(dataPath, "utf-8");
  const issues: Article[] = JSON.parse(raw);
  return issues.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getIssueBySlug(slug: string): Article | null {
  const issues = getAllIssues();
  return issues.find((i) => i.slug === slug) ?? null;
}

export function saveAllIssues(issues: Article[]): void {
  fs.writeFileSync(dataPath, JSON.stringify(issues, null, 2), "utf-8");
}
