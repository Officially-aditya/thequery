import fs from "fs";
import path from "path";

const dataPath = path.join(process.cwd(), "data", "digest.json");

export interface DigestIssue {
  title: string;
  slug: string;
  date: string;
  summary: string;
  content: string;
}

export function getAllIssues(): DigestIssue[] {
  const raw = fs.readFileSync(dataPath, "utf-8");
  const issues: DigestIssue[] = JSON.parse(raw);
  return issues.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getIssueBySlug(slug: string): DigestIssue | null {
  const issues = getAllIssues();
  return issues.find((i) => i.slug === slug) ?? null;
}

export function saveAllIssues(issues: DigestIssue[]): void {
  fs.writeFileSync(dataPath, JSON.stringify(issues, null, 2), "utf-8");
}
