import { getResearchData } from "@/lib/research-data";
import { benchmarkCsv } from "@/lib/research-csv";

export const revalidate = 300;

export async function GET(_request: Request, { params }: { params: Promise<{ file: string }> }) {
  const { file } = await params;
  if (file !== "models.json" && file !== "benchmarks.csv") return new Response("Not found", { status: 404 });
  const data = await getResearchData();
  const isJson = file === "models.json";
  return new Response(isJson ? JSON.stringify(data, null, 2) + "\n" : benchmarkCsv(data.benchmarks), {
    headers: {
      "Content-Type": isJson ? "application/json; charset=utf-8" : "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="thequery-${file}"`,
      "Cache-Control": "public, s-maxage=300, stale-while-revalidate=600",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
