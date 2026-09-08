import Link from "next/link";
import type { Metadata } from "next";
import { getAllBooks } from "@/lib/books";
import { SITE_URL } from "@/lib/site";

export const revalidate = 300;
export const metadata: Metadata = {
  title: "Research and data",
  description: "TheQuery's AI learning resources, model data, methodology, and citation guidance.",
  alternates: { canonical: `${SITE_URL}/research` },
};

export default async function ResearchPage() {
  const books = await getAllBooks();
  return <div className="max-w-[720px] mx-auto px-4 py-12 space-y-8 text-text-secondary leading-relaxed">
    <h1 className="font-serif text-3xl font-bold text-text-primary">Research and data</h1>
    <p>TheQuery publishes AI education and analysis for developers. This page brings together the resources you can read, reference, and use to check model claims.</p>
    <section className="space-y-3">
      <h2 className="font-serif text-xl font-semibold text-text-primary">AI model catalog</h2>
      <p>Explore model specifications and reported benchmark results in our <Link href="/comparisons" className="text-accent underline">model comparisons</Link>. Scores come from the sources identified alongside the data. They are not independent experiments conducted by TheQuery.</p>
      <p><a href="/research/data/models.json" className="text-accent underline">Download model data (JSON)</a> · <a href="/research/data/benchmarks.csv" className="text-accent underline">Download benchmark observations (CSV)</a></p>
      <p><a href="https://github.com/Officially-aditya/thequery-ai-data" className="text-accent underline">Versioned snapshots and citation metadata on GitHub</a></p>
      <p>The downloads preserve evaluator, source, benchmark version, harness, tools, reasoning effort, and evaluation date where recorded. Missing values mean unknown, not zero or disabled.</p>
    </section>
    <section id="methodology" className="space-y-3">
      <h2 className="font-serif text-xl font-semibold text-text-primary">Methodology and limitations</h2>
      <p>The catalog compiles published model specifications and evaluation claims. Multiple observations can exist for one model and benchmark. A shared benchmark name does not establish comparable conditions: check the version, harness, tool access, evaluator, and reasoning budget before comparing scores.</p>
      <p>Export timestamps identify when a snapshot was retrieved, not when a result was independently reproduced. Source links provide provenance, not a guarantee that the source remains available or that a vendor claim is correct. Coverage is selective and does not represent every model or evaluation.</p>
      <p>Read <Link href="/guides/how-to-check-ai-benchmark-claims" className="text-accent underline">How to Audit AI Benchmark Claims: A Practical Guide</Link> for the checks to apply before drawing conclusions.</p>
    </section>
    <section className="space-y-3">
      <h2 className="font-serif text-xl font-semibold text-text-primary">Free technical books</h2>
      <ul className="space-y-4">{books.map(book => <li key={book.slug}>
        <Link href={`/books/${book.slug}`} className="text-accent underline">{book.title}</Link>
        <p className="text-sm">{book.description}</p>
      </li>)}</ul>
      <p>Both books can be read without an account. Each landing page includes a suggested citation and BibTeX.</p>
    </section>
    <section id="citation" className="space-y-3">
      <h2 className="font-serif text-xl font-semibold text-text-primary">Citing and correcting the data</h2>
      <p>Cite TheQuery, “AI model catalog,” {`${SITE_URL}/research`}, and your retrieval date. Also cite the original evaluator and source for any individual score. Keep the downloaded snapshot so readers can reproduce which data you used.</p>
      <p>Source materials retain their original rights. No new license for third-party material is implied. For corrections, send the model, benchmark, source URL, and proposed correction to <a href="mailto:addy@thequery.in" className="text-accent underline">addy@thequery.in</a>.</p>
      <Link href="/press" className="text-accent underline">Information for journalists</Link>
    </section>
  </div>;
}
