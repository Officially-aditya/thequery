"use client";

import { useState } from "react";

export default function BookCitation({ citation, bibtex }: { citation: string; bibtex: string }) {
  const [message, setMessage] = useState("");
  async function copy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setMessage("Copied to clipboard.");
    } catch {
      setMessage("Copy unavailable. Select and copy the text below.");
    }
  }
  return (
    <section id="cite" className="mt-10 space-y-4 border-t border-border pt-6">
      <h2 className="font-serif text-xl font-semibold">Cite this book</h2>
      <p className="text-sm text-text-secondary break-words">{citation}</p>
      <button className="text-sm text-accent underline" onClick={() => copy(citation)}>Copy citation</button>
      <details>
        <summary className="cursor-pointer text-sm text-accent">BibTeX</summary>
        <pre className="my-3 overflow-x-auto rounded-lg bg-bg-secondary p-4 text-xs"><code>{bibtex}</code></pre>
        <button className="text-sm text-accent underline" onClick={() => copy(bibtex)}>Copy BibTeX</button>
      </details>
      <p role="status" className="text-sm text-text-muted">{message}</p>
    </section>
  );
}
