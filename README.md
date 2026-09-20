# TheQuery

<p align="center">
  <strong>Demystifying Artificial Intelligence from First Principles to Production Systems</strong>
</p>

<p align="center">
  <a href="https://thequery.in"><strong>thequery.in »</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js" alt="Next.js 16" />
  <img src="https://img.shields.io/badge/React-19-blue?style=flat-square&logo=react" alt="React 19" />
  <img src="https://img.shields.io/badge/TypeScript-5-blue?style=flat-square&logo=typescript" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-v4-38bdf8?style=flat-square&logo=tailwindcss" alt="Tailwind CSS v4" />
  <img src="https://img.shields.io/badge/Database-Neon_Postgres-00e599?style=flat-square&logo=postgresql" alt="Neon Postgres" />
</p>

---

## Overview

**TheQuery** is an open-access platform dedicated to AI knowledge, engineering principles, and systems architecture. Whether you are a curious learner, researcher, or production engineer, TheQuery provides structured, in-depth resources covering modern artificial intelligence—from foundational mathematics and machine learning algorithms to retrieval-augmented generation (RAG), knowledge graphs, and frontier LLM applications.

---

## Core Content Pillars

### 📚 Books & Master Courses
Comprehensive, structured multi-chapter courses designed to build deep mental models:
- **[AI: From First Principles](content/books/ai-from-first-principles/)**: Foundational mathematics, classical machine learning, neural architectures, transformers, and production realities.
- **[RAG + Knowledge Graph Master Course](content/books/rag-kg-master-course/)**: Vector spaces, embeddings, dense vs. sparse retrieval, hybrid architectures, knowledge graph engineering, and production evaluation.

### 🧭 Practical Guides
Problem-first, post-mortem-driven deep dives into real engineering bottlenecks:
- Failure-mode analysis (e.g. *Why RAG Fails in Production*, *Your Model Hit 99% Accuracy. Then You Shipped It.*).
- Pragmatic architectural guidance avoiding theoretical fluff.

### 📖 Living AI Glossary
A curated dictionary of 190+ terms across ML, LLM engineering, and retrieval algorithms:
- Cross-linked inline throughout articles and guides via automated markdown term detection.
- Definitions, mathematical formulations, and practical implementation context.

### 📰 Technical Articles & Research
Timely analyses and technical teardowns of:
- Frontier AI research papers and open-source models.
- AI developer tooling, agentic frameworks, and on-device machine learning.
- Industry shifts and technical infrastructure developments.

### ⚖️ Model Comparisons & Benchmarks
Side-by-side technical evaluations of leading frontier and open-weight models across standard benchmarks (MMLU, GPQA, SWE-bench, TerminalBench), latency, pricing, and context windows.

### 💡 AI Word of the Day
Daily micro-learning concepts introducing key terms and paradigms with concise technical definitions and context.

---

## Tech Stack & Architecture

- **Framework**: [Next.js 16](https://nextjs.org/) (App Router, Server Components, Route Handlers)
- **UI & Runtime**: [React 19](https://react.dev/), [TypeScript 5](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/) with `@tailwindcss/postcss`
- **Database**: Serverless PostgreSQL via [Neon](https://neon.tech/) (`@neondatabase/serverless`)
- **Content & Markdown**:
  - `react-markdown` with `remark-gfm` & `rehype-raw`
  - Mathematical typesetting via `remark-math` & `rehype-katex` (KaTeX)
  - Code syntax highlighting via `rehype-highlight`
  - Frontmatter parsing with `gray-matter`
- **Charts & Visualizations**: [Recharts](https://recharts.org/)
- **Analytics**: `@vercel/analytics`
- **Testing**: Node.js native test runner (`node --test`)

---

## Project Structure

```text
thequery/
├── app/                  # Next.js App Router (pages, layouts, API routes)
│   ├── articles/         # Article list & slug detail pages
│   ├── books/            # Book course reader & chapter views
│   ├── comparisons/      # AI model comparisons & benchmark views
│   ├── glossary/         # Glossary catalog & term pages
│   ├── guides/           # Technical engineering guides
│   ├── ai-word-of-the-day/ # Word of the day archive & views
│   ├── admin/            # Admin dashboard routes
│   └── sitemap.ts        # Dynamic multi-content sitemap generator
├── components/           # Reusable UI & content components
│   ├── MarkdownRenderer.tsx # Custom renderer with automated glossary linking
│   ├── BookCitation.tsx  # Citation & reference callout component
│   ├── ChapterSidebar.tsx# Book reading sidebar navigation
│   ├── GlossarySearch.tsx# Interactive glossary search
│   └── ReadingProgress.tsx # Reading progress tracker
├── content/              # Content repositories
│   └── books/            # Markdown chapter files & metadata for books
├── data/                 # Structured JSON data stores
│   ├── articles.json     # Articles & technical reports
│   ├── glossary.json     # Curated glossary terms & definitions
│   ├── guides.json       # Production guides
│   └── ai-word-of-the-day.json
├── lib/                  # Data access layers, content loaders, and utilities
├── scripts/              # Migration, database setup, and seed scripts
└── tests/                # Test suites using native node:test
```

---

## Getting Started

### Prerequisites
- **Node.js** >= 20.x
- **npm** >= 10.x (or `pnpm` / `yarn`)
- Neon PostgreSQL database instance (optional for local static content browsing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Officially-aditya/thequery.git
   cd thequery
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Configure your database connection and admin credentials in `.env`:
   ```env
   NEW_DATABASE_URL=postgresql://USER:PASSWORD@YOUR-NEON-HOST/neondb?sslmode=require
   ADMIN_USER=admin@example.com
   ADMIN_PASSWORD=replace-with-a-long-unique-password
   ```

4. **Database Setup & Migrations** *(if using database-backed features)*:
   ```bash
   npm run db:setup
   npm run db:migrate
   npm run db:seed
   ```

5. **Start the Development Server**:
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Available Scripts

| Command | Description |
| :--- | :--- |
| `npm run dev` | Starts the Next.js development server at `localhost:3000` |
| `npm run build` | Runs database migrations and builds the production bundle |
| `npm run start` | Starts the production server |
| `npm run lint` | Runs ESLint checks across the codebase |
| `npm test` | Runs the test suite using Node's native test runner |
| `npm run db:setup` | Initializes the database schema |
| `npm run db:migrate` | Runs database migration scripts |
| `npm run db:seed` | Seeds initial content into the database |
| `npm run db:copy-neon` | Utility script to copy/replicate Neon project data |

---

## License

This project is private and proprietary to TheQuery. All rights reserved.
