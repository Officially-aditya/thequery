# TheQuery

**TheQuery** is a website dedicated to AI knowledge and principles — a resource for exploring the ideas, concepts, and ethical foundations that shape modern artificial intelligence.

Whether you're a curious beginner or a seasoned practitioner, TheQuery aims to make AI knowledge accessible, structured, and meaningful. From core machine learning concepts to the principles guiding responsible AI development, this is your go-to reference for understanding the world of AI.

---


## Editorial database and admin

The public site and `/admin` are backed by Neon Postgres. Copy `.env.example` to `.env`, add the Neon `DATABASE_URL`, and configure the single editorial account with `ADMIN_USER` and `ADMIN_PASSWORD`.

```bash
npm run db:setup
```

This applies the idempotent schema migration, imports the existing articles, guides, glossary terms, books, and chapters, and stores a salted password hash for the configured admin account. The plaintext credentials stay in environment variables and are never committed.

At `/admin`, the editorial workspace provides dedicated content editors for articles, guides, glossary entries, books, and chapters. Articles and guides use ordered Markdown, comparison-table, and chart blocks; sources are managed separately and rendered at the end of the public page.
