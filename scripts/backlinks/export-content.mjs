import { neon } from '@neondatabase/serverless';
import {mkdir,writeFile} from 'node:fs/promises';
const sql=neon(process.env.NEW_DATABASE_URL);
const books=await sql`SELECT slug,title,summary,metadata FROM content_items WHERE kind='book' AND status='published' ORDER BY sort_order`;
for(const book of books) book.chapters=await sql`SELECT slug,title,body,blocks,sources FROM content_items WHERE kind='chapter' AND parent_slug=${book.slug} AND status='published' ORDER BY sort_order,slug`;
const guides=await sql`SELECT slug,title,body,sources,metadata FROM content_items WHERE kind='guide' AND status='published' AND slug IN ('rag-fails-in-production','how-to-check-ai-benchmark-claims')`;
await mkdir('.firecrawl/backlinks',{recursive:true});
await writeFile('.firecrawl/backlinks/content.json',JSON.stringify({books,guides},null,2));
console.log(books.map(b=>({slug:b.slug,chapters:b.chapters.length,blocks:b.chapters.reduce((s,c)=>s+(c.blocks?.length??0),0),images:b.chapters.reduce((s,c)=>s+(c.body.match(/!\[/g)?.length??0),0)})));
