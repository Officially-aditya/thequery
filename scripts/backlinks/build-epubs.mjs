import { readFile, mkdir, writeFile } from 'node:fs/promises';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { execFileSync } from 'node:child_process';
const {books}=JSON.parse(await readFile('.firecrawl/backlinks/content.json','utf8'));
const xml=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const modified=new Date().toISOString().replace(/\.\d+Z$/,'Z');
await mkdir('public/books/downloads',{recursive:true});
for(const book of books){
 const dir=`.firecrawl/backlinks/epub-${book.slug}`;
 await mkdir(`${dir}/META-INF`,{recursive:true}); await mkdir(`${dir}/OEBPS`,{recursive:true});
 await writeFile(`${dir}/mimetype`,'application/epub+zip');
 await writeFile(`${dir}/META-INF/container.xml`,'<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>');
 const wrap=(title,body)=>`<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en"><head><title>${xml(title)}</title><style>body{font-family:serif;line-height:1.55;margin:5%}pre{white-space:pre-wrap;font-size:.8em;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:.85em}td,th{border:1px solid #aaa;padding:.3em}h1,h2,h3{line-height:1.2}a{color:#2454a4}</style></head><body>${body}</body></html>`;
 const files=[];
 await writeFile(`${dir}/OEBPS/title.xhtml`,wrap(book.title,`<h1>${xml(book.title)}</h1><p>By ${xml(book.metadata.author)}</p><p>TheQuery</p><p>Offline export generated ${modified.slice(0,10)}. Content edition: ${xml(book.metadata.lastModified)}.</p><p>Copyright remains with the author. Free to read; no account required.</p><p>Online edition: <a href="https://www.thequery.in/books/${book.slug}">https://www.thequery.in/books/${book.slug}</a></p>`));
 for(const [i,ch] of book.chapters.entries()){
  const body=renderToStaticMarkup(React.createElement(ReactMarkdown,{remarkPlugins:[remarkGfm,remarkMath],rehypePlugins:[[rehypeKatex,{output:'mathml'}]],components:{a:({href,children})=>React.createElement('a',{href:href?.startsWith('/')?`https://www.thequery.in${href}`:href},children)}},ch.body));
  const file=`chapter-${i+1}.xhtml`; files.push({file,title:ch.title,math:body.includes('<math')});
  await writeFile(`${dir}/OEBPS/${file}`,wrap(ch.title,body));
 }
 await writeFile(`${dir}/OEBPS/nav.xhtml`,wrap('Contents',`<nav epub:type="toc" id="toc"><h1>Contents</h1><ol><li><a href="title.xhtml">Title page</a></li>${files.map(f=>`<li><a href="${f.file}">${xml(f.title)}</a></li>`).join('')}</ol></nav>`));
 await writeFile(`${dir}/OEBPS/content.opf`,`<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="book-id">https://www.thequery.in/books/${book.slug}</dc:identifier><dc:title>${xml(book.title)}</dc:title><dc:creator>${xml(book.metadata.author)}</dc:creator><dc:language>en</dc:language><dc:publisher>TheQuery</dc:publisher><dc:rights>Copyright the author. All rights reserved.</dc:rights><meta property="dcterms:modified">${modified}</meta></metadata><manifest><item id="title" href="title.xhtml" media-type="application/xhtml+xml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>${files.map((f,i)=>`<item id="ch${i}" href="${f.file}" media-type="application/xhtml+xml"${f.math?' properties="mathml"':''}/>`).join('')}</manifest><spine><itemref idref="title"/><itemref idref="nav"/>${files.map((f,i)=>`<itemref idref="ch${i}"/>`).join('')}</spine></package>`);
 execFileSync('python3',['-c',`import zipfile,pathlib,xml.etree.ElementTree as ET
p=pathlib.Path(${JSON.stringify(dir)})
for f in p.rglob('*'):
 if f.suffix in ['.xhtml','.opf','.xml']: ET.parse(f)
with zipfile.ZipFile(${JSON.stringify(`public/books/downloads/${book.slug}.epub`)},'w') as z:
 z.write(p/'mimetype','mimetype',compress_type=zipfile.ZIP_STORED)
 for f in sorted(p.rglob('*')):
  if f.is_file() and f.name!='mimetype':z.write(f,str(f.relative_to(p)),compress_type=zipfile.ZIP_DEFLATED)
`]);
 console.log(`Built ${book.slug}: ${files.length} chapters`);
}
