-- Publish the PHP2Shell postmortem with its verified sources and related articles.
WITH article_body AS (
  SELECT $php2shell_body$On the morning of October 22, 2026, Cloudflare CEO Matthew Prince posted a count. Thousands of exploit attempts in the ten hours before PHP2Shell went public. Tens of thousands in the 45 minutes after. Hundreds of thousands within two hours. Millions by morning.

The count was Cloudflare’s, and it was not the only number worth remembering. Security firms also reported widespread scans and probes. What appears to be the first confirmed code execution in the wild came on October 24, against a decoy server, three days after the patches shipped.

The number that deserves to outlast the week is a different one. Wiz said the same model it used to find the flaws also helped it build the exploits, detect the attacks and validate the patch. One toolchain did the defender’s job and the attacker’s job. That is the lesson of PHP2Shell for the AI industry, and it matters more than the 9.2 severity scores or the shutdowns.

This is a postmortem, written after the first week. It covers what happened, what the attack numbers did and did not show, and why the capability to detect a flaw could also help weaponize it.

## What happened

PHP2Shell was the name for two vulnerabilities, CVE-2026-109618 and CVE-2026-109619, each scored 9.2 on the 10-point CVSS severity scale, which works like a hurricane category: one number that says how worried to be. They sat in the PHP engine itself, not in WordPress, Drupal or Joomla. Wiz said nearly every PHP release back to version 5.4, from 2012, was affected by at least one of them. It also estimated that PHP ran roughly 70% of websites. That usage estimate concerns sites whose server-side language can be identified; it does not measure how many sites were exposed. Even a bare Hello World page on an affected version could be vulnerable.

The impact was remote code execution, or RCE, with no login required. It meant a stranger could run commands on a server as though sitting at its keyboard, the difference between a stranger reading your shop window and one being handed the till and the stockroom keys. One mechanism was a memory bug called a use-after-free. When PHP hit a fatal error, especially running out of memory, it could discard a request’s memory but leave a pointer to it behind. In some cases, a later request handled by the same worker trusted the stale pointer. Picture a coat check that throws out a coat during a fire alarm but keeps the ticket on file. An attacker hangs a coat of their choosing on that hook, and the next person to present the old ticket gets handed it.

The fixes shipped on October 21, 2026, in PHP 8.5.12, 8.4.27, 8.3.36 and 8.2.35. The full sequence ran like this.

| Date (2026) | Event |
| --- | --- |
| July 30 | Deep research on the PHP memory problem begins after the PHP Foundation enrolls the project in OpenAI’s Patch the Planet program |
| August 19 to 24 | Wiz finds the first flaw with AI models, confirms code execution on Drupal and WordPress within two days, and reports to PHP on August 24 |
| September 21 | PHP acknowledges Wiz’s reports and begins a coordinated fix |
| October 17 | Amazon logs the earliest known exploitation attempt, four days before the patches |
| October 20 | The CVE IDs leak at about 15:00 UTC, and PHP publishes its pre-announcement at 19:30 UTC, pulling the fix date forward |
| October 20 to 21 | Calif mistakenly posts a proof-of-concept video, later deleted but already circulating |
| October 21 | Chainguard publishes patched images at 07:41 UTC; PHP releases the fixes at 14:00 UTC, a day ahead of the original plan |
| October 21 to 22 | Sites go offline across the internet; Cloudflare reports millions of attempts by morning |
| October 24 | Perplexity’s honeypots capture what appears to be the first confirmed in-the-wild code execution |

CERT-EU and the U.K.’s NCSC advised organizations to take systems offline where they could not patch at once. Sites including Pinboard, the GrapheneOS forum and OBS pulled parts of their services, and Laravel halted its Cloud deployments while it rebuilt on patched images.

## What the numbers showed

Prince’s figures were Cloudflare’s own counts across its own network, shared on X without underlying data, so they were one vendor’s measurement and not an audit. They also blended two different things. Amazon Threat Intelligence logged an exploitation attempt on October 17, four days before the patches, from a commercial anonymization network, and said it was unclear how the actor learned enough to try. That suggested someone had early information. The flood after the announcement was what publishing a vulnerability reliably produces.

Much of the observed activity was probing. A scan is rattling every door handle on a street: cheap, loud and uninformative about who got in. The Hacker News described the activity through October 25 as widespread scanning.

A working exploit faced technical obstacles. Datadog described two constraints: the attacker had to size a request so memory allocation failed at exactly the right moment, and had to defeat ASLR to find valid memory addresses. ASLR randomizes where a program sits in memory, like rearranging the furniture in a dark room every night so a burglar cannot walk it from memory. The bug chain included a memory disclosure that worked as the flashlight, which is why a working exploit was harder than a crash.

Then one worked. On October 24, Adel Karimi, who leads detection and response at Perplexity, recorded what appeared to be the first confirmed in-the-wild exploitation, against a honeypot running Apache with PHP 8.4.26. A honeypot is a decoy server left deliberately inviting so defenders can watch who tries the door. The captured commands used wget, curl or the PHP command line to fetch further payloads from a remote server, and no second-stage files were observed.

Not every early claim survived. Palo Alto Networks Unit 42 first reported a European government intrusion in which an attacker reached password hashes and multi-factor secrets within two hours and appeared to use an agentic AI. It then retracted the link to PHP2Shell, saying it could not confirm that the initial access involved these CVEs.

## The disclosure that leaked

Coordinated disclosure is the practice of privately telling the people who maintain and distribute software about a flaw, with a deadline, so fixes are ready before the public learns of it. It works like warning a restaurant’s suppliers about a gas leak before announcing it in the lobby. Every supplier who gets the memo is also one more person who can mention it.

PHP’s plan was a release on October 22 at 14:00 UTC, and the secrecy held until two days before. The PHP Foundation said Chainguard unilaterally broke the embargo by publishing patched images about six hours early, disregarding weeks of work by many parties. Chainguard CEO Dan Lorenc answered that he followed disclosure guidelines after finding evidence of active exploitation, and that he had independently found the flaw and built an exploit. Both accounts can be true at once. The Foundation described a rule. Lorenc described what he saw.

Analyst Frank Dickson put the structural problem in one line: “A patch is a treasure map.” Attackers can compare the old code with the new to find which door got a new lock. An embargo protects defenders only until someone else finds the same bug, and Amazon’s October 17 sighting predated every leak the Foundation described.

## One tool, two hats

Security people sort researchers by hat. A white hat finds a flaw and reports it so it gets fixed. A black hat finds the same flaw and uses it, or sells it. The terms are usually traced to old Western films, where the hero wore the white hat and the villain the black. The labels describe what a person does after finding the flaw, not how they found it.

A lockpick set is the same object in a locksmith’s hand and a burglar’s. It carries no hat of its own. PHP2Shell showed that frontier AI models had become a lockpick set that finds the lock, builds the pick and tests it against several brands of door. Wiz said it used Wiz Atlas, Gemini 4 Argon and GPT-5.6 Sol throughout, and credited Argon with contributions to finding the flaw, developing exploits, detecting attacks and validating the patch. Wiz also demonstrated an exploit script against the latest WordPress, Drupal and Joomla images, and said the models let it adapt working exploits across applications in hours.

The July wp2shell episode in WordPress had made the point earlier and more bluntly. Searchlight Cyber’s researcher found that flaw with GPT-5.6 Sol Ultra for about $25 in compute and roughly ten hours of work, and Searchlight noted that exploit brokers pay orders of magnitude more for such bugs. What made the researcher a white hat was a choice made after the model finished: he took the result through a disclosure process rather than a payday. Public exploit code appeared within hours of the patch, and Searchlight put weaponized exploitation at under two days.

## What a detector is without the label

The lasting lesson was that the same capabilities can support detection or attack, depending on how they are used. Three things from the first week after disclosure showed how.

A scanner that tests whether a server runs vulnerable PHP asks the same question whether the answer goes to a patch list or a target list. Xint released a scanner so hosts could check their own exposure. Defused Cyber’s honeypots caught requests carrying a dummy upload that checked for the same vulnerable upload-progress behavior with no payload attached, which Defused read as someone building a target list. Same question, different spreadsheet.

A detector can be built by making an exploit work in a lab. ReliaQuest reproduced CVE-2026-109618 in a controlled environment to find the sequence of errors defenders should watch for, and Wiz said its model helped with attack detection and patch validation. Testing a smoke detector can involve lighting a small fire. The work that reveals what defenders should watch for can also show an attacker how a flaw behaves.

The economics ran in one direction. Searchlight’s $25 figure is a vendor’s own number, but even a rough version of it shows how cheap discovery had become next to what such bugs fetch on the exploit market. Calif’s write-up, as TNW reported it, said it used generally available models and no special tooling, and that knowing PHP’s history mattered more than the model. If that is accurate, at least some of the capability did not require a special access program.

The honest limit is what attackers actually did with it. We found no public evidence in the first week that attackers used AI to build PHP2Shell exploits, and Unit 42’s report of AI-driven post-exploitation was withdrawn as to this flaw. Public reports described widespread scanning and circulating proof-of-concept code. One possible reason for the slower appearance of confirmed PHP2Shell code execution is the bug class: exploiting its memory corruption required overcoming ASLR, while Searchlight reported weaponized exploitation of the WordPress logic flaw within two days of its patch. That comparison is an inference. No source tests the explanation directly, and the difference in public reports does not establish how much exploitation occurred overall. The two cases show the dual-use potential; they do not establish why the observed timelines differed.

## What it meant for the AI industry

OpenAI framed the episode as a win. It said frontier AI found the flaws before attackers did, opening what it called a Defender’s Window, and it committed $1 billion in subsidized access to its Daybreak cyber program for under-resourced defenders. The record supported part of that. A roughly two-month head start existed, and Cloudflare switched on firewall rules for all customers about nine hours before disclosure. The record also qualified it. Amazon logged an attempt four days before the patches, and the embargo broke two days before its date.

Credit for the discovery was also shared among parties with something to sell. Google’s Wiz credited Google’s Gemini model, and OpenAI credited its own program. Wiz compared the flaws with Heartbleed and Log4Shell. None of that makes the find unimportant. It means every lab and security vendor now has a commercial reason to say its model found the big one first.

The structural point was about access. In April 2026, Anthropic withheld Claude Mythos from public release and initially offered it to a small group of organizations through Project Glasswing. OpenAI expanded its Trusted Access for Cyber program through verification, arguing that legitimate defenders should be able to apply. The distinction narrowed on October 6, when Anthropic expanded its Cyber Verification Program into three access tiers for qualifying security professionals, including access to Mythos 5.1, and folded Glasswing into the new program. Both companies now use verification and access controls, though their requirements differ. PHP2Shell tested the limits of that approach. Calif’s account suggests generally available models were enough for its research, which means access controls on a specialized model may not contain the broader capability. The disclosure process around the research, the part that actually separated defense from attack, also leaked two days before its original deadline.

Two consequences followed. Vulnerability discovery became a volume business: OpenAI said PHP2Shell was among several serious vulnerabilities it had found and expected “similar findings to increase sharply,” which, if borne out, would put more pressure on defenders’ patching capacity. And the PHP Foundation’s own history showed the cost of the old pace. A contractor saw the first signs of the memory problem in 2024, found no concrete vectors, and moved on to issues with clearer impact.

## The postmortem verdict

PHP2Shell was a defensive success in the steps that could be checked. The patch shipped, vendors had mitigations ready, and the first week produced millions of reported attempts alongside a small number of publicly confirmed cases of code execution. Public reporting cannot establish the total number of successful attacks. It was also a warning. Flaws that sat in PHP for more than a decade became easier for researchers to understand and exploit with model assistance, a capability that can serve attackers as well as defenders.

For anyone building or buying AI security tools, the offensive versus defensive label stopped being useful. The questions that matter are who holds the tool, what process surrounds it, and how long that process survives contact with the internet. PHP’s embargo ran for weeks and broke two days before its date.

The model never chose a hat. The process around it did.

**Previously on TheQuery:** [DayBreak is the Avengers Initiative for Cybersecurity. The Problem Is, AI Arms it Both Sides.](https://www.thequery.in/articles/defenders-better-tools-everyone-else-ai-cybersecurity) and [Anthropic Gave Its Dangerous Model to Defenders](https://www.thequery.in/articles/anthropic-project-glasswing-mythos-cybersecurity)$php2shell_body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'article:php2shell-postmortem-detector-was-the-weapon',
  'article',
  'php2shell-postmortem-detector-was-the-weapon',
  '',
  'articles/php2shell-postmortem-detector-was-the-weapon',
  'PHP2Shell Postmortem: The Detector Was the Weapon',
  'A week after PHP2Shell, public reports showed widespread scans and confirmed code execution on a decoy. AI helped researchers find the flaws, develop exploits, and validate fixes.',
  article_body.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', article_body.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'PHP2Shell: AI uncovers critical RCE vulnerabilities affecting 70% of the Internet - Wiz', 'url', 'https://www.wiz.io/blog/php2shell-php-rce'),
    jsonb_build_object('title', 'PHP2Shell exposes the weakness in coordinated vulnerability disclosure - CSO Online', 'url', 'https://www.csoonline.com/article/4239359/php2shell-exposes-the-weakness-in-coordinated-vulnerability-disclosure-2.html'),
    jsonb_build_object('title', 'PHP2Shell Exploitation Attempts Continue Amid Suspected N. Korea Global Scanning Activity - The Hacker News', 'url', 'https://thehackernews.com/2026/10/php2shell-exploitation-hits-government.html'),
    jsonb_build_object('title', 'wp2shell: Discovering One of 2026’s Biggest Zero-Days, and the Future of Exposure Management - Searchlight Cyber', 'url', 'https://www.slcyber.io/blog/wp2shell-discovering-one-of-2026s-biggest-zero-days-and-the-future-of-exposure-management'),
    jsonb_build_object('title', 'PHP rushes out emergency patches for two critical bugs AI found first - The Next Web', 'url', 'https://thenextweb.com/news/php2shell-php-rce-ai-openai-wiz'),
    jsonb_build_object('title', 'Expanding the Cyber Verification Program - Anthropic', 'url', 'https://www.anthropic.com/news/cyber-verification-program')
  ),
  '{}'::jsonb,
  NULL,
  NULL,
  'published',
  DATE '2026-10-28',
  0
FROM article_body
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path,
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  body = EXCLUDED.body,
  blocks = EXCLUDED.blocks,
  sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata,
  cover_image_url = EXCLUDED.cover_image_url,
  cover_image_alt = EXCLUDED.cover_image_alt,
  status = EXCLUDED.status,
  published_at = EXCLUDED.published_at,
  sort_order = EXCLUDED.sort_order,
  updated_at = NOW();
