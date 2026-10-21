# Why is REA making product based companies sweat

By Addy · October 22, 2026

REA, short for Reverse Engineer Anything, gives AI coding agents tools to inspect existing software, trace how a feature works, and help build a version of it. For a product company, the uncomfortable implication is straightforward: the application it spent years developing can also become a starting point for someone else's competing product. [REA's repository](https://github.com/morluto/rea) describes the tool and its supported targets.

That puts pressure on the idea of a software moat. If an agent can help reproduce the features customers are paying for, how much protection does owning the implementation still provide? Recreating a feature is not the same as replacing a complete commercial service, but a company whose advantage depends on that feature being difficult to build has a question to answer.

For software handling sensitive business data, security and compliance can give customers a reason to keep paying even after a competitor matches the features. They are paying for safeguards that work, evidence they can inspect, and an operator responsible for keeping the service running. Established companies have to earn that trust, and challengers can earn it too. Reproducing an application does not automatically settle who can be trusted to operate it.

## What REA actually demonstrates

Reverse engineering means working backward from a program to understand its construction. Think of a mechanic examining an engine without its workshop manual. REA helps an agent conduct that examination, using tools such as Ghidra and Hopper to inspect the instructions and relationships inside compiled software. The agent then uses the findings to explain behavior or write and test an implementation. It is a workflow for investigation and reconstruction, with work to verify along the way.

The project's [DX-Ball example](https://github.com/morluto/rea) makes the distinction concrete. Its repository reports reconstructing a function that converts a screen position into a left or right sound setting, matching the original function in 3,205 test cases and reproducing all 63 compiled bytes. These are project-reported results, not an independent evaluation of a finished commercial replacement.

Its [Notion example](https://github.com/morluto/rea) traces how the desktop application sends copied content to the computer's clipboard. That documents a specific path through the packaged application. It does not demonstrate a replacement for Notion's entire service.

These are useful achievements. Understanding software without its original source can involve substantial work. Making parts of that work easier puts pressure on companies whose advantage rests on a reproducible feature. The leap from that result to every software company losing its business remains a hypothesis.

## A working copy still needs an operator

There is a boundary between the application someone can inspect and the service a company operates. A desktop application or website can reveal substantial behavior while depending on systems and data elsewhere. Examining a shopfront does not give you the warehouse, stock records, or delivery operation behind it. A reconstruction may reproduce the customer's experience while needing a separate implementation of everything that supports it.

But suppose the reconstruction succeeds. Suppose an agent produces a payroll application with the right screens, calculations, approval flow, and exports. The security question survives that success.

Who can read employees' bank details? Can a manager approve their own payment? Does someone who has left the company retain access? What happens if a software update changes a calculation after payroll has been approved?

Those are hypothetical acceptance questions for the replacement, rather than findings about REA. They show why a buyer can accept that the software works and still decline to put employee records into it.

The distinction also limits the argument. A local utility processing disposable files presents a different decision from a service holding payroll records. Customers with little operational risk may reasonably choose a cheaper copy. Security and compliance become stronger selling points as the consequences of failure increase.

## Security lives in the controls people cannot see

A security control is a safeguard that prevents, detects, or helps correct something going wrong. Separate keys for separate filing cabinets are a physical version of access controls. A software equivalent should prevent a user from reading records they are not entitled to see, including when they bypass the intended screen.

Consider an application serving two unrelated companies. Its ordinary screens may look correct even if a mistake lets one company's user request the other's records. A convincing demonstration of the interface would not settle that question. A buyer needs evidence that the boundary holds under the conditions that matter.

An audit log serves a different purpose. Like a visitor book, it records who did what and when. Its usefulness depends on whether the relevant actions are recorded, whether the record can be altered, and whether someone can retrieve it during an investigation. Adding an activity page is only part of that work.

Security also continues after release. The practical test resembles a fire drill: can the people responsible detect a problem, contain it, communicate clearly, and restore service? A recovery procedure that nobody has exercised offers a buyer less assurance than one supported by evidence.

None of this requires assuming AI-written software is inherently insecure or that established vendors are inherently safe. Both need scrutiny. The commercial advantage belongs to the operator who can demonstrate the safeguards and keep them working as the product changes.

## Compliance is evidence, with a scope and a date

Security and compliance are related, but they answer different questions. Security concerns protection against harm. Compliance concerns meeting the requirements that apply to the organization, service, or contract. A buyer needs to understand which requirements an assurance document actually addresses.

SOC 2 is a useful example. It is an independent examination and report about specified controls at a service organization. A Type 2 report addresses their operation over a period. Think of an inspection report for a particular building: the building inspected, the dates, the findings, and the inspector's opinion all matter.

AICPA's own review checklist asks whether the report covers the relevant systems, processes, and locations. It asks whether the reporting period is sufficiently recent, whether the categories covered meet the buyer's needs, and whether exceptions affect the services being purchased. It also asks about testing the controls' operating effectiveness. [AICPA's SOC 2 report review checklist](https://assets.ctfassets.net/rb9cdnjh59cm/3xbcLlNc5rd72So4nQpNIk/7a3e8e5945c78c35fc5e116859b96e6a/SOC_2_Report_%C3%82_Review_Checklist.pdf) makes the evaluation more specific than recognizing a badge.

That specificity explains why copying an application does not transfer its provider's assurance. The new operator has its own systems, people, changes, and operating record to establish. Even an identical implementation would leave questions about how it is managed.

AI can help with the work. There is no reason to treat writing policies, implementing safeguards, or collecting records as immune to automation. But generating a policy does not establish that its procedures were followed. Producing an inspection template does not establish that the building passed inspection.

A compliance claim also cannot guarantee that a breach will never occur. Its value lies in relevant, credible evidence that helps a buyer assess the service and its operator.

## A secure cloud provider does not finish the job

The strongest practical objection is that a small team can rely on an established cloud provider for much of the infrastructure. It can. That reduces the work a challenger must take on and weakens any claim that only a large incumbent can operate safely.

AWS nevertheless describes security and compliance as shared responsibilities. AWS protects its underlying infrastructure, while customer responsibilities vary with the service selected. For an EC2 virtual machine, those responsibilities include the customer's operating system and application software. Managed services shift more work to AWS, while leaving responsibilities such as data management and permissions with the customer. [AWS's shared responsibility model](https://aws.amazon.com/compliance/shared-responsibility-model/) sets out that division.

The analogy is renting space in a guarded building. The owner can secure the premises while the tenant remains responsible for who receives access to its rooms. Buying good infrastructure helps. The application operator still has decisions to make and controls to run.

REA itself offers a relevant detail. Its repository says analysis happens on the user's machine, but the agent receives the results and its model provider has its own data policy. [REA's data-handling explanation](https://github.com/morluto/rea) describes that distinction. A team using the tool with sensitive material must consider where the findings go as well as where the original application sits.

That is the kind of question a company can pay a supplier to resolve and document. The value comes from a functioning arrangement for handling the data, with responsibilities understood across the services involved.

## The counterargument: assurance gets cheaper too

Security and compliance are not permanent protection from competition. A challenger can use managed services, implement appropriate controls, commission independent examinations, and build an operating record. If it supplies equivalent assurance at a lower total cost, the incumbent has a competitor with a stronger offer.

The same argument applies to AI. If agents reduce the work needed to implement safeguards, test them, and assemble evidence, they reduce part of the cost of competing on trust as well. An established vendor cannot assume that the administrative work surrounding its product will remain expensive for everyone else.

The more defensible advantage is accumulated evidence of competent operation. Customers can assess how the company handles changes, answers security questions, investigates failures, and supports recovery. Some of that evidence takes time to produce. All of it can lose value if the company stops performing.

This is why a certification alone is a weak answer to the moat question. A buyer needs assurance relevant to its actual use, and a rival can obtain assurance too. A moat exists only to the extent that the company delivers something competitors cannot quickly or economically match.

Security and compliance also belong alongside other advantages. Customer relationships, useful proprietary data, integration into business processes, and distribution can matter. The claim here is narrower: when a product handles sensitive work, a copied feature set leaves an important part of the buying decision unresolved.

For low-risk software, that unresolved part may be small. Product companies serving those markets have less room to use enterprise assurance as a reason their economics should survive unchanged.

## What a product company must still sell

The implication for founders is uncomfortable but practical. If the main reason a customer pays is that a feature took months to implement, tools that shorten reconstruction threaten that advantage. The time spent building it is a cost already incurred. It does not establish what a buyer should pay for the next year.

A security and compliance case has to connect to work the customer would otherwise own. Someone must review access, check changes, investigate suspicious behavior, restore lost records, and provide evidence when a buyer or auditor asks for it. A supplier that performs those jobs well can save the customer effort and reduce uncertainty.

Purchasing the service does not remove every responsibility from the buyer. It can provide a defined operator for agreed parts of the job, with commitments the customer can examine. Whether that arrangement is worth its price depends on the workload, the supplier's performance, and the alternatives.

That creates a demanding test for existing product companies. They need to explain their value after a competitor matches the visible features. Claims about trust should lead to evidence. Claims about compliance should identify the scope. Claims about reliability should survive a conversation about what happens during a failure.

REA does not prove that everything can be recreated using AI. It does make the assumption worth examining before a company's pricing depends on nobody trying. For sensitive business software, the next source of value may sit in the operation around the product as much as in the implementation itself.

The code can be copied. The responsibility still needs an owner.

*Sources:*

- [REA: Reverse Engineer Anything](https://github.com/morluto/rea) - Official project repository
- [SOC 2 report review checklist](https://assets.ctfassets.net/rb9cdnjh59cm/3xbcLlNc5rd72So4nQpNIk/7a3e8e5945c78c35fc5e116859b96e6a/SOC_2_Report_%C3%82_Review_Checklist.pdf) - AICPA
- [Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/) - Amazon Web Services

**Previously on TheQuery:** [Your SaaS Is Built. Now the Hard Part.](https://thequery.in/articles/saas-distribution-2026) and [DayBreak is the Avengers Initiative for Cybersecurity. The Problem Is, AI Arms it Both Sides.](https://thequery.in/articles/defenders-better-tools-everyone-else-ai-cybersecurity)
