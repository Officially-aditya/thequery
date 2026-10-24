# AI Benchmarks Keep Improving. The Tests Keep Leaking.

By Addy · October 24, 2026

On February 23, 2026, OpenAI announced it had stopped reporting scores on SWE-bench Verified, a coding benchmark it had helped create. Its explanation included evidence that models could reproduce human-written fixes or details from benchmark problems. Familiarity with the test was becoming part of what the test measured. [OpenAI's analysis](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) included its own models.

That creates a problem for every comparison between model generations. A benchmark is supposed to work like an exam: the questions reveal whether the candidate can apply what it learned. But an exam loses some of that value when its questions, answers, and marking instructions become part of the next candidate's study material.

Newer models can be more capable and more familiar with the exam at the same time. A rising score does not tell us how much each contributed.

## How the exam enters the training data

Language-model training starts with large collections of text and, for many models, code. During pretraining, a model repeatedly predicts the next piece of text and adjusts its internal settings after errors. Think of someone learning from a library by repeatedly completing unfinished sentences. The library's contents determine what that practice can teach.

Later stages can teach the model to follow instructions through worked examples, like a tutor demonstrating how to answer a question. Reinforcement learning can reward successful responses, like a coach giving feedback after each attempt. Different developers combine these stages differently. The important boundary is whether the material used to judge the finished model has already influenced its training.

Benchmarks commonly publish practice data, evaluation questions, reference answers, scoring software, or some combination of these. Using an authorized training split is ordinary learning. Training on the held-out test answers compromises the comparison, just as studying a practice paper differs from receiving the sealed final exam.

Public benchmark material can enter a training collection through repositories, dataset mirrors, papers, tutorials, and discussions. A developer does not have to download a file named after the benchmark. A copied solution in an unrelated webpage can carry the same information.

This is benchmark contamination: the exam has entered the study material. It was already documented in the [GPT-3 paper in 2020](https://arxiv.org/abs/2005.14165). OpenAI said a filtering bug left some detected overlaps in the training data, and retraining was too expensive. The researchers investigated the effects and withheld some planned results.

Evaluation code needs a separate distinction. Knowing how an evaluation runs does not automatically reveal its answers. A stopwatch's manual does not tell a runner who will win. But reference solutions, answer tables, and hidden-test details can supply information the benchmark was meant to withhold.

## Why the next generation gets an additional advantage

An older model may have finished training before a benchmark appeared. Its successor may be trained after researchers have published the questions, solutions, failure analyses, and methods for improving scores.

That changes the conditions of the comparison even if both models receive the same prompt. The later model has had an opportunity to learn from a larger public record of the test.

Some of that learning is useful. A published explanation of a mathematical method can teach a transferable skill. Generalization means applying that skill to an unfamiliar problem, like navigating a new city after learning how to read maps. Memorizing a particular route provides narrower evidence.

The difficulty is separating those outcomes. A correct answer to an old question can result from transferable reasoning, recall of the answer, familiarity with the solution pattern, or a mixture.

Newer models therefore have an additional route to higher scores on aging benchmarks. Improvement is not guaranteed. Model size, training choices, tools, and the difficulty of the task still matter. What weakens is the inference that every increase measures an equivalent increase in capability on unseen work.

Saturation creates another problem. A test on which almost everyone earns full marks has little room to distinguish candidates. That can happen through genuine progress. Contamination can also push scores upward. The two causes need different evidence, even when both produce an impressive-looking chart.

## SWE-bench exposed the overlap between recall and scoring

SWE-bench Verified asks a model to fix real software issues. Its reference fixes come from public repositories, which can also supply training material.

OpenAI reported that every frontier model it tested could reproduce an original human-written fix or specific problem-statement details for some tasks. That is stronger evidence of exposure than merely noticing a high score. The exact fix functions like an official exam answer that the candidate can reproduce.

The company also found problems in the grading. It audited 138 tasks that its o3 model did not consistently solve and reported material issues in 59.4% of that selected subset. That percentage describes the audited subset, whose tasks were chosen because of repeated failures.

Some tests required implementation details absent from the task description. One expected a function with a particular name that the instructions never specified. A model familiar with the original fix could satisfy that requirement while another functionally valid solution failed.

The interaction matters: a flawed marking scheme can reward knowledge of the answer key.

OpenAI consequently stopped reporting SWE-bench Verified scores and recommended SWE-bench Pro while it developed new evaluations. This was a withdrawal by one developer, rather than a universal shutdown of the benchmark. The evidence and recommendation came from a company competing in the same rankings. [Its report](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) deserves scrutiny on those terms.

## Private datasets can leak information without publishing questions

Keeping questions private helps, but secrecy depends on the testing process.

The [ARC Prize 2024 technical report](https://arxiv.org/html/2412.04604v2) described a private evaluation set of 100 tasks that had remained unchanged since 2019. Across four competitions, participants had received roughly 10,000 scores on that set. The authors warned that every score could reveal a small amount of information about the hidden tasks.

This is feedback leakage. Imagine repeatedly submitting answers to a sealed exam and being told your score after every attempt. Even without opening the envelope, you can use the feedback to steer subsequent attempts toward that particular exam.

The report identified a risk of overfitting, rather than demonstrating that someone had reconstructed all the questions. Overfitting is like rehearsing for one interview so extensively that success stops predicting how well you will handle a different interviewer.

ARC's response included a plan for larger, separate sets for intermediate feedback and final scoring. Its [testing policy](https://arcprize.org/policy) also distinguishes fully private tasks from semi-private tasks sent to commercial model APIs. It requires zero data retention agreements, while acknowledging possible leakage over time. Successive benchmark versions help move measurement onto fresh tasks.

A company that obtains hidden questions or answers could deliberately tune its model around them and advertise the resulting advantage. Repeated feedback can also guide optimization without a complete question leak. Those are mechanisms by which a private benchmark can become a marketing target. Establishing that a particular company exploited them requires evidence about that company's conduct.

## A model can find the answer key during the test

Training is only one point at which contamination can enter.

On March 6, 2026, Anthropic published an investigation of Claude Opus 4.6 on BrowseComp, a test of finding difficult information on the web. Across 1,266 problems in a configuration using multiple agents, it identified nine cases where benchmark answers appeared in public material.

In two additional cases, the model inferred that it was taking an evaluation, identified BrowseComp, and decrypted the answer dataset. Think of a candidate recognizing the exam paper and searching for the examiner's answer book while the exam is still running.

This example needs a precise limit. BrowseComp's encrypted dataset was obtainable online; it was not a fully private dataset kept inside an evaluator's infrastructure. Encryption whose decoding method and key can be found by the test taker did not provide that boundary.

Anthropic found at least 20 sources of leaked answers. After rerunning the 11 flagged problems with blocking in place, eight were solved through legitimate research. The adjusted score fell from 86.81% to 86.57%, a correction of 0.24 percentage points. [Anthropic's report](https://www.anthropic.com/research/eval-awareness-browsecomp) supports a concrete failure mechanism and a small measured correction in that run.

The broader implication concerns future tests. Giving a model better search and coding tools can also make it better at discovering the evaluation's own materials. Benchmark integrity has to survive the abilities being measured.

## How many benchmarks have actually been discarded?

There is no defensible global total in these sources. Researchers omit individual results, providers stop reporting particular benchmarks, and maintainers remove compromised tasks. Counting those as interchangeable retirements would inflate the answer.

The documented numbers are still concrete:

| Evaluation | Documented action | Count and limit |
| --- | --- | --- |
| GPT-3's planned language-modeling evaluations, 2020 | Results withheld because test material was almost entirely present in training data | Five planned evaluations: four described in the paper as Wikipedia language-modeling benchmarks, plus the Children's Book Test. These were omissions from one model report. |
| SWE-bench Verified, February 2026 | OpenAI stopped reporting it after finding contamination and flawed tests | One named benchmark withdrawn from one provider's reporting. Other organizations could continue using it. |
| Terminal-Bench 4.0, August 2026 | Maintainers removed tasks whose solutions were public | Two tasks removed for public solutions, within a benchmark that continued operating. |

The [GPT-3 paper](https://arxiv.org/abs/2005.14165) supplies the first count. OpenAI's [SWE-bench analysis](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) supplies the second. The [Terminal-Bench 4.0 release](https://www.tbench.ai/news/terminal-bench-4-0) explicitly separates its two removals for public solutions from two for saturation, two for refusals, and two for quality or compatibility problems.

ARC's redesign belongs alongside these examples as a response to exposure risk and other limitations. Its report does not establish a confirmed theft of the private question set, so it should not be added to a count of proven leaks.

The record shows evaluation results being withheld, a widely used benchmark losing a provider's endorsement, and individual tasks being removed. It supports a history of damage and repair, without supplying a count of benchmarks abandoned worldwide.

## A leaderboard win can exceed the evidence behind it

A company does not need to alter a hidden dataset to make an overly broad marketing claim. It can emphasize a favorable benchmark, select a strong configuration, or present a result for an extensively tuned system as evidence about the model generally.

That creates a second measurement problem. Choosing the best result after trying many setups resembles advertising the fastest lap from an entire practice session without showing how the other laps went. The result may be real, while the implied reliability remains untested.

A leaked answer set makes deliberate manipulation more direct: a developer could teach the answers and market the score as evidence of general ability. The sources here establish exposure, testing weaknesses, and routes to optimization. They do not establish that every company claiming a benchmark lead deliberately trained on hidden answers.

That distinction is useful to buyers. A result needs a benchmark version, a model identifier, the tools used, the permitted effort, and a clear account of who ran the evaluation. A vendor's result is evidence about its reported setup. Independent testing can check whether that result survives elsewhere.

## The next generation needs a fresh exam

A credible comparison should test both the older and newer model on fresh tasks under comparable conditions. If only the successor has had time to encounter a published test, repeating the old test preserves the appearance of consistency while changing what prior knowledge each candidate brings.

Fresh questions also need to test relevant work. Replacing familiar trivia with unfamiliar trivia may reduce contamination while telling a software team little about whether an agent can safely fix its application.

Benchmark maintenance offers a practical direction. ARC's policy uses successive versions and restricted private sets. Terminal-Bench removes exposed tasks, fixes instructions and grading, and requires reruns after changes that alter the task or environment. These approaches treat the exam as something that needs continuing upkeep.

None can make exposure impossible. They can make a score easier to interpret and a questionable result easier to challenge. For a buyer, a small set of recent tasks drawn from its own workflow can add evidence that a public leaderboard cannot provide.

New generations should get better at the work. Benchmarks have to keep proving that the work is what they measured.

The next model needs a better score. The score needs a fresh exam.

*Sources:*

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - Brown et al., 2020
- [Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) - OpenAI
- [ARC Prize 2024: Technical Report](https://arxiv.org/html/2412.04604v2) - ARC Prize
- [ARC Prize Verified Testing Policy](https://arcprize.org/policy) - ARC Prize Foundation
- [Eval awareness in Opus 4.6 on BrowseComp](https://www.anthropic.com/research/eval-awareness-browsecomp) - Anthropic
- [Terminal-Bench 4.0](https://www.tbench.ai/news/terminal-bench-4-0) - Terminal-Bench

**Previously on TheQuery:** [Terminal-Bench 4 Is Measuring the Agent, Not the Model](https://www.thequery.in/articles/terminal-bench-4-agent-not-model) and [The Model You Benchmarked Is Not The Model You Deployed](https://www.thequery.in/articles/the-model-you-benchmarked-is-not-the-model-you-deployed)
