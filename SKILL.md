---
name: literature-tracker
description: Agent-driven personal research literature tracking for searching arXiv papers, reading them with token-aware fallbacks, generating Chinese summaries and structured enrichment JSON, extracting LaTeX formulas, producing Markdown reports, answering questions over the local paper library, and matching methods to a research codebase. Use when the user asks what papers are new, asks to explain an arXiv paper, searches the local literature library, compares methods, or asks whether a paper can transfer to their project.
---

# Literature Tracker

Use the repository CLI as the execution backend. Keep the conversation as the user interface; do not ask the user to navigate output folders unless they request the files.

## Route the request

- Weekly or broad discovery: run `python src/run.py`.
- Single arXiv paper: run `python src/run.py --paper-id <arxiv_id>`.
- Bottleneck-focused paper review: add a local `config/research_context.yaml`, then run `python src/run.py --paper-id <arxiv_id> --research-context config/research_context.yaml`.
- Audit existing papers without model calls: run `python src/run.py --quality-audit --research-context config/research_context.yaml`.
- Search/read without model calls: add `--no-llm`.
- Skip the arXiv LaTeX formula stage when speed matters: add `--skip-formulas`.
- Local literature question: run `python src/run.py --qa-question "..."`.
- Interactive local-library questions: run `python src/run.py --qa`.
- Paper-to-project transfer analysis: run `python src/run.py --project-path <path> --project-only`.
- Multi-step Agent task: run `python src/run.py --agent-question "..."`; use `--agent-session` for continuing memory. Multi-paper comparison is routed to the guarded `compare_papers` tool.
- Offline planner evaluation: run `python src/run.py --agent-eval`.
- Historical Agent trace audit: run `python src/run.py --agent-run-audit`.
- Rebuild only the ranked report from existing artifacts: run `python src/run.py --skip-search --no-llm --skip-formulas --max-read 0`.

Choose the narrowest route that satisfies the request. Do not run the full weekly pipeline for a single-paper explanation or a local-library question.

## Artifact contract

- `data/candidates_*.json`: search candidates and scoring metadata.
- `data/papers/<id>.json`: internal DeepXiv reading artifacts; this is the source for summarization.
- `output/papers/<id>.summary.md`: Chinese technical explanation.
- `output/papers/<id>.enrichment.json`: structured method, modules, formulas, transferability, and field relevance.
- `output/papers/<id>.evidence.json`: local quality tier, missing evidence, transferability score, provenance, and evidence card.
- `output/papers/<id>.formulas.json`: structured LaTeX formulas with labels and context.
- `output/agent_runs/*.json`: Agent plan/dependencies, policy events, tool calls, token usage, termination reason, source IDs, and reflection checks.
- `output/agent_memory/*.json`: compact session memory; it does not contain API keys or full paper text.
- `output/reports/*.md`: field reports, ranked reports, and optional synthesis.
- `output/index.json`: global machine-readable index; consult it before scanning all artifacts.

## Incremental and failure rules

- Respect `seen_ids.json`; it means the paper was successfully read and persisted.
- Do not treat `metadata_only` or `failed` as a completed read.
- Reuse existing summary, enrichment, and formula artifacts unless the user asks to regenerate them.
- Report missing full text or failed external lookups explicitly, and continue with available metadata when possible.
- Do not expose API keys, local `.env` contents, or private project code in the response.
- Treat quality scores as triage, not proof of scientific correctness. Separate paper facts, agent inference, and hypotheses.
- Treat Agent reflection scores as traceability signals, not factual verification. Every strong conclusion should point back to a tool result or be marked as inference.
- Treat plan dependencies, tool allowlists, duplicate-call blocks, and total output-token budgets as execution gates; a completed LLM response is not automatically a successful run.
- Treat failed or unverified Agent turns as non-reusable memory; only Reflection-passed answers and grounded citations enter future session context.
- Keep `research_context.yaml` local by default. Do not send it to an external LLM unless the user explicitly authorizes that data flow.

## Response contract

Return the useful result in the conversation: paper IDs, titles, key method, evidence, limitations, transferability, and artifact paths when relevant. Cite arXiv IDs or the generated report sections rather than dumping raw JSON. Clearly separate facts extracted from the paper from the agent's inference.
