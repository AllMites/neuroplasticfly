# AI usage

Running log for the JOSS AI-usage disclosure (issue #7). Started 2026-10-04; entries before that date are summarised from memory and the commit history.

## Tools
- Claude Code (Anthropic), models Claude Opus 5 and 5.5, Claude Sonnet 5 and 5.5, Claude Fable 5 and 5.1

## Before 2026-10-04 (summary)
- Code: most of the code was written by the AI. The author directed the work and checked the results.
- Analysis: preregistrations, label rules and analysis scripts drafted with the AI. Rules were committed before data (see PROVENANCE.md).
- Writing: drafted with the AI in the author's voice. The author read over and edited every draft, added personal touches, and gave the direction.
- Decisions made by the author: all research questions, scope, results, methods, which points matter, the claims made, and what is preregistered vs post hoc.

## Log (from 2026-10-04)
| Date | Change | AI did | Author did |
|---|---|---|---|
| 2026-10-04 | JOSS roadmap issues #1-#7 | drafted the issue text | chose the scope (S2, JOSS 2027-03) |
| 2026-10-04 | AI_USAGE.md log (#7, PR #8) | drafted the log structure and the pre-2026-10-04 summary | filled the bracketed fields, merged |
| 2026-10-04 | CPU test workflow (#4, PR #9) | wrote tests.yml and chose which data-free tests run | merged |
| 2026-10-06 | prereg.py, template, docs/prereg.md (#3, PR #10) | wrote the code, test and docs from the author's existing workflow | asked for #3 as a public PR; reviews before merge |
| 2026-10-06 | installable package without moving hashed files (#1, PR #13) | wrote the package shim, entry points, test_import.py; ran --verify, --rescore and reproduce.py (unchanged numbers) | asked for the open issues in parallel; merged |
| 2026-10-06 | signal locator API (#2, PR #11) | wrote regime/locator.py, docs/probes.md, test; chose the slope definition from the author's earlier locators | merged |
| 2026-10-06 | CPU CI: small-graph engine test, provenance verify (#4, PR #12) | wrote the fixture and test, classified every test (CPU / data / CUDA) | merged |
