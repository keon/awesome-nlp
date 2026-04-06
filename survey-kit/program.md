# survey autoresearch program

This is an experiment to have an LLM improve a literature survey through small, reversible iterations.

## Current run

- Topic: modern NLP
- Run tag: `nlp-overview-2026-04-06`
- Target reader: ML engineers, research engineers, and graduate students who know basic deep learning but want a compact map of NLP
- In scope: text-centric NLP, transfer learning, foundation models, retrieval augmentation, multilingual transfer, evaluation
- Out of scope: speech-only systems, vision-language systems, and exhaustive task-by-task benchmark tables

## Setup

To set up a new survey run, work with the user to:

1. Confirm the scope, target reader, and exclusions.
2. Read the in-scope files:
   - `survey-kit/README.md` — workflow overview.
   - `survey-kit/program.md` — the rules in this file.
   - `survey-kit/survey.md` — the only file you should modify during revision loops.
3. Confirm the canonical papers or sources that must appear.
4. Initialize `survey-kit/results.tsv` with the header row if it does not exist.
5. Produce a baseline draft before any optimization attempts.

## Editable surface

What you CAN do:

- Modify `survey-kit/survey.md`.
- Reorganize sections, claims, tables, and citations inside that file.
- Remove material that does not earn its place.

What you CANNOT do unless the user explicitly asks:

- Change the scope, rubric, or audience assumptions in this file.
- Rewrite the workflow in `survey-kit/README.md`.
- Add new files, tools, or dependencies for a normal revision loop.

## Goal

Write the most useful survey for the stated audience.

Useful means:

- broad enough to cover the field's main lines of work
- selective enough to avoid list-like sprawl
- organized around a clear taxonomy
- explicit about tradeoffs, not just summaries
- grounded in citations for major claims

## Baseline first

The very first run should establish the baseline survey draft. Do not optimize before a baseline exists.

A good baseline includes:

- title and one-paragraph thesis
- target reader statement
- scope and exclusions
- section outline
- canonical papers grouped into a first-pass taxonomy
- at least one synthesis paragraph per major section

## Evaluation rubric

After each revision, score the draft on a 1 to 5 scale for:

1. coverage
2. structure
3. synthesis
4. evidence
5. reader utility
6. compression

Track the total score as the main metric.

## Simplicity criterion

All else being equal, simpler is better.

- A meaningful improvement with equal complexity is a keep.
- A tiny improvement with much more structure or prose is usually a discard.
- Equal score with less redundancy is a keep.
- Equal score with clearer taxonomy is a keep.

## Revision loop

LOOP:

1. Read the current `survey-kit/survey.md`.
2. Choose one concrete revision hypothesis.
3. Edit only `survey-kit/survey.md`.
4. Re-score the survey using the rubric above.
5. Log the iteration in `survey-kit/results.tsv`.
6. Keep the revision only if the new score improves enough to justify the change.
7. If the change does not help, revert and try a different hypothesis.

## Logging format

Use tab-separated values in `survey-kit/results.tsv` with this header:

`iteration	score	status	hypothesis	notes`

Where:

- `iteration` is a short tag such as `baseline`, `r1`, `r2`
- `score` is the rubric total, such as `22/30`
- `status` is `keep` or `discard`
- `hypothesis` is the single change under test
- `notes` records the reason for the decision

## Output style

Prefer:

- direct claims
- compact comparison tables
- explicit tradeoffs
- short, high-signal paragraphs
- forward pointers between sections when helpful

Avoid:

- laundry lists
- repeating abstracts
- large chronological dumps without synthesis
- inflated prose
- unsupported field-wide claims
