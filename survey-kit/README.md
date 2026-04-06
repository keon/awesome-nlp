# Autoresearch-style survey kit

This kit adapts the philosophy from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to survey writing.

## Core idea

Treat survey writing like a controlled optimization loop:

- keep the editable surface tiny
- define a fixed evaluation rubric
- run short, comparable iterations
- keep only revisions that improve the survey
- prefer simplification over churn

In this adaptation:

- the **human** edits `program.md`
- the **agent** edits `survey.md`
- each iteration is logged in `results.tsv`

## Files

- `program.md` — operating instructions, scope, rubric, and keep/discard policy
- `survey.md` — the only file the agent should rewrite during iterations
- `results.tsv` — experiment log for revisions and outcomes

## Suggested loop

1. Establish a baseline survey draft.
2. Pick one revision hypothesis.
3. Apply only that change to `survey.md`.
4. Score the result against the rubric in `program.md`.
5. Keep the revision only if it improves the draft enough to justify the added complexity.
6. Log the result in `results.tsv`.
7. Repeat.

## Good revision hypotheses

- reorganize the taxonomy around methods instead of chronology
- merge two redundant sections
- add a failure modes section
- replace a long narrative block with a compact comparison table
- remove low-signal citations that do not support the core claims

## Bad revision hypotheses

- rewrite the whole survey without a specific reason
- add a new section because it "might be useful"
- expand citations without improving synthesis
- keep a more complex structure for a negligible gain

## Scoring guidance

Use a small stable rubric across iterations. Example dimensions:

- **coverage** — includes the canonical papers and main lines of work
- **structure** — sections are organized around a useful taxonomy
- **synthesis** — compares approaches instead of listing them
- **evidence** — major claims are supported by sources
- **reader utility** — a newcomer can understand the field and next actions
- **compression** — minimal redundancy and fluff

Score on a fixed scale, such as 1 to 5 per dimension.

## Simplicity rule

Borrow the same bias from autoresearch:

- equal quality + simpler draft = keep
- tiny gain + much more complexity = discard
- same coverage + less redundancy = keep

## Suggested operating rhythm

- first pass: scope and baseline outline
- second pass: canonical papers and taxonomy
- later passes: one targeted structural or synthesis improvement at a time

The point is not to produce the longest survey. The point is to produce the clearest and most useful one.


## Automation

This kit can be refreshed automatically in GitHub Actions using `survey-kit/generate_survey.py`.

The automation intentionally uses a **hybrid** design:

- **AutoSurvey-style retrieval**: first gather current sources and a compact research memo from the web
- **autoresearch-style revision loop**: then revise the single editable artifact (`survey.md`) with a small, reviewable weekly change

The workflow lives at `.github/workflows/weekly-survey.yml` and opens or updates a pull request instead of committing directly to the default branch.

### Required repository secret

- `OPENAI_API_KEY`

### Optional repository variable

- `OPENAI_MODEL` — defaults to `gpt-5` if unset

### Manual runs

The workflow also supports `workflow_dispatch` inputs so you can run it on demand from GitHub Actions with custom values for:

- `topic`
- `audience`
- `model`
- `force_pr`

If you leave them blank, the workflow falls back to the default NLP survey settings.

Set `force_pr` to `true` when you want the action to open or update the automation PR even if that run does not produce a file diff.
