# Edit-distance review & changes

Branch: `claude/stoic-rubin-igtf2d`

This document summarizes a review of the normalized edit-distance machinery in
OmniDocBench's end2end evaluation, the changes made as a result, and the
considerations / caveats that go with them.

---

## 1. Background: how the metric works

The end2end text score is a normalized edit distance:

```
edit(a, b) = Levenshtein.distance(a, b) / max(len(a), len(b))
```

computed per matched GT/prediction element and aggregated. The pipeline first
**matches** GT lines to predicted lines (Hungarian assignment using this
distance as the cost), then scores each matched pair, then aggregates
(`call_Edit_dist` in `src/metrics/cal_metric.py`). The metric's behavior
therefore depends as much on the *matching* stage as on the formula itself.

Key files:
- `src/core/matching/match.py`, `src/core/matching/match_quick.py` — matching.
- `src/core/preprocess/data_preprocess.py` — normalization.
- `src/metrics/cal_metric.py` — metric aggregation.
- `src/dataset/end2end_dataset.py` — per-page matching pipeline + sample
  assembly.

---

## 2. Findings from the review

### 2.1 Issues that were fixed

| # | Severity | Issue |
|---|----------|-------|
| A | Bug (crash) | A page whose elements all normalize to empty crashed `call_Edit_dist` with `KeyError('Edit_num')`; a mixed page emitted a `0/0` `RuntimeWarning`. `Edit_num` was only set when an element had non-empty gt or pred, so the column could be missing entirely. |
| B | Blind spot | Unmatched (extra / hallucinated) predictions were never penalized **and** were partly dropped before they could even be inspected — at three places: `merge_duplicates_add_unmatched` discarded leftover preds in the quick matcher, and the text/formula/table pipelines filter `gt`-less entries downstream. |

### 2.2 Limitations noted but intentionally **not** changed

These are characteristics of the benchmark's design. They are documented here
and pinned by tests (see §4), but changing them would alter published numbers,
so they were left as-is:

- **Normalization discards punctuation and whitespace.** `clean_string`
  (`data_preprocess.py`) keeps only word characters + CJK, so `"hello, world."`
  and `"helloworld"` score identically. Punctuation/spacing errors are invisible
  to the text edit distance.
- **`normalized_formula` lower-cases everything.** Symbols differing only by
  case (`P` vs `p`, `X` vs `x`) are treated as identical.
- **BLEU/METEOR pass gt as `predictions` and pred as `references`**
  (`cal_metric.py` `call_BLEU` / `call_METEOR`). BLEU is asymmetric (brevity
  penalty), so this inverts the length penalty. Edit distance is symmetric and
  unaffected. Flagged for a maintainer decision; not touched.
- **Per-sample averaging dilutes hallucinations.** `edit_sample_avg` treats a
  concatenated blob of many hallucinated lines as one row. The length-weighted
  `edit_whole` is the more faithful headline number.

---

## 3. Changes made

### 3.1 Fix: both-empty / all-empty elements no longer crash `call_Edit_dist`
`src/metrics/cal_metric.py`

- `Edit_num` / `metric['Edit_dist']` are now **always** populated (0 when both
  sides are empty), so the `Edit_num` column always exists.
- Zero-length elements are excluded from the length-weighted aggregates.
- **Behavior preserved for normal pages**: those zero-length rows previously
  contributed `0/0` to the sums (treated as 0) and were skipped by
  `df['ratio'].mean()`, so excluding them yields identical numbers. An all-empty
  page now returns `NaN` sentinels instead of raising.

### 3.2 Feature: spurious-prediction (hallucination) diagnostic — text
`src/core/matching/match_quick.py`, `src/dataset/end2end_dataset.py`,
`src/metrics/cal_metric.py`, `configs/end2end.yaml`

The decision (confirmed with the maintainer) was to **measure** hallucinations
additively rather than change what the headline metrics penalize.

- `merge_duplicates_add_unmatched` now **surfaces** leftover unmatched
  predictions as a `gt`-less entry (`gt_idx == [""]`) instead of dropping them.
  The downstream pipelines still filter `gt`-less entries, so headline
  Edit_dist / TEDS / CDM scores are **unchanged** (verified: 0 `gt`-less entries
  leak into `text_block`).
- `End2EndDataset._build_spurious_pred_record` builds a per-page record from the
  text-mixing matcher output: `spurious_amount` (normalized chars in
  predictions matching no GT line) and `total_amount` (all normalized
  predicted chars), tagged `'unit': 'chars'`.
- New sample category `spurious_pred`; new metric `Spurious_pred`.

### 3.3 Feature: spurious-prediction diagnostic — tables
`src/dataset/end2end_dataset.py`, `src/metrics/cal_metric.py`,
`configs/end2end.yaml`

- Tables are discrete, so the table diagnostic is **count-based**:
  `spurious_amount / total_amount` (table counts, tagged `'unit': 'tables'`)
  per page.
- `End2EndDataset._build_spurious_table_record` counts matched vs. total
  predicted tables per page.
- New sample category `spurious_table`, scored by the same `Spurious_pred`
  metric.
- `call_Spurious_pred` reads **unit-neutral** fields (`spurious_amount` /
  `total_amount`) so one metric serves both the char-based text diagnostic
  and the count-based table diagnostic; a `unit` field on each record
  documents which. Result keys: `page_avg`, `weighted`, `spurious_total`,
  `total`.
  (Both record builders originally also stored category-named duplicates —
  `pred_chars`/`spurious_chars`, `pred_tables`/`spurious_tables` — that no
  code ever read; these were removed in favor of the `unit` field.)
- Headline table TEDS / Edit_dist unchanged (verified: 0 `gt`-less entries leak
  into the `table` category).

---

### 3.4 Fix: softened the 0.7 rejection cliff in quick_match
`src/core/matching/match_quick.py`

`process_matches` rejects Hungarian pairs with edit > 0.7 (now the named
constant `QUICK_MATCH_REJECT_EDIT`) so fuzzy recovery can try to re-match both
sides. Previously, pairs that recovery could not improve were scored as a
missed GT (edit=1) **plus** an unmatched prediction — a discontinuity where a
pair at 0.71 scored strictly worse than one at 0.70, and (combined with
paragraph-boundary misalignment, see §2.2) a perfectly extracted paragraph
could score as a total miss.

Now `restore_rejected_pairs` restores such pairs with their **actual** edit
distance after recovery has had its chance. Pairs with edit ≥ 1 (no shared
content) stay rejected so garbage predictions still surface as unmatched.
The score is thereby continuous in prediction quality.

Notes:
- On the demo set the headline `edit_whole` is unchanged; restorable
  rejections are rare for good predictions. The improvement shows on pages
  with heavy segmentation errors, and in pairing quality: restored pairs
  replace the arbitrary "force-pair leftover GT with leftover pred at edit=1"
  entries, which also feeds cleaner pairings to the reading-order metric and
  the spurious diagnostic.
- The paragraph-segmentation double-counting itself (a boundary landing
  mid-paragraph leaks edits into both neighboring pairs) is **still present**
  and pinned by `TestParagraphSegmentationSensitivity`; fixing it needs
  boundary repair or alignment-based matching (see follow-ups).

## 4. Tests
`tests/test_edit_distance.py` (run: `python -m pytest tests/test_edit_distance.py -v`)

Characterization tests pin behavior so any future change is explicit:

- **Core formula** — both-empty, one-empty, symmetry, `[0,1]` bound, matrix guard.
- **Normalization limitations** — punctuation/whitespace stripped, formula
  case-folded (marked `LIMITATION`).
- **Fixes** — both-empty scored 0 and excluded from aggregates; all-empty page
  returns `NaN` without crashing.
- **Hallucination handling** — pure hallucination scored 1.0 if it reaches the
  metric; quick-match merge now surfaces leftover preds.
- **Diagnostics** — text record builder (char counts), table record builder
  (table counts), and the shared `Spurious_pred` metric in both units.

Tests originally written to pin a bug were flipped from `CURRENT BEHAVIOR` to
`FIXED BEHAVIOR` in the same commit as the fix, so the change in behavior is
reviewable in the diff.

---

## 5. How to read the new numbers

Per element, `Spurious_pred` reports:

| key | meaning |
|-----|---------|
| `page_avg` | mean over pages of `spurious / total` (pages with no predictions excluded) |
| `weighted` | corpus-level `sum(spurious) / sum(total)` |
| `spurious_total` / `total` | raw totals (chars for `spurious_pred`, table counts for `spurious_table`) |

Demo-set sanity check (18 pages):
- `spurious_pred` ≈ 3.2% weighted (618 / 19,122 chars; 6/18 pages affected).
- `spurious_table` ≈ 9.1% weighted (1 / 11 tables; 1 page affected).

---

## 6. Considerations & caveats

- **Different units across diagnostics.** `spurious_pred` is char-weighted;
  `spurious_table` is count-weighted. They are reported under the same metric
  name but in **separate categories** — do not sum them into a single number.
- **Headline metrics are deliberately unchanged.** The diagnostics are additive;
  Edit_dist / TEDS / CDM scores remain comparable to previously published runs.
  This was verified by checking that no `gt`-less entries leak into the
  `text_block` or `table` categories on the demo set.
- **Coverage.** The text diagnostic covers the text-mixing path (text + inline /
  isolated equations). Standalone display-formula hallucinations are **not yet**
  counted — a `spurious_formula` category would be the natural next step.
- **Table denominator.** `total_amount` on `spurious_table` records counts
  tables routed through the table matcher (predicted HTML tables + LaTeX
  tables converted to HTML). Markdown
  tables that were converted upstream are included via that conversion;
  anything not classified as a table by `md_tex_filter` is out of scope.
- **The diagnostics are not yet in the run-summary / notebook tables.**
  `Spurious_pred` shows up in the per-element result JSON and console output but
  is not wired into `build_notebook_metric_summary` / the headline report. Add
  it there if it should appear in the summary tables.
- **`unmatch_table_pred`** (the split text items for extra tables in the
  GT-present case) remains computed-but-unused in `process_get_matched_elements`;
  the count-based table diagnostic does not rely on it. It could be removed or
  repurposed separately.

---

## 7. Suggested follow-ups (not done)

0. Fix paragraph-boundary double-counting: content-identical predictions whose
   paragraph boundary lands mid-paragraph leak large edits into both
   neighboring pairs (pinned by `TestParagraphSegmentationSensitivity`).
   Options: local boundary repair between adjacent matched pairs (cheap), or
   segmentation-invariant alignment-projection matching (principled).
1. Add a `spurious_formula` diagnostic for standalone display formulas.
2. Surface `Spurious_pred` in the run-summary / notebook tables.
3. Fix the BLEU/METEOR `predictions`/`references` swap (a behavior change for
   those two metrics).
4. Decide whether punctuation/whitespace and formula case should remain
   normalized away; if not, adjust `clean_string` / `normalized_formula`.
5. De-duplicate the two identical `_normalized_edit_distance` implementations
   (`match.py` and `end2end_dataset.py`).
