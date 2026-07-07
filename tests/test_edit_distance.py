"""Characterization tests for the normalized edit-distance machinery.

These tests pin the *current* behavior of the edit-distance metric and its
supporting matching / normalization code. Some of them deliberately assert
behavior that is arguably wrong (hallucination blind spots, over-aggressive
normalization, an unhandled empty-pair edge case). Those are marked with
``CURRENT BEHAVIOR`` / ``LIMITATION`` so that when the underlying issue is
fixed the test is updated in lock-step and the change in behavior is explicit
and reviewable.

Run:
    python -m pytest tests/test_edit_distance.py -v
"""

import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.matching.match import (  # noqa: E402
    _normalized_edit_distance,
    compute_edit_distance_matrix_new,
    match_gt2pred_simple,
)
from src.core.matching.match_quick import merge_duplicates_add_unmatched  # noqa: E402
from src.core.preprocess.data_preprocess import (  # noqa: E402
    clean_string,
    normalized_formula,
    normalized_text,
)
from src.metrics.cal_metric import call_Edit_dist  # noqa: E402


@pytest.fixture(autouse=True)
def _cleanup_edit_dist_side_effects():
    # call_Edit_dist writes ./result/<save_name>_per_page_edit.json as a side
    # effect; remove the default-named artifact so local runs stay clean.
    yield
    stray = REPO_ROOT / "result" / "default_per_page_edit.json"
    if stray.exists():
        stray.unlink()


# ===========================================================================
# 1. Core normalized edit-distance formula
# ===========================================================================
class TestNormalizedEditDistanceFormula:
    def test_both_empty_returns_zero(self):
        assert _normalized_edit_distance("", "") == 0

    def test_one_side_empty_returns_one(self):
        assert _normalized_edit_distance("abc", "") == 1
        assert _normalized_edit_distance("", "abc") == 1

    def test_identical_returns_zero(self):
        assert _normalized_edit_distance("abc", "abc") == 0.0

    def test_single_substitution_normalized_by_max_len(self):
        # one of three chars differs -> 1/3
        assert _normalized_edit_distance("abc", "abd") == pytest.approx(1 / 3)

    def test_is_symmetric(self):
        a, b = "the quick brown fox", "the quik brwn fox"
        assert _normalized_edit_distance(a, b) == _normalized_edit_distance(b, a)

    def test_is_bounded_in_unit_interval(self):
        for a, b in [("", "x"), ("abc", "xyz"), ("longer string", "x"), ("a", "")]:
            d = _normalized_edit_distance(a, b)
            assert 0 <= d <= 1

    def test_matrix_guards_both_empty_cell(self):
        # both-empty cell must be 0 (not a ZeroDivisionError)
        m = compute_edit_distance_matrix_new(["", "ab"], ["", "ac"])
        assert m[0][0] == 0.0      # both empty
        assert m[0][1] == 1.0      # "" vs "ac"
        assert m[1][0] == 1.0      # "ab" vs ""
        assert m[1][1] == 0.5      # "ab" vs "ac"


# ===========================================================================
# 2. Normalization: what the edit distance does NOT see
#    (these pin LIMITATIONS of normalized_text / normalized_formula)
# ===========================================================================
class TestNormalizationSwallowsErrors:
    def test_punctuation_is_stripped_before_scoring(self):
        # LIMITATION: all punctuation is removed, so a model that drops every
        # comma/period scores a perfect match.
        assert clean_string("hello, world.") == "helloworld"
        assert normalized_text("hello, world.") == normalized_text("helloworld")

    def test_whitespace_is_stripped_collapsing_word_boundaries(self):
        # LIMITATION: spaces are removed, so word merges/splits are invisible.
        assert normalized_text("hello world") == normalized_text("helloworld")

    def test_formula_is_case_folded(self):
        # LIMITATION: normalized_formula lower-cases everything, so symbols that
        # differ only by case (P vs p, X vs x) are treated as identical.
        assert normalized_formula("P_{n}") == normalized_formula("p_{n}")


# ===========================================================================
# 3. Hallucination handling
# ===========================================================================
class TestHallucinationPenalty:
    def test_edit_dist_penalizes_pure_hallucination(self):
        # CORRECT BEHAVIOR: a prediction with no matching GT scores 1.0.
        samples = [
            {"img_id": "p.jpg", "gt": "", "pred": "aaaaaaaaaaaaaaaa",
             "norm_gt": "", "norm_pred": "aaaaaaaaaaaaaaaa"},
        ]
        _, res = call_Edit_dist(samples).evaluate()
        assert samples[0]["metric"]["Edit_dist"] == 1.0
        assert res["Edit_dist"]["edit_whole"] == 1.0

    def test_simple_match_collapses_extra_preds_into_one_entry(self):
        # CURRENT BEHAVIOR (dilution): several hallucinated predictions are
        # concatenated into a SINGLE unmatched entry (gt_idx == [""]). In the
        # per-sample average this counts as one row at edit=1, not N rows.
        gt_items = [{
            "category_type": "text_block", "text": "the quick brown fox",
            "position": [0, 1], "order": 1, "attribute": {},
        }]
        pred_items = [
            {"category_type": "text_all", "content": "the quick brown fox", "position": [0, 1]},
            {"category_type": "text_all", "content": "hallucinated extra one", "position": [2, 3]},
            {"category_type": "text_all", "content": "hallucinated extra two", "position": [4, 5]},
        ]
        match, _ = match_gt2pred_simple(gt_items, pred_items, "text_all", "img")

        halluc_entries = [e for e in match if e["gt_idx"] == [""]]
        assert len(halluc_entries) == 1
        # both extras folded into the single entry
        assert list(halluc_entries[0]["pred_idx"]) == [1, 2]
        assert halluc_entries[0]["edit"] == 1

    def test_quick_merge_surfaces_leftover_unmatched_preds(self):
        # FIXED BEHAVIOR: merge_duplicates_add_unmatched now re-adds unmatched
        # predictions as a gt-less entry (gt_idx == [""]) instead of dropping
        # them, so the spurious-prediction diagnostic can see hallucinated text.
        out = merge_duplicates_add_unmatched(
            converted_results=[],            # nothing matched
            norm_gt_lines=["gtline"],
            norm_pred_lines=["predaaa", "predbbb"],
            gt_lines=["gtline"],
            pred_lines=["predaaa", "predbbb"],
            all_gt_indices={0},
            all_pred_indices={0, 1},
        )
        # the missing GT is still surfaced as a miss...
        missing_gt = [e for e in out if e["gt_idx"] == [0]]
        assert len(missing_gt) == 1
        assert missing_gt[0]["pred_idx"] == [""]
        # ...and both leftover predictions are now present in one gt-less entry
        spurious = [e for e in out if e["gt_idx"] == [""]]
        assert len(spurious) == 1
        assert list(spurious[0]["pred_idx"]) == [0, 1]
        assert spurious[0]["edit"] == 1


# ===========================================================================
# 4. Aggregation edge cases in call_Edit_dist
# ===========================================================================
class TestEditDistAggregationEdgeCases:
    def test_both_empty_sample_is_scored_zero_and_excluded_from_aggregates(self):
        # FIXED BEHAVIOR: a both-empty sample now gets Edit_num=0 / Edit_dist=0
        # (so the column always exists) and is excluded from the length-weighted
        # aggregates, leaving the surviving page's score untouched and emitting
        # no divide-by-zero warning.
        samples = [
            {"img_id": "ok.jpg", "gt": "abcd", "pred": "abcd",
             "norm_gt": "abcd", "norm_pred": "abcd"},
            {"img_id": "empty.jpg", "gt": "", "pred": "",
             "norm_gt": "", "norm_pred": ""},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # no 0/0 warning allowed
            _, res = call_Edit_dist(samples).evaluate()

        empty_sample = samples[1]
        assert empty_sample["metric"]["Edit_dist"] == 0.0
        assert empty_sample["Edit_num"] == 0
        assert empty_sample["upper_len"] == 0
        # aggregate reflects only the real (ok.jpg) element
        assert res["Edit_dist"]["edit_whole"] == 0.0

    def test_all_empty_samples_return_nan_without_crashing(self):
        # FIXED BEHAVIOR: a page whose elements all normalize to empty no longer
        # raises KeyError('Edit_num'); it returns NaN sentinels instead.
        samples = [
            {"img_id": "empty.jpg", "gt": "", "pred": "",
             "norm_gt": "", "norm_pred": ""},
        ]
        _, res = call_Edit_dist(samples).evaluate()
        assert res["Edit_dist"]["edit_whole"] == "NaN"
        assert res["Edit_dist"]["ALL_page_avg"] == "NaN"
        assert res["Edit_dist"]["edit_sample_avg"] == "NaN"


# ===========================================================================
# 5. Spurious-prediction diagnostic (additive; does not touch Edit_dist)
# ===========================================================================
class TestSpuriousPredDiagnostic:
    def test_record_builder_counts_unmatched_pred_chars(self):
        from src.dataset.end2end_dataset import End2EndDataset
        match = [
            {"gt_idx": [0], "norm_pred": "matchedpred", "pred": "matchedpred"},  # 11
            {"gt_idx": [""], "norm_pred": "halluc", "pred": "halluc"},           # 6 (spurious)
        ]
        # method uses no instance state, so an unbound call with self=None is fine
        rec = End2EndDataset._build_spurious_pred_record(None, match, "p.jpg")
        assert rec["img_id"] == "p.jpg"
        assert rec["pred_chars"] == 17
        assert rec["spurious_chars"] == 6

    def test_record_builder_returns_empty_when_no_predictions(self):
        from src.dataset.end2end_dataset import End2EndDataset
        match = [{"gt_idx": [0], "norm_pred": "", "pred": ""}]
        assert End2EndDataset._build_spurious_pred_record(None, match, "p.jpg") == {}

    def test_record_builder_emits_unit_neutral_amounts(self):
        from src.dataset.end2end_dataset import End2EndDataset
        match = [
            {"gt_idx": [0], "norm_pred": "matchedpred", "pred": "matchedpred"},
            {"gt_idx": [""], "norm_pred": "halluc", "pred": "halluc"},
        ]
        rec = End2EndDataset._build_spurious_pred_record(None, match, "p.jpg")
        # the metric consumes the unit-neutral fields (here: characters)
        assert rec["spurious_amount"] == rec["spurious_chars"] == 6
        assert rec["total_amount"] == rec["pred_chars"] == 17

    def test_table_record_builder_counts_spurious_tables(self):
        from src.dataset.end2end_dataset import End2EndDataset
        # 3 predicted tables, 1 matched a GT table -> 2 spurious
        rec = End2EndDataset._build_spurious_table_record(None, 3, 1, "p.jpg")
        assert rec["img_id"] == "p.jpg"
        assert rec["pred_tables"] == 3
        assert rec["spurious_tables"] == 2
        assert rec["total_amount"] == 3
        assert rec["spurious_amount"] == 2

    def test_table_record_builder_empty_when_no_pred_tables(self):
        from src.dataset.end2end_dataset import End2EndDataset
        assert End2EndDataset._build_spurious_table_record(None, 0, 0, "p.jpg") == {}

    def test_metric_reports_page_and_corpus_ratios(self):
        from src.metrics.cal_metric import call_Spurious_pred
        samples = [
            {"img_id": "a.jpg", "spurious_amount": 20, "total_amount": 100},  # 0.2
            {"img_id": "b.jpg", "spurious_amount": 0, "total_amount": 50},    # 0.0
        ]
        _, res = call_Spurious_pred(samples).evaluate()
        assert samples[0]["metric"]["Spurious_pred"] == pytest.approx(0.2)
        assert res["Spurious_pred"]["page_avg"] == pytest.approx(0.1)
        assert res["Spurious_pred"]["weighted"] == pytest.approx(20 / 150)

    def test_metric_works_for_table_counts(self):
        from src.metrics.cal_metric import call_Spurious_pred
        # same metric, table-count unit: 2 spurious of 3 predicted, plus a clean page
        samples = [
            {"img_id": "a.jpg", "spurious_amount": 2, "total_amount": 3},
            {"img_id": "b.jpg", "spurious_amount": 0, "total_amount": 1},
        ]
        _, res = call_Spurious_pred(samples).evaluate()
        assert res["Spurious_pred"]["weighted"] == pytest.approx(2 / 4)
        assert res["Spurious_pred"]["page_avg"] == pytest.approx((2 / 3 + 0) / 2)

    def test_metric_handles_empty_and_zero_pred_pages(self):
        from src.metrics.cal_metric import call_Spurious_pred
        _, res_empty = call_Spurious_pred([]).evaluate()
        assert res_empty["Spurious_pred"]["page_avg"] == "NaN"
        assert res_empty["Spurious_pred"]["weighted"] == "NaN"

        samples = [{"img_id": "a.jpg", "spurious_amount": 0, "total_amount": 0}]
        _, res_zero = call_Spurious_pred(samples).evaluate()
        # a page with no predictions contributes no ratio to the page average
        assert res_zero["Spurious_pred"]["page_avg"] == "NaN"
        assert res_zero["Spurious_pred"]["weighted"] == "NaN"


# ===========================================================================
# 6. Paragraph-segmentation sensitivity of quick_match
# ===========================================================================
class TestParagraphSegmentationSensitivity:
    """Pin how the line-based matcher scores content-identical predictions that
    only differ in where paragraph boundaries fall.

    CURRENT BEHAVIOR (bug): the matcher handles merged paragraphs and clean
    sub-splits, but a boundary landing MID-paragraph leaks large edit distances
    even though the concatenated text is character-identical to the GT. The
    misplaced fragment is double-counted (as an insertion in one pair and a
    deletion in the other).

    The 0.7 rejection cliff has been softened: pairs above the threshold that
    fuzzy recovery cannot improve are restored with their actual edit distance
    instead of being scored as a miss (edit=1) plus a spurious prediction. The
    double-counting itself remains; a segmentation-robust matcher should drive
    all these totals to ~0, and the remaining buggy assertions should be
    flipped when that lands.
    """

    PARA_A = ("The quick brown fox jumps over the lazy dog while the sun "
              "sets slowly behind the mountains.")
    PARA_B = ("Meanwhile the river flows gently through the valley carrying "
              "leaves and small branches downstream.")

    def _match(self, pred_texts):
        from src.core.matching.match_quick import match_gt2pred_quick
        gt_items = [
            {"category_type": "text_block", "text": self.PARA_A,
             "order": 1, "position": [0, 1], "attribute": {}},
            {"category_type": "text_block", "text": self.PARA_B,
             "order": 2, "position": [2, 3], "attribute": {}},
        ]
        pred_items = [
            {"category_type": "text_all", "content": text,
             "position": [i * 10, i * 10 + 9]}
            for i, text in enumerate(pred_texts)
        ]
        return match_gt2pred_quick(gt_items, pred_items, "text_all", "img")

    @staticmethod
    def _total_edit(match):
        return sum(float(m["edit"]) for m in match)

    def _preds_content_identical_to_gt(self, pred_texts):
        # precondition helper: normalized concatenation identical to GT's
        gt_norm = normalized_text(self.PARA_A) + normalized_text(self.PARA_B)
        pred_norm = "".join(normalized_text(t) for t in pred_texts)
        return gt_norm == pred_norm

    def test_same_split_scores_zero(self):
        preds = [self.PARA_A, self.PARA_B]
        assert self._preds_content_identical_to_gt(preds)
        assert self._total_edit(self._match(preds)) == pytest.approx(0.0)

    def test_merged_paragraphs_score_zero(self):
        # dropping the paragraph break entirely is handled (fuzzy recovery)
        preds = [self.PARA_A + " " + self.PARA_B]
        assert self._preds_content_identical_to_gt(preds)
        assert self._total_edit(self._match(preds)) == pytest.approx(0.0)

    def test_clean_oversplit_scores_near_zero(self):
        # splitting each paragraph in half is handled (truncation merge)
        half_a = self.PARA_A.rfind(" ", 0, len(self.PARA_A) // 2)
        half_b = self.PARA_B.rfind(" ", 0, len(self.PARA_B) // 2)
        preds = [self.PARA_A[:half_a], self.PARA_A[half_a + 1:],
                 self.PARA_B[:half_b], self.PARA_B[half_b + 1:]]
        assert self._preds_content_identical_to_gt(preds)
        assert self._total_edit(self._match(preds)) < 0.05

    def test_shifted_boundary_leaks_large_edit_bug(self):
        # CURRENT BEHAVIOR (bug): move the paragraph boundary to the middle of
        # PARA_B; content is unchanged but both matched rows leak large edits.
        shift = self.PARA_B.rfind(" ", 0, len(self.PARA_B) // 2)
        preds = [self.PARA_A + " " + self.PARA_B[:shift],
                 self.PARA_B[shift + 1:]]
        assert self._preds_content_identical_to_gt(preds)

        match = self._match(preds)
        matched_rows = [m for m in match if m["gt_idx"] != [""]]
        assert len(matched_rows) == 2
        # each row eats a big chunk of edit distance despite identical content
        assert all(float(m["edit"]) > 0.2 for m in matched_rows)
        assert self._total_edit(match) > 0.5

    def test_large_shift_keeps_true_pair_distance(self):
        # FIXED BEHAVIOR (cliff softened): shift the boundary 80% into PARA_B;
        # the second pair's edit exceeds the 0.7 rejection threshold, but the
        # pair is restored with its actual edit distance instead of being
        # scored as a full miss (edit=1) plus a spurious prediction. The
        # segmentation double-counting itself still inflates the total.
        shift = self.PARA_B.rfind(" ", 0, int(len(self.PARA_B) * 0.8))
        preds = [self.PARA_A + " " + self.PARA_B[:shift],
                 self.PARA_B[shift + 1:]]
        assert self._preds_content_identical_to_gt(preds)

        match = self._match(preds)
        edits = sorted(float(m["edit"]) for m in match if m["gt_idx"] != [""])
        assert 0.7 < edits[-1] < 1.0     # kept at its true distance, not 1.0
        # no spurious row: the restored pair consumes the prediction
        assert all(m["gt_idx"] != [""] for m in match)

    def test_rejected_pair_above_threshold_keeps_actual_edit(self):
        # FIXED BEHAVIOR (cliff softened): a pair whose edit lands just above
        # the 0.7 rejection threshold keeps its actual edit distance instead
        # of jumping to 1.0 (miss) + spurious prediction.
        from src.core.matching.match_quick import (
            QUICK_MATCH_REJECT_EDIT,
            match_gt2pred_quick,
        )
        gt_items = [
            {"category_type": "text_block",
             "text": "completely different first line content here",
             "order": 1, "position": [0, 1], "attribute": {}},
            {"category_type": "text_block",
             "text": "alpha beta gamma delta epsilon zeta eta theta",
             "order": 2, "position": [2, 3], "attribute": {}},
        ]
        pred_items = [
            {"category_type": "text_all",
             "content": "completely different first line content here",
             "position": [0, 9]},
            # shares only the first two words with its GT -> edit ~0.76
            {"category_type": "text_all",
             "content": "alpha beta xxxx yyyy zzzz qqqq wwww rrrr",
             "position": [10, 19]},
        ]
        match = match_gt2pred_quick(gt_items, pred_items, "text_all", "img")
        edits = sorted(float(m["edit"]) for m in match if m["gt_idx"] != [""])
        assert edits[0] == pytest.approx(0.0)
        assert QUICK_MATCH_REJECT_EDIT < edits[-1] < 1.0
        # nothing became spurious: the restored pair consumed the prediction
        assert all(m["gt_idx"] != [""] for m in match)

    def test_fully_unrelated_pair_is_not_restored(self):
        # edit >= 1 pairs stay rejected: a prediction sharing nothing with its
        # GT must not be laundered into a "match" by the restore step.
        from src.core.matching.match_quick import match_gt2pred_quick
        gt_items = [
            {"category_type": "text_block",
             "text": "completely different first line content here",
             "order": 1, "position": [0, 1], "attribute": {}},
            {"category_type": "text_block",
             "text": "alpha beta gamma delta epsilon zeta eta theta",
             "order": 2, "position": [2, 3], "attribute": {}},
        ]
        pred_items = [
            {"category_type": "text_all",
             "content": "completely different first line content here",
             "position": [0, 9]},
            {"category_type": "text_all",
             "content": "0123456789 0123456789 0123456789",
             "position": [10, 19]},
        ]
        match = match_gt2pred_quick(gt_items, pred_items, "text_all", "img")
        edits = sorted(float(m["edit"]) for m in match if m["gt_idx"] != [""])
        assert edits[-1] == 1.0

    def test_penalty_grows_with_shift_distance(self):
        # CURRENT BEHAVIOR: even a one-word boundary shift is penalized, and
        # the penalty grows with the shift, despite identical content.
        one_word = self.PARA_B.find(" ")
        mid = self.PARA_B.rfind(" ", 0, len(self.PARA_B) // 2)

        def total_for(shift):
            preds = [self.PARA_A + " " + self.PARA_B[:shift],
                     self.PARA_B[shift + 1:]]
            assert self._preds_content_identical_to_gt(preds)
            return self._total_edit(self._match(preds))

        small, large = total_for(one_word), total_for(mid)
        assert small > 0.1        # nonzero even for a one-word shift
        assert large > small      # grows with shift distance


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
