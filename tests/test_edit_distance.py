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

    def test_quick_merge_drops_leftover_unmatched_preds(self):
        # CURRENT BEHAVIOR (blind spot): merge_duplicates_add_unmatched re-adds
        # unmatched GT lines as misses, but NEVER re-adds unmatched predictions.
        # So leftover hallucinated preds disappear entirely from scoring.
        out = merge_duplicates_add_unmatched(
            converted_results=[],            # nothing matched
            norm_gt_lines=["gtline"],
            norm_pred_lines=["predaaa", "predbbb"],
            gt_lines=["gtline"],
            pred_lines=["predaaa", "predbbb"],
            all_gt_indices={0},
            all_pred_indices={0, 1},
        )
        # the missing GT is surfaced...
        missing_gt = [e for e in out if e["gt_idx"] == [0]]
        assert len(missing_gt) == 1
        assert missing_gt[0]["pred_idx"] == [""]
        # ...but neither hallucinated prediction is present anywhere
        entries_with_pred = [e for e in out if e.get("pred_idx") not in ([""], "")]
        assert entries_with_pred == []


# ===========================================================================
# 4. Aggregation edge cases in call_Edit_dist
# ===========================================================================
class TestEditDistAggregationEdgeCases:
    def test_both_empty_sample_leaves_metric_unset(self):
        # CURRENT BEHAVIOR: when both gt and pred are empty the metric block is
        # never populated (Edit_num stays unset) and the page contributes a
        # NaN / divide-by-zero that pandas silently drops.
        samples = [
            {"img_id": "ok.jpg", "gt": "abcd", "pred": "abcd",
             "norm_gt": "abcd", "norm_pred": "abcd"},
            {"img_id": "empty.jpg", "gt": "", "pred": "",
             "norm_gt": "", "norm_pred": ""},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, res = call_Edit_dist(samples).evaluate()

        empty_sample = samples[1]
        assert empty_sample.get("metric") is None        # never set
        assert empty_sample.get("Edit_num") is None       # never set
        assert empty_sample["upper_len"] == 0
        # the empty page is silently excluded; surviving avg is from ok.jpg only
        assert res["Edit_dist"]["edit_whole"] == 0.0

    def test_all_empty_samples_crash_with_keyerror(self):
        # CURRENT BEHAVIOR (bug): if EVERY sample is both-empty, no sample ever
        # gets an 'Edit_num' field, so the column is missing from the DataFrame
        # and aggregation raises KeyError instead of returning a score. A page
        # whose elements all normalize to empty therefore crashes the metric.
        samples = [
            {"img_id": "empty.jpg", "gt": "", "pred": "",
             "norm_gt": "", "norm_pred": ""},
        ]
        with pytest.raises(KeyError, match="Edit_num"):
            call_Edit_dist(samples).evaluate()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
