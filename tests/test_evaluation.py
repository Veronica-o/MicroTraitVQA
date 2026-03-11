"""
Tests for the evaluation metrics.
These don't need a GPU or model weights — just run pytest.
"""

import sys
sys.path.insert(0, ".")

import pytest
from bio_vqa.vqa import caption_overlap, completeness, cross_model_agreement, evaluate
from bio_vqa.models import Figure, VQAResult


# helpers
def make_figure(caption="chromatogram retention peaks"):
    return Figure("fig1", "Figure 1", caption, "/fake/img.jpg", "img.jpg")

def make_result(model="model-a", answer="chromatogram peaks", question="Q1"):
    return VQAResult("fig1", question, "figure_understanding", model, answer, 1.0, False)


# --- caption_overlap ---

def test_overlap_identical():
    assert caption_overlap("chromatogram peak retention", "chromatogram peak retention") == 1.0

def test_overlap_no_match():
    assert caption_overlap("apple banana", "xenon plasma") == 0.0

def test_overlap_partial():
    s = caption_overlap("peak retention time 7.3", "retention time chromatogram")
    assert 0.0 < s < 1.0

def test_overlap_empty_inputs():
    assert caption_overlap("", "some caption") == 0.0
    assert caption_overlap("some answer", "") == 0.0

def test_overlap_stop_words_only():
    # stop words are filtered out, so this should be 0
    assert caption_overlap("the a an is are", "the a an is are") == 0.0


# --- completeness ---

def test_completeness_empty():
    assert completeness("") == 0.0

def test_completeness_very_short():
    assert completeness("Yes") < 0.2

def test_completeness_long_answer():
    long = " ".join(["word"] * 35) + "."
    assert completeness(long) <= 1.05

def test_completeness_punctuation_adds_bonus():
    assert completeness("A peak at 7.3 minutes.") > completeness("A peak at 7 3 minutes")


# --- cross_model_agreement ---

def test_agreement_single_model():
    assert cross_model_agreement(["only one answer"]) == 1.0

def test_agreement_identical():
    assert cross_model_agreement(["peak at 7.3", "peak at 7.3"]) == 1.0

def test_agreement_total_mismatch():
    assert cross_model_agreement(["apple banana", "xenon reactor"]) == 0.0

def test_agreement_partial():
    s = cross_model_agreement(["peak at 7.3 chromatogram", "peak retention 7.3 UHPLC"])
    assert 0.0 < s < 1.0


# --- evaluate ---

def test_evaluate_fills_scores():
    fig = make_figure()
    r = make_result()
    evaluate([r], [fig])
    assert r.caption_overlap > 0.0
    assert r.completeness > 0.0
    assert r.composite_score > 0.0

def test_evaluate_composite_is_mean():
    fig = make_figure()
    r = make_result()
    evaluate([r], [fig])
    assert r.composite_score == round((r.caption_overlap + r.completeness) / 2, 4)

def test_evaluate_single_model_agreement_is_one():
    fig = make_figure()
    r = make_result()
    evaluate([r], [fig])
    assert r.cross_model_agreement == 1.0

def test_evaluate_two_models_agreement():
    fig = make_figure()
    r1 = make_result(model="a", answer="retention time peaks")
    r2 = make_result(model="b", answer="retention time spectrum")
    evaluate([r1, r2], [fig])
    assert r1.cross_model_agreement == r2.cross_model_agreement
    assert 0.0 < r1.cross_model_agreement <= 1.0

def test_evaluate_unknown_figure_zero_overlap():
    fig = make_figure()
    r = make_result()
    r.figure_id = "does_not_exist"
    evaluate([r], [fig])
    assert r.caption_overlap == 0.0

def test_evaluate_returns_same_list():
    fig = make_figure()
    results = [make_result()]
    returned = evaluate(results, [fig])
    assert returned is results
