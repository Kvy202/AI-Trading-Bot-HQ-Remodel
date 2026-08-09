from __future__ import annotations

import pytest

from tools import model_loss_balance_probe as probe


@pytest.mark.parametrize("kind,expected", [
    ("lstm", {"in_dim": 27, "hidden": 64, "layers": 2, "dropout": 0.1}),
    ("tcn", {"in_dim": 27, "hid": 64, "levels": 4, "kernel": 3, "dropout": 0.1}),
    ("tx", {"in_dim": 27, "d_model": 64, "nhead": 4, "nlayers": 2, "dropout": 0.1}),
])
def test_probe_uses_exact_full_production_candidate_architecture(kind, expected):
    model, config = probe.production_model(kind)
    assert config == expected
    assert model is not None


@pytest.fixture(scope="module")
def report():
    return probe.build_probe_report(seed=24201)


def test_probe_is_synthetic_only_and_cannot_freeze_real_weights(report):
    assert report["synthetic_only"] is True
    assert report["synthetic_balance_only"] is True
    assert report["final_weight_evidence"] is False
    assert report["eligible_to_freeze_real_weights"] is False
    assert report["real_balance_freeze_created"] is False
    assert report["real_balance_freeze_status"] == "pending_training_data"


def test_probe_reports_every_formulation_and_required_diagnostics(report):
    for kind, item in report["architectures"].items():
        assert item["exact_production_candidate_architecture"] is True
        assert set(item["formulations"]) == {
            "normalized_mse_fixed", "normalized_huber_fixed", "normalized_huber_training_balanced",
        }
        assert item["output_shapes"] == {"ret_reg": [8], "ret_cls_logits": [8, 2], "rv_reg": [8]}
        for form in item["formulations"].values():
            assert "weighted_gradient_ratios" in form
            assert "pairwise_gradient_cosines" in form
            assert "classification_projection" in form
            assert "gradient_clipping" in form


def test_probe_digest_is_deterministic_for_repeated_architecture_probe(report):
    first = probe.probe_architecture("lstm", seed=24201)
    second = probe.probe_architecture("lstm", seed=24201)
    assert first == second
    assert len(report["probe_digest"]) == 64


def test_phase24_1_reduced_probe_is_retained_and_distinguished(report):
    assert report["existing_probe_scope"] == "reduced_synthetic_architecture"
    assert report["this_probe_scope"] == "full_production_candidate_architectures_synthetic_data"
