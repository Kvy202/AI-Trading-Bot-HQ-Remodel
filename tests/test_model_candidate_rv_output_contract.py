from __future__ import annotations

import json

import pytest

from ml_dl.dl_models import TemporalConvNet, TinyLSTM, TinyTransformer
from tools import model_candidate_train as train


MODEL_CASES = (
    ("lstm", TinyLSTM),
    ("tcn", TemporalConvNet),
    ("tx", TinyTransformer),
)


def _old_forward(model, kind, values):
    if kind == "lstm":
        _, (hidden, _) = model.lstm(values)
        state = hidden[-1]
        return {
            "ret_reg": model.head_ret(state).squeeze(-1),
            "rv_reg": model.head_rv(state).squeeze(-1),
            "ret_cls_logits": model.head_cls(state),
        }
    if kind == "tcn":
        state = model.net(values.transpose(1, 2))[:, :, -1]
    else:
        state = model.enc(model.proj(values))[:, -1, :]
    return {
        "ret_reg": model.head_ret_reg(state).squeeze(-1),
        "ret_cls_logits": model.head_ret_cls(state),
        "rv_reg": model.head_rv_reg(state).squeeze(-1),
    }


def _force_negative_raw_rv(model, kind):
    torch = train._torch()
    head = model.head_rv[-1] if kind == "lstm" else model.head_rv_reg
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-2.0)


@pytest.mark.parametrize(("kind", "model_class"), MODEL_CASES)
def test_default_identity_strictly_loads_incumbent_and_reproduces_old_forward(kind, model_class):
    torch = train._torch()
    config = train.ARCHITECTURE_DEFAULTS[kind]
    incumbent_state = torch.load(
        train.BASE_DIR / "model_artifacts" / f"dl_{kind}_latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = model_class(**config).cpu().eval()

    result = model.load_state_dict(incumbent_state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert model.rv_output_transform == "identity"
    assert list(model.state_dict()) == list(incumbent_state)
    assert {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in incumbent_state.items()
    }

    torch.manual_seed(25001)
    values = torch.randn(2, 64, 27)
    with torch.no_grad():
        expected = _old_forward(model, kind, values)
        actual = model(values)
    assert all(torch.equal(actual[name], expected[name]) for name in expected)


@pytest.mark.parametrize(("kind", "model_class"), MODEL_CASES)
def test_softplus_changes_only_rv_and_state_dict_is_strictly_compatible(kind, model_class):
    torch = train._torch()
    functional = torch.nn.functional
    config = train.ARCHITECTURE_DEFAULTS[kind]
    torch.manual_seed(25002)
    identity = model_class(**config, rv_output_transform="identity").cpu().eval()
    _force_negative_raw_rv(identity, kind)
    identity_state = identity.state_dict()
    softplus = model_class(**config, rv_output_transform="softplus").cpu().eval()

    result = softplus.load_state_dict(identity_state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert list(softplus.state_dict()) == list(identity_state)
    assert all(
        softplus.state_dict()[name].shape == identity_state[name].shape
        for name in identity_state
    )

    values = torch.randn(3, 64, 27)
    with torch.no_grad():
        identity_output = identity(values)
        softplus_output = softplus(values)

    assert torch.equal(softplus_output["ret_cls_logits"], identity_output["ret_cls_logits"])
    assert torch.equal(softplus_output["ret_reg"], identity_output["ret_reg"])
    assert torch.equal(softplus_output["rv_reg"], functional.softplus(identity_output["rv_reg"]))
    assert not torch.equal(softplus_output["rv_reg"], identity_output["rv_reg"])
    assert torch.isfinite(softplus_output["rv_reg"]).all()
    assert (softplus_output["rv_reg"] > 0).all()


@pytest.mark.parametrize(("kind", "model_class"), MODEL_CASES)
def test_unknown_rv_output_transform_is_rejected(kind, model_class):
    with pytest.raises(ValueError, match="unsupported rv_output_transform"):
        model_class(**train.ARCHITECTURE_DEFAULTS[kind], rv_output_transform="clamp")


@pytest.mark.parametrize("kind", train.ALLOWED_KINDS)
def test_new_candidate_factory_defaults_softplus_but_allows_explicit_identity(kind):
    assert train.make_candidate_model(kind).rv_output_transform == "softplus"
    assert train.make_candidate_model(
        kind, {"rv_output_transform": "identity"}
    ).rv_output_transform == "identity"


@pytest.mark.parametrize("kind", train.ALLOWED_KINDS)
def test_new_candidate_architecture_contract_is_explicit_and_deterministic(kind):
    contract = train.architecture_contract(kind)
    rv_contract = contract["rv_output_contract"]
    digest = rv_contract["rv_output_contract_digest"]
    payload = {key: value for key, value in rv_contract.items() if key != "rv_output_contract_digest"}

    assert contract["architecture_mathematics_modified"] is True
    assert contract["constructor"]["rv_output_transform"] == "softplus"
    assert contract["rv_output_transform"] == "softplus"
    assert contract["rv_output_support"] == "strictly_positive"
    assert contract["post_hoc_rv_clipping_applied"] is False
    assert contract["rv_output_contract_digest"] == digest
    assert digest == train.json_digest(payload)
    assert len(digest) == 64
    assert json.dumps(rv_contract, sort_keys=True) == json.dumps(
        train.candidate_rv_output_contract(), sort_keys=True
    )
