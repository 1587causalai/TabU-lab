"""Reproducible component checks; no training, receipt issuance, or model claim.

Schema v1 versions the JSON envelope, not a frozen suite of probes. The source
digest includes this module and binds the actual implementation and checks.
Consumers must inspect both that digest and the emitted check names/outcomes;
different source digests are different verification identities, even under v1.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .answers import CategoricalAnswers, NumericAnswers
from .losses import encoding_mse
from .readout import RestorationReadout, unit_kernel_logits


def numeric_example() -> dict[str, float]:
    """Uniform evidence (c,x)=(0,1),(1,3),(2,5); predict at c=3.

    Robust coordinates: median 3, half-IQR 1, scale 1, so the encoding is
    (-2, 0, 2); the affine decode keeps the hand-computed answers exact.
    """
    codec = NumericAnswers.from_visible(torch.tensor([1.0, 3.0, 5.0]), epsilon=1e-6)
    logits = torch.zeros(1, 3, dtype=torch.float64)
    source = torch.arange(3, dtype=torch.float64)[:, None]
    target = torch.tensor([[3.0]], dtype=torch.float64)
    outputs = {}
    for mode, expected in (("nw", 3.0), ("ll", 17 / 3)):
        result = RestorationReadout(mode, ridge=1 / 3)(
            logits, torch.arange(3), codec.encoded, support_cells=source, target_cells=target
        )
        actual = codec.decode(result.encoding)[0]
        torch.testing.assert_close(actual, actual.new_tensor(expected))
        outputs[mode] = float(actual)
    return outputs


def check_normal_equations() -> None:
    gen = torch.Generator().manual_seed(27)
    support = torch.randn(4, 3, generator=gen, dtype=torch.float64)
    targets = torch.randn(2, 3, generator=gen, dtype=torch.float64)
    logits = torch.randn(2, 4, generator=gen, dtype=torch.float64)
    for width in (1, 32, 128):
        answers = torch.randn(4, width, generator=gen, dtype=torch.float64)
        result = RestorationReadout("ll", ridge=0.2)(
            logits, torch.arange(4), answers, support_cells=support, target_cells=targets
        )
        references = []
        for target, weights in zip(targets, logits.softmax(-1), strict=True):
            # Independent augmented (d+1)-dimensional normal equations, all RHS.
            design = torch.cat((torch.ones(4, 1, dtype=torch.float64), support - target), dim=1)
            penalty = torch.diag(torch.tensor([0.0, 0.2, 0.2, 0.2], dtype=torch.float64))
            solution = torch.linalg.solve(
                design.T @ (weights[:, None] * design) + penalty,
                design.T @ (weights[:, None] * answers),
            )
            references.append(solution[0])
        torch.testing.assert_close(result.encoding, torch.stack(references), atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(result.coefficients.sum(-1), torch.ones(2, dtype=torch.float64))


def categorical_example(width: int) -> CategoricalAnswers:
    codebook = torch.zeros(2, width, dtype=torch.float64)
    codebook[0, :8] = 1
    codebook[1, 8:16] = 1
    return CategoricalAnswers.from_visible(
        torch.tensor([0, 1, 0]), torch.tensor([0, 1]), codebook, domain_size=3
    )


def check_gradients() -> None:
    gen = torch.Generator().manual_seed(19)
    tensors = [
        torch.randn(*shape, generator=gen, dtype=torch.float64).requires_grad_()
        for shape in ((2, 2), (3, 2), (2, 2), (3, 2))
    ]
    numeric = NumericAnswers.from_visible(torch.tensor([1.0, -0.5, 2.0]), epsilon=0.01)
    for mode in ("nw", "ll"):
        for codec in (numeric, categorical_example(32), categorical_example(128)):
            truth = codec.encode_targets(
                torch.tensor([0.2, 1.3]) if codec is numeric else torch.tensor([1, 0])
            )

            def loss(tu, su, tc, sc, mode=mode, codec=codec, truth=truth):
                logits = unit_kernel_logits(tu, su, bandwidth=1.3)
                result = RestorationReadout(mode, ridge=0.3)(
                    logits, torch.arange(3), codec.encoded, support_cells=sc, target_cells=tc
                )
                return encoding_mse(result.encoding, truth)

            if not torch.autograd.gradcheck(loss, tuple(tensors), atol=1e-5, rtol=1e-4):
                raise AssertionError("Unit/Cell answer-code MSE finite-difference check failed")


def check_answer_codes() -> None:
    for width in (32, 128):
        codec = categorical_example(width)
        truth = codec.encode_targets(torch.tensor([0, 1]))
        torch.testing.assert_close(encoding_mse(truth, truth), torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(codec.decode(truth), torch.tensor([0, 1]))
        # Equal distance resolves by schema order, including an absent declared class.
        torch.testing.assert_close(codec.decode(truth.mean(0, keepdim=True)), torch.tensor([0]))
        try:
            codec.encode_targets(torch.tensor([0, 2]))
        except ValueError as error:
            if "no-answer-code" not in str(error):
                raise
        else:
            raise AssertionError("unknown truth class must fail without inventing an answer code")


def check_robust_numeric_coordinates() -> dict[str, float]:
    """Median/half-IQR coordinates: quantile convention, tail stability, floors."""
    base = NumericAnswers.from_visible(
        torch.arange(8.0, 17.0, dtype=torch.float64), epsilon=1e-9
    )
    torch.testing.assert_close(base.median, base.median.new_tensor(12.0))
    torch.testing.assert_close(base.scale, base.scale.new_tensor(2.0))
    tailed = torch.arange(8.0, 17.0, dtype=torch.float64)
    tailed[-1] = 16000.0
    moved = NumericAnswers.from_visible(tailed, epsilon=1e-9)
    torch.testing.assert_close(moved.median, base.median, rtol=0, atol=0)
    torch.testing.assert_close(moved.scale, base.scale, rtol=0, atol=0)
    # Fixed quantile convention: two points interpolate Q(1/4), Q(3/4) linearly.
    pair = NumericAnswers.from_visible(torch.tensor([0.0, 10.0], dtype=torch.float64), epsilon=1e-9)
    torch.testing.assert_close(pair.median, pair.median.new_tensor(5.0))
    torch.testing.assert_close(pair.scale, pair.scale.new_tensor(2.5))
    # Constant columns and single supports use the declared floor, not std.
    flat = NumericAnswers.from_visible(torch.zeros(4, dtype=torch.float64), epsilon=0.02)
    torch.testing.assert_close(flat.scale, flat.scale.new_tensor(0.02))
    torch.testing.assert_close(flat.encoded, torch.zeros(4, 1, dtype=torch.float64))
    one = NumericAnswers.from_visible(torch.tensor([7.0], dtype=torch.float64), epsilon=0.02)
    torch.testing.assert_close(one.scale, one.scale.new_tensor(0.02))
    # Encode/decode round trip on the shared coordinate.
    torch.testing.assert_close(base.decode(base.encoded), torch.arange(8.0, 17.0, dtype=torch.float64))
    return {"median": float(base.median), "half_iqr_scale": float(base.scale)}


def check_nearest_code_stability() -> None:
    """Large LL extrapolation and shared coordinates must not create false ties."""
    for width in (32, 128):
        book = torch.zeros(2, width, dtype=torch.float64)
        book[:, :7] = 1
        book[0, 7] = 1
        book[1, 8] = 1
        codec = CategoricalAnswers.from_visible(
            torch.tensor([0, 1]), torch.tensor([0, 1]), book, domain_size=2
        )
        extrapolated = (1 - 1e17) * book[0] + 1e17 * book[1]
        shared_large = book[1].clone()
        shared_large[:7] = 1e17
        predictions = torch.stack((extrapolated, shared_large))
        torch.testing.assert_close(codec.decode(predictions), torch.tensor([1, 1]))

        book = torch.zeros(3, width, dtype=torch.float64)
        book[0, 7] = 1
        book[0, 9:16] = 1
        book[1, :8] = 1
        book[2, :7] = 1
        book[2, 8] = 1
        codec = CategoricalAnswers.from_visible(
            torch.tensor([2, 0, 1]), torch.arange(3), book, domain_size=3
        )
        result = RestorationReadout("ll", ridge=1)(
            torch.tensor([[0.0, -40.0, -40.0]], dtype=torch.float64),
            torch.arange(3),
            codec.encoded,
            support_cells=torch.tensor([[0.0], [-1.0], [1.0]], dtype=torch.float64),
            target_cells=torch.tensor([[1e33]], dtype=torch.float64),
        )
        torch.testing.assert_close(codec.decode(result.encoding), torch.tensor([2]))


def check_rotation_lift_mse() -> None:
    """sqrt(d/p) times an isometric lift preserves coordinate-mean MSE and gradients."""
    gen = torch.Generator().manual_seed(61)
    logits, cells, targets = [
        torch.randn(*shape, generator=gen, dtype=torch.float64).requires_grad_()
        for shape in ((2, 3), (3, 2), (2, 2))
    ]
    numeric = NumericAnswers.from_visible(torch.tensor([1.0, -0.5, 2.0]), epsilon=0.01)
    for codec in (numeric, categorical_example(32)):
        width = codec.encoded.shape[1]
        rotation = torch.linalg.qr(torch.randn(128, width, generator=gen, dtype=torch.float64)).Q
        lift = rotation * (128 / width) ** 0.5
        truth = codec.encode_targets(
            torch.tensor([0.2, 1.3]) if codec is numeric else torch.tensor([1, 0])
        )
        for mode in ("nw", "ll"):
            readout = RestorationReadout(mode, ridge=0.3)

            def predict(answers, readout=readout):
                return readout(
                    logits, torch.arange(3), answers, support_cells=cells, target_cells=targets
                ).encoding

            raw_loss = encoding_mse(predict(codec.encoded), truth)
            lifted_loss = encoding_mse(predict(codec.encoded @ lift.T), truth @ lift.T)
            torch.testing.assert_close(lifted_loss, raw_loss, atol=1e-12, rtol=1e-12)
            raw_grads = torch.autograd.grad(
                raw_loss.sum(), (logits, cells, targets), allow_unused=True
            )
            lift_grads = torch.autograd.grad(
                lifted_loss.sum(), (logits, cells, targets), allow_unused=True
            )
            for raw, lifted in zip(raw_grads, lift_grads, strict=True):
                if raw is None:
                    assert lifted is None
                else:
                    torch.testing.assert_close(lifted, raw, atol=1e-12, rtol=1e-12)


def verify_components() -> dict:
    checks = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, probe in (
            ("hand_computed_numeric_nw_ll", numeric_example),
            ("robust_median_half_iqr_coordinates", check_robust_numeric_coordinates),
            ("scalar_32d_128d_augmented_normal_equations", check_normal_equations),
            ("numeric_categorical_encoding_mse_finite_differences", check_gradients),
            ("32d_128d_answer_codes_nearest_class_and_unknown_truth", check_answer_codes),
            ("32d_128d_nearest_code_large_finite_stability", check_nearest_code_stability),
            ("normalized_rotation_lift_mse_and_gradient_equivalence", check_rotation_lift_mse),
        ):
            try:
                detail = probe()
                checks.append({"name": name, "outcome": "passed", "detail": detail})
            except (AssertionError, RuntimeError, ValueError, FloatingPointError) as error:
                checks.append({"name": name, "outcome": "failed", "error": str(error)})
    finally:
        torch.set_num_threads(previous_threads)
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return {
        "schema_version": "tabu.restoration.component-check.v1",
        "status": "local_unissued",
        "scope": "answer codecs, encoded readout and per-target encoding MSE only",
        "outcome": "passed" if all(c["outcome"] == "passed" for c in checks) else "failed",
        "component_source_sha256": digest.hexdigest(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "checks": checks,
    }


def verify_model() -> dict:
    """Run components plus bounded all-variant and update/resume model probes."""
    from .end_to_end_checks import check_model_variants, check_optimizer_continuation

    result = verify_components()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, probe in (
            ("sixteen_model_variants_forward_backward_and_damage", check_model_variants),
            ("two_update_serialized_optimizer_continuation", check_optimizer_continuation),
        ):
            try:
                result["checks"].append({"name": name, "outcome": "passed", "detail": probe()})
            except (AssertionError, RuntimeError, ValueError, FloatingPointError) as error:
                result["checks"].append({"name": name, "outcome": "failed", "error": str(error)})
    finally:
        torch.set_num_threads(previous_threads)
    result["schema_version"] = "tabu.restoration.model-check.v1"
    result["scope"] = (
        "five-step CPU reference correctness; not F0 fit, benchmark, or issued evidence"
    )
    result["outcome"] = (
        "passed" if all(c["outcome"] == "passed" for c in result["checks"]) else "failed"
    )
    return result
