"""Several policies merged into one .onnx (tools/policy_ensemble.py).

The scorecard grades A.onnx+B.onnx+C.onnx as its Ensemble; the car runs one
file. These hold the merged file to the scorecard's Ensemble output for
output, and decision for decision through the chunk blend, under the car's
own PolicyModel, and its sidecar to the members it was made of.

policy-ensemble.onnx is the merge of policy-per-gear.onnx and
policy-ens-m1/-m2.onnx (its untrained weights, rescaled): no training here.
Rebuilding it needs the `onnx` package, which only the training venv has.
"""

import json
import pathlib

import numpy as np
import pytest

from crawler.autopilot import policy as pl
import policy_ensemble as pens
import policy_scorecard as sc

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
MERGED = FIXTURES / "policy-ensemble.onnx"
MEMBERS = [FIXTURES / n for n in ("policy-per-gear.onnx", "policy-ens-m1.onnx",
                                  "policy-ens-m2.onnx")]


def feeds(model, n, seed=0):
    rng = np.random.default_rng(seed)
    shapes = {i.name: [d if isinstance(d, int) else 1 for d in i.shape]
              for i in model.session.get_inputs()}
    for _ in range(n):
        f = {k: rng.normal(0, 1, s).astype(np.float32) for k, s in shapes.items()}
        yield f["x"], f["prev"], f.get("mode")


def outputs_match(a, b, tol=1e-5):
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k].shape == b[k].shape, k
        np.testing.assert_allclose(a[k], b[k], rtol=tol, atol=tol, err_msg=k)


def test_the_merged_file_is_the_scorecards_ensemble_output_for_output():
    merged = pl.PolicyModel(str(MERGED))
    ens = sc.Ensemble(pl.PolicyModel(str(p)) for p in MEMBERS)
    one = pl.PolicyModel(str(MEMBERS[0]))
    differs = False
    for x, prev, mode in feeds(merged, 50):
        got = merged.run(x, prev, mode)
        outputs_match(got, ens.run(x, prev, mode))
        differs |= not np.allclose(got["throttle"], one.run(x, prev, mode)["throttle"])
    # The members are not one model three times over.
    assert differs


def test_the_merged_file_decides_as_the_scorecards_ensemble_through_the_blend():
    merged = pl.PolicyModel(str(MERGED))
    ens = sc.Ensemble(pl.PolicyModel(str(p)) for p in MEMBERS)
    assert (merged.chunk, merged.ensemble_m, merged.has_mode, merged.steer_head) == \
        (ens.chunk, ens.ensemble_m, ens.has_mode, ens.steer_head)
    blends = [pl.ChunkEnsembler(m.chunk, m.ensemble_m, m.ensemble_link_s)
              for m in (merged, ens)]
    for j, (x, prev, mode) in enumerate(feeds(merged, 40, seed=1)):
        got = [pl.read_output(b.apply(m.run(x, prev, mode), 100.0 + 0.1 * j, None))[0]
               for b, m in zip(blends, (merged, ens))]
        assert got[0][0] == got[1][0] and got[0][3] == got[1][3]
        np.testing.assert_allclose(got[0][1:3], got[1][1:3], rtol=1e-5, atol=1e-5)


def test_the_merged_sidecar_names_its_members_and_keeps_none_of_their_scores():
    meta = json.loads(MERGED.with_suffix(".json").read_text())
    members = meta["ensemble"]["members"]
    assert [m["onnx"] for m in members] == [p.name for p in MEMBERS]
    assert [m["onnx_sha256"] for m in members] == [pens.file_sha256(str(p)) for p in MEMBERS]
    assert pl.PolicyModel(str(MERGED)).sha256 not in [m["onnx_sha256"] for m in members]


def test_the_sidecar_is_the_first_members_with_the_union_of_sessions(tmp_path):
    paths = []
    for i, sessions in enumerate((["b", "a"], ["c", "a"])):
        p = tmp_path / f"m{i}.onnx"
        p.write_bytes(bytes([i]))
        meta = {**json.loads(MEMBERS[0].with_suffix(".json").read_text()),
                "train_sessions": sessions, "seed": i, "parameters": 10,
                "created": f"2026-10-30T0{i}:00:00Z", "cv": {"mean": {}}, "val": {}}
        p.with_suffix(".json").write_text(json.dumps(meta))
        paths.append((str(p), meta))
    meta = pens.merged_sidecar([m for _, m in paths], [p for p, _ in paths])
    assert meta["train_sessions"] == ["a", "b", "c"]
    assert meta["seed"] == [0, 1] and meta["parameters"] == 20
    assert meta["created"] == "2026-10-30T01:00:00Z"
    assert "cv" not in meta and "val" not in meta
    assert meta["steer_head"] == paths[0][1]["steer_head"]


def test_members_that_differ_in_the_contract_are_refused():
    metas = [json.loads(p.with_suffix(".json").read_text())
             for p in (MEMBERS[0], FIXTURES / "policy-chunk5.onnx")]
    with pytest.raises(ValueError, match="steer_head"):
        pens.merged_sidecar(metas, [str(MEMBERS[0]), str(FIXTURES / "policy-chunk5.onnx")])


def test_merging_the_members_again_makes_the_committed_file(tmp_path):
    pytest.importorskip("onnx", reason="onnx is in the training venv only")
    out = tmp_path / "again.onnx"
    pens.merge(str(out), [str(p) for p in MEMBERS])
    again, committed = pl.PolicyModel(str(out)), pl.PolicyModel(str(MERGED))
    for x, prev, mode in feeds(again, 10, seed=2):
        outputs_match(again.run(x, prev, mode), committed.run(x, prev, mode), tol=0)
    with pytest.raises(SystemExit, match="already exists"):
        pens.merge(str(out), [str(p) for p in MEMBERS])


def test_members_whose_graphs_differ_are_refused(tmp_path):
    onnx = pytest.importorskip("onnx", reason="onnx is in the training venv only")
    with pytest.raises(ValueError, match="outputs differ"):
        pens.merge_graphs([onnx.load(str(MEMBERS[0])),
                           onnx.load(str(FIXTURES / "policy-chunk5.onnx"))])
