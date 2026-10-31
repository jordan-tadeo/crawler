"""Several trained policies as ONE .onnx the car runs like any other.

    python tools/policy_ensemble.py OUT.onnx A.onnx B.onnx [C.onnx ...]

Writes OUT.onnx and OUT.json. The graph is every member's graph side by side
on the same inputs (each member's names under m0/, m1/, ..), and its outputs
are the scorecard's Ensemble (tools/policy_scorecard.py) worked out inside
the graph:

  logits      each member's gear softmax per chunk offset, averaged, and
              handed on as its log (+1e-9), so read_output and the
              ChunkEnsembler take it as logits, as the scorecard's does
  the rest    throttle, steer_*, mode_logits: the plain mean of the members'
              raw outputs (mode_logits too: logits averaged, not softmaxes)

So `A.onnx+B.onnx+C.onnx` on the scorecard and OUT.onnx on its own score the
same to float32 noise (the scorecard's softmax is float64), and OUT.onnx has
one sha256, which is what the registry, the release gate's scorecard row and
the car's /footprint rows key on. Nothing on the car knows it is an ensemble:
its cost is the members' run one after another, in one session.

The sidecar is the first member's, with the scorecard's Ensemble.meta rule
for train_sessions (the union, sorted), the members recorded under
"ensemble", "parameters" summed, "created" the newest member's, and the
first member's own fold scores ("cv", "val") left out: they were never the
ensemble's. The members must share every key the car's runtime depends on
(models.contract_of) and the same inputs and outputs, which is checked.

Needs the `onnx` package: the training venv (tools/requirements-train.txt),
not the car's.
"""

import argparse
import copy
import hashlib
import json
import os
import shlex
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crawler.autopilot import policy as pl  # noqa: E402
import models as mdl                        # noqa: E402

EPS = 1e-9          # the scorecard's Ensemble: log(p + 1e-9)
LEFT_OUT = ("cv", "val")


def prefix(i):
    return f"m{i}/"


def _io(values):
    return [(v.name, v.type.SerializeToString()) for v in values]


def renamed_graph(graph, pre, inputs):
    """graph's nodes and initializers with every name but its inputs under
    pre. -> (nodes, initializers)."""
    keep = set(inputs)

    def name(n):
        return n if (n in keep or n == "") else pre + n

    nodes = []
    for node in graph.node:
        n = copy.deepcopy(node)
        if any(a.type in (a.GRAPH, a.GRAPHS) for a in n.attribute):
            raise ValueError(f"{node.op_type} has a subgraph: not merged here")
        n.name = pre + (n.name or n.op_type)
        n.input[:] = [name(x) for x in n.input]
        n.output[:] = [name(x) for x in n.output]
        nodes.append(n)
    inits = []
    for init in graph.initializer:
        t = copy.deepcopy(init)
        t.name = name(t.name)
        inits.append(t)
    return nodes, inits


def merge_graphs(members):
    """onnx ModelProtos -> one ModelProto whose outputs are their Ensemble."""
    import onnx
    from onnx import helper, numpy_helper

    first = members[0]
    for i, m in enumerate(members[1:], 1):
        if _io(m.graph.input) != _io(first.graph.input):
            raise ValueError(f"member {i}'s inputs differ from member 0's")
        if _io(m.graph.output) != _io(first.graph.output):
            raise ValueError(f"member {i}'s outputs differ from member 0's")
        if list(m.opset_import) != list(first.opset_import):
            raise ValueError(f"member {i}'s opset differs from member 0's")
    inputs = [v.name for v in first.graph.input]
    outputs = [v.name for v in first.graph.output]
    if "logits" not in outputs:
        raise ValueError("the members have no logits output")

    nodes, inits = [], []
    for i, m in enumerate(members):
        n, t = renamed_graph(m.graph, prefix(i), inputs)
        nodes += n
        inits += t
    inits.append(numpy_helper.from_array(np.array([-1, len(pl.ACTIONS)], np.int64),
                                         "ens/gear_rows"))
    inits.append(numpy_helper.from_array(np.array(EPS, np.float32), "ens/eps"))

    k = range(len(members))
    for out in outputs:
        if out == "logits":
            probs = []
            for i in k:
                rows = f"ens/logits_rows_{i}"
                nodes.append(helper.make_node("Reshape", [prefix(i) + out, "ens/gear_rows"],
                                              [rows], name=f"ens/reshape_{i}"))
                probs.append(f"ens/probs_{i}")
                nodes.append(helper.make_node("Softmax", [rows], [probs[-1]], axis=-1,
                                              name=f"ens/softmax_{i}"))
            nodes += [
                helper.make_node("Mean", probs, ["ens/probs"], name="ens/mean_probs"),
                helper.make_node("Add", ["ens/probs", "ens/eps"], ["ens/probs_eps"],
                                 name="ens/add_eps"),
                helper.make_node("Log", ["ens/probs_eps"], ["ens/log_rows"], name="ens/log"),
                helper.make_node("Shape", [prefix(0) + out], ["ens/logits_shape"],
                                 name="ens/logits_shape"),
                helper.make_node("Reshape", ["ens/log_rows", "ens/logits_shape"], [out],
                                 name="ens/reshape_back"),
            ]
        else:
            nodes.append(helper.make_node("Mean", [prefix(i) + out for i in k], [out],
                                          name=f"ens/mean_{out}"))

    graph = helper.make_graph(nodes, f"ensemble{len(members)}", list(first.graph.input),
                              list(first.graph.output), inits)
    model = helper.make_model(graph, opset_imports=list(first.opset_import),
                              producer_name="policy_ensemble")
    model.ir_version = first.ir_version
    onnx.checker.check_model(model)
    return model


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def merged_sidecar(metas, paths, command=None):
    """The sidecar above, from the members' sidecars and their .onnx paths."""
    first = metas[0]
    for i, m in enumerate(metas[1:], 1):
        diff = mdl.contract_diff(first, m)
        if diff:
            raise ValueError(f"member {i} differs from member 0 in {diff}")
    meta = {k: v for k, v in first.items() if k not in LEFT_OUT}
    meta["train_sessions"] = sorted(set().union(*(m.get("train_sessions", []) for m in metas)))
    created = [m["created"] for m in metas if m.get("created")]
    if created:
        meta["created"] = max(created)
    params = [m.get("parameters") for m in metas]
    if all(isinstance(p, int) for p in params):
        meta["parameters"] = sum(params)
    meta["seed"] = [m.get("seed") for m in metas]
    if command is not None:
        meta["train_command"] = command
    meta["ensemble"] = {
        "rule": "gear softmax averaged per chunk offset, as its log (+1e-9); "
                "every other output averaged",
        "members": [{"onnx": os.path.basename(p),
                     "onnx_sha256": file_sha256(p),
                     "json_sha256": file_sha256(os.path.splitext(p)[0] + ".json"),
                     "seed": m.get("seed"),
                     "created": m.get("created"),
                     "train_command": m.get("train_command")}
                    for p, m in zip(paths, metas)],
    }
    return meta


def merge(out_path, paths, command=None):
    """Write out_path (.onnx) and its sidecar from the member .onnx paths."""
    import onnx

    if len(paths) < 2:
        raise ValueError("an ensemble needs at least two members")
    json_out = os.path.splitext(out_path)[0] + ".json"
    for p in (out_path, json_out):
        if os.path.exists(p):
            raise SystemExit(f"{p} already exists; nothing is overwritten")
    metas = []
    for p in paths:
        with open(os.path.splitext(p)[0] + ".json") as f:
            metas.append(json.load(f))
    meta = merged_sidecar(metas, paths, command)
    model = merge_graphs([onnx.load(p) for p in paths])
    onnx.save(model, out_path)
    mdl.write_verified_json(meta, json_out)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("members", nargs="+")
    args = ap.parse_args(argv)
    if not args.out.endswith(".onnx"):
        ap.error("OUT must be a .onnx")
    command = "tools/policy_ensemble.py " + " ".join(
        shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    merge(args.out, args.members, command)
    print(f"{args.out}: the ensemble of {len(args.members)} "
          f"({file_sha256(args.out)[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
