"""
Phase 7: 50 scientific questions + 25 multi-hop problems.

Every item has an answer key that is NOT produced by the component under test:
  - discovery/parameter keys come from the dataset generator's known truth;
  - KG/compiler/complexity keys are written down from the corpus documents'
    own frontmatter and elementary calculus, then compared with the system's output;
  - retrieval keys are the dataset's true mechanism family.

Item format: {id, category, hops, question, expected, answer, correct, detail}.

Single questions (50)
  Q01-12 structure identification on 12 datasets (full pipeline)
  Q13-24 parameter estimation on the same datasets (within phase7.param_tol)
  Q25-29 KG: parameters of each documented mechanism
  Q30-34 compiler: derivatives each documented equation requires
  Q35-39 KG provenance: which document supports each mechanism
  Q40-44 complexity: which of two equations is simpler
  Q45-50 retrieval: is the true mechanism among those retrieved (recall) for 6 datasets
Multi-hop (25)
  M01-12 data -> description -> RAG -> KG templates -> validation -> selection -> forward OOD prediction
  M13-17 observation text -> RAG -> KG mechanism -> KG parameters -> compiler -> required derivatives
  M18-21 parameter -> KG (HAS_PARAMETER, reversed) -> equations -> compiler -> union of derivatives
  M22-25 mechanism -> KG equation -> nesting test (does it reduce to diffusion when a parameter -> 0?)
"""

from typing import Dict, List

import sympy

from evaluation.engine import build_pool, family_of, score_run, validate_and_select
from hypotheses.complexity import equation_complexity
from hypotheses.equation_compiler import parse_equation
from hypotheses.validation import canonical_signature
from knowledge_graph.query import documents_supporting, equations_for_mechanism, find_entity_by_name, parameters_for_mechanism
from knowledge_graph.schema import EntityType, RelationType
from llm.schemas import Hypothesis

MECHS = ["diffusion", "advection", "reaction", "reaction-diffusion", "advection-diffusion"]

# Answer keys written from the corpus frontmatter / calculus, independent of the system.
KEY_PARAMS = {"diffusion": {"D"}, "advection": {"c"}, "reaction": {"r", "K"},
              "reaction-diffusion": {"D", "r", "K"}, "advection-diffusion": {"c", "D"}}
KEY_DERIVS = {"diffusion": {"u_t", "u_xx"}, "advection": {"u_t", "u_x"}, "reaction": {"u_t"},
              "reaction-diffusion": {"u_t", "u_xx"}, "advection-diffusion": {"u_t", "u_x", "u_xx"}}
KEY_DOC = {"diffusion": "doc_diffusion", "advection": "doc_advection", "reaction": "doc_reaction",
           "reaction-diffusion": "doc_reaction_diffusion", "advection-diffusion": "doc_advection_diffusion"}
KEY_SIMPLER = [  # (a, b, simpler)
    ("u_t = D*u_xx", "u_t = D*u_xx + r*u*(1-u/K)", "a"),
    ("u_t + c*u_x = D*u_xx", "u_t = D*u_xx", "b"),
    ("u_t = r*u*(1-u/K)", "u_t = D*u_xx + r*u*(1-u/K)", "a"),
    ("u_t + c*u_x = 0", "u_t + c*u_x = D*u_xx", "a"),
    ("u_t = D*u_xx + k*u", "u_t = D*u_xx", "b"),
]
KEY_NESTS_DIFFUSION = {"advection": False, "reaction": False, "reaction-diffusion": True, "advection-diffusion": True}
# Generic signature phrases written independently of the corpus text.
SIGNATURE_TEXT = {
    "diffusion": "the profile spreads out and flattens in place over time and its amplitude decays, without moving",
    "advection": "the whole profile is carried across the domain at constant speed and keeps its shape",
    "reaction": "each point grows or decays on its own toward a saturation level with no spatial coupling",
    "reaction-diffusion": "the profile spreads out while also growing locally toward a carrying capacity",
    "advection-diffusion": "the profile drifts across the domain while also spreading out and flattening",
}


def _item(id_, cat, hops, q, expected, answer, correct, detail=None):
    return {"id": id_, "category": cat, "hops": hops, "question": q, "expected": expected,
            "answer": answer, "correct": bool(correct), "detail": detail or {}}


def _hyp(eq):
    names = sorted({str(s) for s in sympy.sympify(eq.replace("=", "-(") + ")").free_symbols}
                   - {"u", "u_t", "u_x", "u_xx", "u_xxx"})
    return Hypothesis(id="q", name="q", equation=eq, parameters=[{"name": n} for n in names])


def run_single_questions(datasets, res, cfg) -> List[Dict]:
    p7 = cfg["phase7"]
    items, runs = [], {}
    for i, ds in enumerate(datasets, start=1):
        pool, _, _, _ = build_pool(ds.x, ds.t, ds.U, res)
        run = validate_and_select(pool, ds.x, ds.t, ds.U, cfg, seed=p7["seed"])
        s = score_run(ds, run, p7["ood_t_end"])
        runs[ds.name] = (run, s)
        items.append(_item(f"Q{i:02d}", "structure_identification", 1,
                           f"Which governing equation structure explains dataset '{ds.name}'?",
                           ds.family, s["selected_family"], s["correct_model"],
                           {"selected_equation": run["selected_hypothesis"].equation if run["selected_hypothesis"] else None}))
    for i, ds in enumerate(datasets, start=13):
        run, s = runs[ds.name]
        err = s["max_rel_param_error"]
        items.append(_item(f"Q{i:02d}", "parameter_estimation", 1,
                           f"Estimate the physical parameters of dataset '{ds.name}'.",
                           ds.truth, s["selected_params"], err is not None and err <= p7["param_tol"],
                           {"max_rel_error": err}))
    g = res.graph
    for i, m in enumerate(MECHS, start=25):
        ans = {p.name for p in parameters_for_mechanism(g, m)}
        items.append(_item(f"Q{i:02d}", "kg_parameters", 1, f"Which parameters appear in the documented equation for {m}?",
                           sorted(KEY_PARAMS[m]), sorted(ans), ans == KEY_PARAMS[m]))
    for i, m in enumerate(MECHS, start=30):
        eq = equations_for_mechanism(g, m)[0].name
        _, fields, _, _ = parse_equation(eq, list(KEY_PARAMS[m]))
        ans = {f for f in fields if f != "u"}
        items.append(_item(f"Q{i:02d}", "equation_derivatives", 1, f"Which derivatives does '{eq}' require?",
                           sorted(KEY_DERIVS[m]), sorted(ans), ans == KEY_DERIVS[m]))
    for i, m in enumerate(MECHS, start=35):
        mech = find_entity_by_name(g, EntityType.MECHANISM, m)
        docs = [r.sources[0].document_id for r in g.relationships_for(mech.entity_id, RelationType.SUPPORTED_BY)]
        items.append(_item(f"Q{i:02d}", "kg_provenance", 1, f"Which source document supports the mechanism '{m}'?",
                           KEY_DOC[m], docs, docs == [KEY_DOC[m]]))
    for i, (a, b, key) in enumerate(KEY_SIMPLER, start=40):
        ca, cb = equation_complexity(_hyp(a))["complexity_score"], equation_complexity(_hyp(b))["complexity_score"]
        ans = "a" if ca < cb else "b"
        items.append(_item(f"Q{i:02d}", "complexity", 1, f"Which is simpler: (a) {a}  or  (b) {b}?",
                           key, ans, ans == key, {"complexity_a": ca, "complexity_b": cb}))
    for i, ds in enumerate([datasets[k] for k in (0, 3, 6, 7, 9, 11)], start=45):
        _, _, ctx, _ = build_pool(ds.x, ds.t, ds.U, res)
        mechs = []
        for e in ctx.retrieved_evidence:
            for m in e.mechanisms:
                if m not in mechs:
                    mechs.append(m)
        items.append(_item(f"Q{i:02d}", "retrieval", 1,
                           f"Is the true mechanism of '{ds.name}' among the retrieved evidence (top-5 chunks)?",
                           ds.family, mechs, ds.family in mechs,
                           {"top1_mechanism": mechs[0] if mechs else None, "top1_correct": bool(mechs and mechs[0] == ds.family)}))
    return items, runs


def run_multihop(datasets, res, cfg, runs) -> List[Dict]:
    p7 = cfg["phase7"]
    g = res.graph
    items = []
    for i, ds in enumerate(datasets, start=1):
        run, s = runs[ds.name]
        ood = s["ood_rel_rmse_t_1_to_end"]
        ok = s["correct_model"] and ood is not None and ood <= p7["ood_tol"]
        items.append(_item(f"M{i:02d}", "data_to_prediction", 6,
                           f"From observations of '{ds.name}': retrieve evidence, build candidates from the KG, "
                           f"validate, select, and predict u on t in (1.0, {p7['ood_t_end']}] (beyond the data).",
                           {"family": ds.family, "ood_rel_rmse_max": p7["ood_tol"]},
                           {"family": s["selected_family"], "ood_rel_rmse": ood}, ok))
    for i, m in enumerate(MECHS, start=13):
        ev = res.retriever.retrieve(SIGNATURE_TEXT[m], top_k=1)
        mech = ev[0].mechanisms[0] if ev and ev[0].mechanisms else None
        eq = equations_for_mechanism(g, mech)[0].name if mech else None
        params = {p.name for p in parameters_for_mechanism(g, mech)} if mech else set()
        derivs = None
        if eq:
            _, fields, _, _ = parse_equation(eq, list(params))
            derivs = sorted(f for f in fields if f != "u")
        items.append(_item(f"M{i:02d}", "text_to_derivatives", 4,
                           f"Observation: '{SIGNATURE_TEXT[m]}'. Which mechanism does retrieval point to, which "
                           f"parameters does its documented equation use, and which derivatives must a PINN compute?",
                           {"mechanism": m, "derivatives": sorted(KEY_DERIVS[m])},
                           {"mechanism": mech, "parameters": sorted(params), "derivatives": derivs},
                           mech == m and derivs == sorted(KEY_DERIVS[m])))
    for i, p in enumerate(["D", "c", "r", "K"], start=18):
        mechs = sorted(m for m in MECHS if p in {q.name for q in parameters_for_mechanism(g, m)})
        derivs = set()
        for m in mechs:
            eq = equations_for_mechanism(g, m)[0].name
            _, fields, _, _ = parse_equation(eq, list(KEY_PARAMS[m]))
            derivs |= {f for f in fields if f != "u"}
        exp_m = sorted(m for m, ps in KEY_PARAMS.items() if p in ps)
        exp_d = set().union(*[KEY_DERIVS[m] for m in exp_m])
        items.append(_item(f"M{i:02d}", "parameter_to_derivatives", 3,
                           f"Which documented mechanisms involve parameter '{p}', and which derivatives do their equations need in total?",
                           {"mechanisms": exp_m, "derivatives": sorted(exp_d)},
                           {"mechanisms": mechs, "derivatives": sorted(derivs)},
                           mechs == exp_m and derivs == exp_d))
    diff_sig = canonical_signature(Hypothesis(id="d", name="d", equation="u_t = D*u_xx", parameters=[{"name": "D"}]))
    for i, m in enumerate(["advection", "reaction", "reaction-diffusion", "advection-diffusion"], start=22):
        eq = equations_for_mechanism(g, m)[0].name
        names = sorted(KEY_PARAMS[m])
        residual, _, _, local = parse_equation(eq, names)
        nests = False
        for p in names:
            if p == "D":
                continue
            reduced = sympy.expand(residual.subs(local[p], 0))
            if reduced.has(sympy.zoo) or reduced.has(sympy.nan):
                continue  # e.g. K -> 0 in u/K: not a meaningful limit
            remaining = [q for q in names if q != p and reduced.has(local[q])]
            if not remaining:
                continue
            h = Hypothesis(id="r", name="r", equation=f"{sympy.sstr(reduced)} = 0",
                           parameters=[{"name": q} for q in remaining])
            if canonical_signature(h) == diff_sig:
                nests = True
        items.append(_item(f"M{i:02d}", "nesting", 3,
                           f"Does the documented {m} equation reduce to pure diffusion when one of its parameters is set to zero?",
                           KEY_NESTS_DIFFUSION[m], nests, nests == KEY_NESTS_DIFFUSION[m]))
    return items
