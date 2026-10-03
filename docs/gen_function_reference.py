"""Generate docs/guide/A_functions.tex: every function/class in the project, with a description.
Fails loudly if the code has a function without a description, or a description for a missing function."""
import ast, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
OUT = os.path.join(ROOT, "docs/guide/A_functions.tex")

D = {
# ---------------- data ----------------
"data/generate_diffusion.py": {
  "load_config": "Read config.yaml into a dict.",
  "initial_condition": "The starting profile u(x,0): one sine (degenerate), two sines (default) or a Gaussian bump.",
  "solve_diffusion_fd": "Explicit finite-difference solver for u_t = D u_xx with zero boundary values; produces the ground truth.",
  "build_full_field": "Build the x and t grids from config and run the solver: the clean 800x61 field.",
  "flatten_field": "Turn the 2-D field into matching 1-D arrays (x, t, u), one row per grid point.",
  "make_sparse_and_noisy": "Draw the 10% noise-free subsample and the 400 noisy observations (5% noise).",
  "main": "Phase 1 entry point: generate and save full.npz, sparse.npz and noisy.npz.",
},
"data/problem_description.py": {
  "build_problem_description": "Compute statistics of the noisy observations (amplitude change, edges, centroid, smoothness) and write them as plain text, never naming a mechanism.",
},
# ---------------- equation_discovery ----------------
"equation_discovery/derivatives.py": {
  "reconstruct_grid": "Interpolate scattered (x, t, u) samples onto a regular grid (scipy griddata, nearest-neighbour at edges).",
  "smooth_grid": "Gaussian smoothing of a reconstructed grid to reduce noise before differentiation.",
  "finite_diff_derivatives": "Central finite differences on a grid: u_t, u_x, u_xx, u_xxx, with edge cropping.",
},
"equation_discovery/discovery_engine.py": {
  "_fields_from_grid": "Compute derivative fields on a grid and flatten them for regression.",
  "discover_from_grid": "Full discovery on gridded data: derivatives, library, sparse regression.",
  "discover_from_scatter": "Full discovery on scattered data: reconstruct, optionally smooth, then discover_from_grid.",
  "equation_to_string": "Format a discovery result as a readable equation string.",
  "evaluate_against_ground_truth": "Compare a discovered equation with the hidden truth (grading only, after discovery).",
  "evaluate_competing_hypotheses": "Fit each named single-term hypothesis separately and report R^2 and coefficients.",
},
"equation_discovery/library.py": {
  "term_label": "Human-readable label for a library term, e.g. ('u','u_x') -> 'u*u_x'.",
  "build_library": "Evaluate every candidate term at every sample: the matrix Theta.",
  "parse_term_specs": "Convert the config's term lists into term tuples.",
},
"equation_discovery/sindy.py": {
  "stlsq": "Sequentially thresholded least squares: fit, zero small coefficients, refit, repeat (columns normalised).",
  "run_sindy": "Fit u_t = Theta xi with STLSQ (pysindy if installed, else stlsq) and return a structured result.",
  "fit_restricted": "Ordinary least squares on a fixed set of terms (used for competing hypotheses).",
},
# ---------------- pinn ----------------
"pinn/data_utils.py": {
  "_to_tensor": "Convert a numpy array to a float32 column tensor.",
  "load_observational_data": "Load noisy.npz or sparse.npz, returning only x, t, u (the only data a PINN may train on).",
  "load_clean_reference": "Load the clean field for evaluation only, never for training.",
  "make_collocation_points": "Random (x, t) points where the physics residual is penalised.",
  "make_bc_points": "Random points on both boundaries with target u = 0.",
  "make_ic_points": "Random points at t = 0 with target u = initial condition.",
  "data_dict_to_tensors": "Convert an (x, t, u) dict into tensors.",
  "filter_by_time": "Keep only observations with t below a cutoff (train/OOD split).",
},
"pinn/derivatives.py": {
  "compute_derivatives": "Exact u, u_t, u_x, u_xx (and u_xxx) of the network output via torch.autograd.",
},
"pinn/forward.py": {"run_forward_pinn": "Train a PINN with physics parameters fixed (known D)."},
"pinn/inverse.py": {"run_inverse_pinn": "Train a PINN with physics parameters trainable (estimate D)."},
"pinn/losses.py": {
  "mse": "Mean squared error helper.",
  "data_loss": "Mismatch between network and observations.",
  "physics_loss": "Mean squared equation residual at collocation points.",
  "bc_loss": "Mismatch at the boundaries.",
  "ic_loss": "Mismatch with the initial condition.",
  "total_loss": "Weighted sum of the four losses.",
},
"pinn/network.py": {
  "PINN": "The one reusable network: (x, t) -> u through four tanh layers of 64 units.",
  "PINN._init_weights": "Xavier initialisation of the weights.",
  "PINN.forward": "Evaluate u at given (x, t).",
},
"pinn/residuals.py": {
  "diffusion_residual": "Residual u_t - D u_xx.",
  "advection_residual": "Residual u_t - c u_x.",
  "reaction_diffusion_residual": "Residual u_t - D u_xx + k u.",
  "make_params": "Wrap initial parameter values as tensors, trainable or fixed.",
},
"pinn/trainer.py": {
  "PINNTrainer": "Holds network, residual function, parameters and loss weights; runs training.",
  "PINNTrainer.trainable_parameters": "Network weights plus any trainable physics parameters.",
  "PINNTrainer.compute_losses": "One evaluation of all loss terms (derivatives, residual, data, BC, IC).",
  "PINNTrainer.train": "Adam steps then an L-BFGS polish, recording loss history.",
  "PINNTrainer._record": "Store one row of loss history and current parameter values.",
  "build_and_train": "Shared entry point: build network and parameters, train, return trainer and history.",
},
# ---------------- validation ----------------
"validation/forward_solver.py": {
  "rhs_function": "Compile any candidate equation into a numerical right-hand side u_t = f(u, u_x, u_xx).",
  "_effective_coefficients": "Estimate diffusion/advection/reaction strengths to choose a stable time step.",
  "solve_forward": "Solve a candidate law forward in time from an initial condition (Euler or RK4).",
},
"validation/model_selection.py": {
  "parameter_contributions": "Term necessity: share of RMS(u_t) contributed by each parameter's terms on the trained PINN.",
  "reduced_signature": "Signature of the equation with inactive terms removed (does it reduce to another candidate?).",
  "bic": "Bayesian information criterion n ln(MSE) + k ln n.",
  "select_model": "The selection rule: gate, fit-equivalence, optional reduction, complexity, BIC.",
},
"validation/conservation.py": {
  "conservation_violation": "Global balance check: does the model's total amount change exactly as its own law says (dM/dt = integral of f)?",
},
"validation/ood.py": {
  "split_by_time": "Split points into t <= cutoff and t > cutoff.",
  "prediction_error": "RMSE and MAE of a model against reference values.",
},
"validation/physics_validation.py": {
  "evaluate_physics_residual": "Mean squared residual of a trained PINN on a fresh grid (not the training points).",
},
"validation/scorer.py": {
  "build_validation_result": "Turn measured losses into supported/rejected using fixed thresholds (Phase 3).",
},
"validation/stability.py": {
  "candidate_monomials": "The distinct field terms on a candidate's right-hand side (e.g. u_xx, u, u^2).",
  "bootstrap_stability": "Refit those terms by least squares on 20 random 80% subsets; report mean, spread, sign stability.",
},
# ---------------- llm ----------------
"llm/__init__.py": {
  "get_provider": "Choose the LLM provider from config and environment (Gemini if a key is set, else the mock).",
},
"llm/base.py": {
  "LLMProvider": "Abstract interface every LLM provider implements.",
  "LLMProvider.generate": "Return the raw text completion for a prompt.",
  "LLMProvider.name": "The provider's class name, recorded in results.",
},
"llm/gemini_provider.py": {
  "_repair_json_escapes": "If the response is not valid JSON, double stray backslashes (LaTeX like \\partial).",
  "GeminiProvider": "Real provider for Google Gemini; the key comes only from GEMINI_API_KEY.",
  "GeminiProvider._get_client": "Lazily import google-genai and create the client; clear error if no key.",
  "GeminiProvider.generate": "Call Gemini in JSON mode, retrying overload errors (429/500/503) with back-off.",
},
"llm/mock_provider.py": {
  "MockLLMProvider": "Deterministic offline provider returning a canned set of four hypotheses.",
  "MockLLMProvider.generate": "Return the same canned JSON every time, whatever the prompt.",
},
"llm/openai_provider.py": {
  "OpenAIProvider": "Legacy OpenAI provider (no longer selectable by get_provider).",
  "OpenAIProvider._get_client": "Lazily create the OpenAI client.",
  "OpenAIProvider.generate": "Call OpenAI chat completions.",
},
"llm/prompts.py": {
  "build_hypothesis_prompt_with_context": "Mode B prompt: observations plus retrieved evidence, mechanisms, equations and parameters.",
  "build_hypothesis_prompt": "Mode A prompt: observations only, with the JSON schema the answer must follow.",
},
"llm/schemas.py": {
  "ParameterSpec": "One parameter: name, trainable flag, initial value, units, optional bounds.",
  "Hypothesis": "The structured hypothesis every LLM answer must fit (pydantic model).",
  "Hypothesis._confidence_in_range": "Validator: confidence must lie in [0, 1].",
  "Hypothesis._equation_has_equals": "Validator: the equation must contain '='.",
  "HypothesisSet": "A validated collection of hypotheses for one problem.",
},
# ---------------- hypotheses ----------------
"hypotheses/adapter.py": {
  "hypothesis_to_pinn_config": "Compile a hypothesis into exactly what build_and_train needs (residual function, parameters).",
},
"hypotheses/complexity.py": {
  "equation_complexity": "Complexity score: terms + free parameters + derivative order + 2 x nonlinear + 0.1 x operations.",
},
"hypotheses/constraints.py": {
  "resolve_bounds": "Combine a hypothesis's declared bounds with the config fallbacks (D > 0, K > 0).",
  "precheck_physics": "Before training: reject non-evolution equations; warn on over-determined or uncoupled boundary setups.",
  "check_fitted_bounds": "After training: flag fitted parameters that violate physical bounds.",
},
"hypotheses/equation_compiler.py": {
  "EquationParseError": "Raised when an equation string cannot be parsed.",
  "parse_equation": "Parse 'lhs = rhs' with SymPy into a residual; detect fields and parameters used.",
  "compile_residual": "Turn the residual into a torch function residual_fn(derivs, params) for the trainer.",
  "equation_str_to_sympy_str": "Normalised human-readable form of an equation.",
},
"hypotheses/pipeline.py": {
  "_extract_json_array": "Strip markdown code fences and parse the LLM's JSON array.",
  "generate_hypotheses": "Prompt, call the provider, parse, schema-check, validate, deduplicate and rank.",
},
"hypotheses/ranking.py": {
  "_completeness_score": "Fraction of descriptive fields the LLM filled in.",
  "_simplicity_score": "1 / (1 + number of parameters).",
  "rank_hypotheses": "Composite score that sets investigation order only, never the verdict.",
},
"hypotheses/search.py": {
  "kg_template_candidates": "Turn equations documented in the knowledge graph into candidate hypotheses.",
  "build_candidate_pool": "Merge LLM and graph candidates and remove structural duplicates.",
  "structural_and_physics_filter": "Run deterministic validation and physics pre-checks; keep the survivors.",
  "make_splits": "Train / random holdout / OOD (t > cutoff) splits of the observations.",
  "_rmse": "RMSE of a network on a split.",
  "validate_candidate": "Inverse PINN on the training split, then gate checks, bounds, term necessity, complexity, BIC.",
  "load_candidate_model": "Reload a saved candidate network.",
  "train_data_only_mlp": "Baseline: the same network trained on data only, without physics.",
  "save_json": "Write a result dict to JSON, creating folders as needed.",
},
"hypotheses/validation.py": {
  "validate_hypothesis": "Deterministic checks: syntax, variables, parameters, derivatives, units.",
  "canonical_signature": "Parameter- and sign-invariant structure of an equation, for duplicate detection.",
  "deduplicate": "Remove hypotheses with identical canonical signatures.",
},
# ---------------- rag ----------------
"rag/chunker.py": {"chunk_document": "Split a document into chunks at its section headers."},
"rag/context.py": {
  "ScientificContext": "The unified evidence object: retrieved evidence, mechanisms, equations, parameters, sources.",
  "build_scientific_context": "Retrieve evidence for the description, then query the graph for every mechanism it mentions.",
},
"rag/corpus_loader.py": {
  "_parse_frontmatter": "Split a Markdown file into its metadata header and body.",
  "load_document": "Load one corpus file as a Document.",
  "load_corpus": "Load every .md file in rag/corpus/.",
},
"rag/documents.py": {
  "DocumentMetadata": "A document's id, title, source, domain and declared mechanism, equations and parameters.",
  "Document": "Metadata plus text.",
  "Chunk": "A piece of a document with its section and id.",
},
"rag/embeddings.py": {
  "EmbeddingProvider": "Interface for turning text into vectors.",
  "EmbeddingProvider.embed": "Embed one text.",
  "EmbeddingProvider.embed_many": "Embed many texts.",
  "EmbeddingProvider.name": "Provider name.",
  "LocalEmbeddingProvider": "Deterministic offline embedding: words hashed into 256 slots, normalised.",
  "LocalEmbeddingProvider._token_index": "Hash a word to a slot (MD5 modulo 256).",
  "LocalEmbeddingProvider.embed": "Count words per slot and normalise to unit length.",
  "APIEmbeddingProvider": "Optional real embedding API provider (unused by default).",
  "APIEmbeddingProvider._get_client": "Lazily create its API client.",
  "APIEmbeddingProvider.embed": "Embed via the API.",
},
"rag/evidence.py": {
  "RetrievalResult": "One retrieved chunk with its similarity score.",
  "Evidence": "A retrieved claim with provenance; its confidence is retrieval confidence only.",
  "evidence_from_retrieval": "Convert a retrieval result into an Evidence object.",
},
"rag/metadata.py": {"matches_filters": "Check a chunk's metadata against filter values."},
"rag/pipeline.py": {
  "RAGPipeline": "Ties chunker, embedder and vector store together.",
  "RAGPipeline.ingest": "Chunk, embed and store every document.",
  "RAGPipeline.get_retriever": "Return a Retriever over the stored chunks.",
},
"rag/retriever.py": {
  "Retriever": "Search interface over the vector store.",
  "Retriever.retrieve": "Embed the query, take the most similar chunks, apply filters, return Evidence.",
},
"rag/vector_store.py": {
  "VectorStore": "Interface: add, search, get.",
  "VectorStore.add": "Store a vector with its payload.",
  "VectorStore.search": "Return the top-k most similar vectors.",
  "VectorStore.get": "Fetch one stored item.",
  "InMemoryVectorStore": "Brute-force cosine-similarity store in memory.",
  "InMemoryVectorStore.add": "Store a vector with its payload.",
  "InMemoryVectorStore.search": "Cosine similarity against every stored vector; return the top-k.",
  "InMemoryVectorStore.get": "Fetch one stored item.",
  "InMemoryVectorStore.all_ids": "List stored ids.",
  "PersistentVectorStore": "The in-memory store plus JSON save/load.",
  "PersistentVectorStore.save": "Write the store to JSON.",
  "PersistentVectorStore.load": "Read the store from JSON.",
},
# ---------------- knowledge_graph ----------------
"knowledge_graph/builder.py": {
  "_slug": "Make a stable id fragment from a name.",
  "_find_chunk_id": "Find the chunk that states a fact (provenance).",
  "build_graph_from_documents": "Create Document, Mechanism, Equation, Variable and Parameter entities and sourced relationships.",
},
"knowledge_graph/graph.py": {
  "_relationship_id": "Stable hash of (subject, predicate, object), so repeated facts merge.",
  "KnowledgeGraph": "The graph container (backed by networkx).",
  "KnowledgeGraph.add_entity": "Add an entity, merging if its id exists.",
  "KnowledgeGraph.get_entity": "Look up an entity by id.",
  "KnowledgeGraph.entities_by_type": "All entities of one type.",
  "KnowledgeGraph.add_relationship": "Add a typed, sourced edge, merging sources for duplicates.",
  "KnowledgeGraph.neighbors": "Entities connected to a given entity.",
  "KnowledgeGraph.relationships_for": "Edges touching a given entity.",
  "KnowledgeGraph.summary": "Counts of entities and relationships by type.",
},
"knowledge_graph/query.py": {
  "find_entity_by_name": "Find an entity by its name.",
  "equations_for_mechanism": "Documented equations that govern a mechanism.",
  "variables_for_mechanism": "Variables in those equations.",
  "parameters_for_mechanism": "Parameters in those equations.",
  "mechanisms_for_observation": "Mechanisms linked to an observation entity (schema-ready).",
  "documents_supporting": "Documents that support an entity.",
  "sources_for_relationship": "Provenance of an edge.",
  "evidence_connecting": "Relationships and shared equations linking two variables.",
},
"knowledge_graph/schema.py": {
  "EntityType": "Allowed entity types (Document, Mechanism, Variable, Parameter, Equation, Observation).",
  "RelationType": "Allowed relationship types: GOVERNS, HAS_VARIABLE, HAS_PARAMETER, SUPPORTED_BY, CAUSES, DEPENDS_ON, and others.",
  "Entity": "A node: id, type, name, metadata.",
  "SourceRef": "Where a fact came from: document and chunk.",
  "Relationship": "A typed edge with its sources.",
},
"knowledge_graph/serialization.py": {
  "graph_to_dict": "Convert the graph to a JSON-able dict.",
  "graph_from_dict": "Rebuild a graph from that dict.",
  "save_graph": "Write the graph to a JSON file.",
  "load_graph": "Read a graph from a JSON file.",
},
# ---------------- evaluation ----------------
"evaluation/ablation.py": {
  "_discovery_only": "The SINDy-alone arm: discover, then match the result to a documented mechanism.",
  "run_ablation": "Run all seven arms on the 12 datasets.",
  "_median": "Median helper.",
  "summarize_ablation": "Per-arm totals: correct model, wrong candidates rejected, parameter and OOD accuracy.",
},
"evaluation/adversarial.py": {
  "_CannedProvider": "Test LLM that returns a fixed (possibly malicious) response.",
  "_CannedProvider.generate": "Return the canned response.",
  "_H": "Shorthand to build a test Hypothesis.",
  "_case": "Record one adversarial case result.",
  "find_leaks": "Search corpus text for any hidden benchmark value.",
  "run_adversarial": "Run the 20 attack cases.",
},
"evaluation/benchmark.py": {
  "_item": "Record one question result.",
  "_hyp": "Build a Hypothesis from an equation string, inferring parameter names.",
  "run_single_questions": "The 50 single questions over the 12 datasets and the graph.",
  "run_multihop": "The 25 multi-hop chains.",
},
"evaluation/datasets.py": {
  "Dataset": "One benchmark system: family, truth, grids, field, initial condition, exact solution.",
  "_diffusion": "Exact two-mode diffusion dataset.",
  "_gauss": "Exact moving, spreading Gaussian; used for advection (D = 0) and advection-diffusion.",
  "_advection": "Pure advection dataset.",
  "_adv_diff": "Advection-diffusion dataset.",
  "_reaction_diffusion": "Reaction-diffusion dataset solved numerically with RK4.",
  "benchmark_datasets": "The 12 datasets (4 families x 3 settings), cached to .npz.",
  "_build_all": "Construct all 12 datasets.",
  "_rd_shell": "Reaction-diffusion dataset without its (expensive) field, for quick metadata.",
  "add_noise": "Add Gaussian noise to a dataset's field.",
},
"evaluation/describe.py": {
  "_centroid": "Amplitude-weighted centre of a profile.",
  "_spread": "Width of a profile.",
  "_n_peaks": "Number of peaks in a profile.",
  "observation_stats": "Amplitude change, drift, spread and peaks between early and late times.",
  "describe_observations": "Fixed-template text description from those statistics (never names a mechanism).",
},
"evaluation/engine.py": {
  "Resources": "Corpus, RAG index and graph, built once and shared.",
  "build_pool": "Description -> RAG/graph context -> candidate pool (evidence equations, graph templates, mock LLM).",
  "_grid_param_contributions": "Term necessity computed on grid derivatives instead of a PINN.",
  "_reduced_term_set": "Term set of an equation with its inactive terms removed.",
  "validate_and_select": "Fast pipeline: least-squares fit per candidate, gate, selection.",
  "predict_ood": "Solve the selected law forward beyond the data window.",
  "family_of": "Map a hypothesis's term set to its mechanism family.",
  "truth_field": "Ground-truth field at given times (evaluation only).",
  "score_run": "Score one run: correct family, parameter error, OOD error.",
  "family_of_row": "Mechanism family from a stored result row.",
},
"evaluation/identify.py": {
  "_syms": "SymPy symbols for the fields.",
  "term_set": "The set of field terms in a hypothesis.",
  "_label_to_expr": "Convert a library label like 'u*u_x' to a SymPy expression.",
  "discovered_coefficients": "Non-zero SINDy coefficients as {term: value}.",
  "match_mechanism": "Find the documented template with exactly the discovered term set.",
  "recover_parameters": "Solve for a template's parameters from discovered coefficients.",
  "grid_fields": "Finite-difference fields of a gridded dataset, flattened.",
  "fit_candidate_on_grid": "Least-squares fit of u_t on a candidate's own terms; train and holdout error.",
  "library_collinearity": "Identifiability check: maximum correlation between library columns.",
},
"evaluation/leakage.py": {
  "ground_truth_references": "List ground-truth identifiers mentioned in a function's source code.",
  "decision_functions": "The six functions that make decisions (pool, validation, splits, selection).",
  "leakage_audit": "Static audit of those functions plus a corpus scan for hidden values.",
},
# ---------------- experiments ----------------
"experiments/02_equation_discovery.py": {
  "load_config": "Read config.yaml.",
  "run_case": "Run discovery on one dataset variant and evaluate it.",
  "main": "Phase 2: clean, sparse, noisy (raw and smoothed); save report, table and plot.",
},
"experiments/03_forward_pinn.py": {"load_config": "Read config.yaml.", "main": "Phase 3 forward PINN with D = 0.1 fixed."},
"experiments/04_inverse_pinn.py": {"load_config": "Read config.yaml.", "main": "Phase 3 inverse PINN estimating D from 0.2."},
"experiments/05_hypothesis_validation.py": {
  "load_config": "Read config.yaml.",
  "run_hypothesis": "Train one hypothesis's PINN and score it.",
  "main": "Phase 3: diffusion vs advection with a time-based OOD split.",
},
"experiments/06_hypothesis_generation.py": {"load_config": "Read config.yaml.", "main": "Phase 4: generate, validate and rank hypotheses; optionally run PINNs."},
"experiments/07_scientific_rag_kg.py": {
  "load_config": "Read config.yaml.",
  "hyp_to_dict": "Serialise a hypothesis.",
  "summarize_generation": "Counts for a generation run (candidates, valid, duplicates, evidence-supported).",
  "main": "Phase 5: corpus, RAG, graph, context, leakage check, Mode A vs Mode B.",
},
"experiments/08_physics_constrained_search.py": {
  "load_config": "Read config.yaml.",
  "stage_prepare": "Phase 6 stage 1: context, generation, pool, filters.",
  "_load_survivors": "Reload the filtered candidates.",
  "_splits": "Build train/holdout/OOD splits from noisy.npz.",
  "_setup": "Domain and initial-condition settings for the PINN.",
  "stage_pinn": "Phase 6 stage 2: inverse PINN per survivor, cached.",
  "oracle_evaluation": "Benchmark-only error against the hidden truth, after selection.",
  "stage_select": "Phase 6 stage 3: stability, selection, baselines, OOD, novel prediction, report.",
  "_plots": "Draw the Phase 6 summary figure.",
  "_report": "Write phase6_report.md.",
  "main": "Run the chosen stages.",
},
"experiments/09_evaluation.py": {
  "save": "Write one Phase 7 result file.",
  "real_llm": "Mode A vs Mode B with a real LLM on 3 datasets (only with --real-llm and a key).",
  "acc": "Format an accuracy fraction.",
  "report": "Write phase7_report.md.",
  "main": "Phase 7: questions, multi-hop, adversarial, ablation, leakage, determinism check, report.",
},
"experiments/10_generalization.py": {
  "_truth_dataset": "The hidden advection-diffusion system (c = 0.4, D = 0.02).",
  "observations": "400 noisy samples, the only data the pipeline sees.",
  "_recon": "Grid reconstruction for SINDy and the description.",
  "_setup": "Domain settings for the PINN.",
  "stage_prepare": "SINDy baseline, pool and filters for the new system.",
  "_survivors": "Reload filtered candidates.",
  "_splits": "Train/holdout/OOD splits.",
  "stage_pinn": "Inverse PINN per candidate.",
  "stage_select": "Selection under both rules, OOD, oracle evaluation.",
  "main": "Run the chosen stages.",
},
"experiments/12_conservation.py": {
  "_windows": "Conservation violation in the train, OOD and beyond-data windows.",
  "evaluate_run": "Conservation check for every saved candidate of one Phase 6 run (no retraining).",
  "main": "Check the mock and Gemini runs plus the physics-free baseline; write results/evaluation/conservation/.",
},
"experiments/11_real_llm_gemini.py": {
  "_load_script": "Import an experiment script by file path.",
  "_phase6_module": "Load the Phase 6 script and redirect its outputs to the Gemini folder.",
  "stage_generate": "Run 09's real_llm Mode A vs Mode B with Gemini.",
  "annotate_report": "Mark the reused Phase 6 report as a real-LLM run.",
  "compare_with_mock": "Side-by-side summary of the mock and Gemini Phase 6 runs.",
  "main": "Check the key, force the Gemini provider, run the chosen stages.",
},
}

_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$", "^": r"\^{}",
        "~": r"\~{}", "{": r"\{", "}": r"\}", "_": r"\_"}


def esc(s):
    return "".join(_ESC.get(ch, ch) for ch in s).replace("--", "-{}-")

found = {}
for f in sorted(glob.glob(os.path.join(ROOT, "*/*.py"))):
    rel = os.path.relpath(f, ROOT)
    if rel.startswith(("tests/", "docs/")):
        continue
    names = []
    for n in ast.parse(open(f).read()).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.append(n.name)
            if isinstance(n, ast.ClassDef):
                names += [f"{n.name}.{m.name}" for m in n.body
                          if isinstance(m, ast.FunctionDef) and not m.name.startswith("__")]
    if names:
        found[rel] = names

missing = [(m, n) for m, ns in found.items() for n in ns if n not in D.get(m, {})]
stale = [(m, n) for m, ds in D.items() for n in ds if n not in found.get(m, [])]
if missing or stale:
    print("MISSING:", missing); print("STALE:", stale); sys.exit(1)

PKG = ["data", "equation_discovery", "pinn", "validation", "llm", "hypotheses", "rag", "knowledge_graph",
       "evaluation", "experiments"]
L = [r"% generated by docs/gen_function_reference.py -- do not edit by hand",
     r"\section{Complete function reference}\label{app:functions}",
     r"Every function and class in the project (tests excluded), grouped by package and module, generated from "
     r"the source code so that nothing is missed. Methods are listed as \texttt{Class.method}. "
     f"In total: {sum(len(v) for v in found.values())} entries in {len(found)} modules.",
     ""]
for pkg in PKG:
    mods = [m for m in found if m.split("/")[0] == pkg]
    if not mods:
        continue
    L.append(r"\subsection*{\texttt{%s/}}" % esc(pkg))
    L.append(r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{5.6cm}>{\raggedright\arraybackslash}p{10.6cm}@{}}")
    for m in mods:
        L.append(r"\multicolumn{2}{@{}l}{\small\bfseries\texttt{%s}}\\[1pt]" % esc(m))
        for n in found[m]:
            L.append(r"\quad\fn{%s} & \small %s \\" % (n, esc(D[m][n])))
        L.append(r"\addlinespace[4pt]")
    L.append(r"\end{longtable}")
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT, sum(len(v) for v in found.values()), "entries")
