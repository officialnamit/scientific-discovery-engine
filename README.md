# Generative Scientific Discovery Engine

**Can an AI system work out the law of physics behind a set of measurements, when nobody tells it the equation?**

This project builds a system that tries. You give it a few hundred noisy measurements of something that changes over
space and time, such as temperature along a rod. It then:

1. reads relevant scientific background,
2. proposes candidate equations that might explain the data,
3. tests each candidate against both the data and the laws of physics,
4. throws out the ones that don't hold up,
5. picks the simplest one that does, and
6. uses it to predict what happens next, beyond the data it was given.

The system never sees the true equation. That's only used afterwards, to grade the answer.

---

## What it found

The test data came from a known law (diffusion: $u_t = D\,u_{xx}$, with $D = 0.1$), hidden from the system.

| Question | Result |
|---|---|
| Did it find the right law? | Yes: `u_t = D*u_xx` |
| How accurate is the unknown constant? | $\hat D = 0.10015$ (0.15% error) |
| Did it reject wrong explanations? | Yes: pure advection (the field drifting sideways) and pure growth/decay were rejected |
| Can it predict times it never saw? | Yes: error $1.3\times10^{-4}$ beyond the observed window; on held-out later times, 6× more accurate than an ordinary neural network |
| Does it work on a different, unseen system? | Yes: recovered advection–diffusion with its two constants within 0.4% and 1.0% |
| Benchmark | 50/50 science questions, 23/25 multi-step problems, 20/20 adversarial "trick" cases |
| With a real LLM (Google Gemini)? | Yes: reached the same correct law ($\hat D = 0.10001$) |
| Does the selected law respect conservation? | Yes: its global balance is off by only 0.10%, vs 14.9% for an ordinary neural network |

---

## How it works

```
 Measurements (400 noisy points)
        │
        ▼
 1. Describe the data in words          e.g. "the peak shrank a lot but didn't move"
        │
        ▼
 2. Look up background (RAG)            search a small library of physics notes
 3. Organise it (knowledge graph)       which mechanisms use which equations and constants
        │
        ▼
 4. Propose candidate equations         an LLM suggests ideas; the knowledge graph adds textbook forms
 5. Sanity-check each candidate         valid maths? physically allowed? not a duplicate?
        │
        ▼
 6. Test each candidate with a PINN     a neural network that must fit the data AND obey the equation;
                                        it also estimates the unknown constants
 7. Reject what doesn't hold up         judged only on data held back from training
 8. Pick the simplest survivor          extra terms that do nothing are removed
        │
        ▼
 9. Predict the future                  check on later, unseen data, then forecast beyond it
```

Two rules make this trustworthy:

- **The LLM only suggests.** It can't mark an equation as correct. Only measured evidence decides.
- **No peeking at the answer.** Every decision uses only the noisy measurements. The true equation is loaded in one
  place, after all decisions are made, purely to report how well the system did. An automated audit checks this.

---

## Quick start

Requires Python 3.10+. Everything runs on a CPU.

```bash
pip install -r requirements.txt
python -m pytest tests -q          # 95 tests, about 15 seconds, no API key or internet needed
```

All results are already saved in `results/`, so you can read them without re-running anything (see
[Where the results are](#where-the-results-are)).

---

## Reproducing all results

The numbers above come from one complete run on macOS (Apple Silicon, CPU only) with Python 3.14.7, PyTorch 2.14.0,
NumPy 2.5.3, SciPy 1.18.1, pysindy 2.1.0 and google-genai 2.28.0. Run every command from the project root, in this
order. Each script reads the outputs of the earlier ones.

**1. Install and test** (about 25 seconds)

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Expected: `95 passed`.

**2. Run the pipeline with the offline mock LLM.** Do this *without* the Gemini key loaded. Each LLM-using script
prints a warning that it is falling back to the mock; that is expected.

| # | Command | What it does | Time | Expected result |
|---|---|---|---|---|
| 1 | `python data/generate_diffusion.py` | Simulates the hidden law; samples 400 noisy points | seconds | noise std 0.01159 |
| 2 | `python experiments/02_equation_discovery.py` | Regression baseline (SINDy) | seconds | clean data: `u_t = 0.1009*u_xx`; noisy data: fails |
| 3 | `python experiments/03_forward_pinn.py` | Neural network solving the known equation | ~4 min | error vs truth 0.00075 |
| 4 | `python experiments/04_inverse_pinn.py` | Estimates the unknown constant, starting from 0.2 | ~4 min | $\hat D$ = 0.099909 |
| 5 | `python experiments/05_hypothesis_validation.py` | Diffusion vs advection | ~8 min | diffusion supported, advection rejected |
| 6 | `python experiments/06_hypothesis_generation.py --skip-pinn` | LLM proposes equations | seconds | 4 valid hypotheses |
| 7 | `python experiments/07_scientific_rag_kg.py` | Retrieval + knowledge graph | ~4 min | 22 entities, 40 relationships |
| 8 | `python experiments/08_physics_constrained_search.py` | **Full discovery pipeline** | ~20 min | selects `u_t = D*u_xx`, $\hat D$ = 0.10015 |
| 9 | `python experiments/10_generalization.py` | Same pipeline on an unseen system | ~16 min | selects advection–diffusion, c = 0.398, D = 0.0198 |
| 10 | `python experiments/09_evaluation.py` | Benchmarks, adversarial cases, ablations | ~30 s | 50/50, 23/25, 20/20 |

Script 10 runs before script 09 because 09 reads its results.

**3. Real-LLM run with Google Gemini.** Put `GEMINI_API_KEY=your-key` in a `.env` file in the project root (it is
git-ignored), then:

```bash
set -a; source .env; set +a
python experiments/11_real_llm_gemini.py --stage generate --model gemini-3.5-flash-lite
python experiments/11_real_llm_gemini.py --stage prepare --model gemini-3.5-flash-lite
python experiments/11_real_llm_gemini.py --stage pinn
python experiments/11_real_llm_gemini.py --stage select
```

The `pinn` stage takes about 25 minutes and makes no LLM calls. Expected: Gemini's candidates all reduce to diffusion,
and the same law is selected with $\hat D$ = 0.10001. Results go only to `results/evaluation/real_llm_gemini/`.

**4. Conservation check** (seconds; uses the models saved by steps 2 and 3)

```bash
python experiments/12_conservation.py
```

Expected: the selected law breaks the balance by 0.10% in the fitted window; the ordinary network by 14.9%.

**On another machine:** which equations are accepted, rejected and selected, and the benchmark scores, reproduce
reliably. Fitted values and error sizes can differ slightly between machines and library versions, because
neural-network training is sensitive to tiny floating-point differences. For example, an earlier run in a different
environment gave $\hat D$ = 0.10002. Gemini's suggestions also vary from run to run.

Every script accepts `--config config.yaml`. The slow ones (08, 10, 11) save each trained network as they go, and
accept `--stage` and `--only` to resume or run part of the work.

---

## Choosing the LLM

The system can propose equations with either of these:

- **Google Gemini** (a real LLM), when the `GEMINI_API_KEY` environment variable is set.
- **A built-in mock**, which returns a fixed set of answers. It needs no key or internet, so tests and runs are fully
  repeatable.

`config.yaml` selects Gemini (`llm.provider: "gemini"`) and falls back to the mock, with a warning, if no key is set.
To use Gemini, put your key in a `.env` file in the project root (it is git-ignored), then load it into your shell:

```bash
set -a; source .env; set +a
```

> **Note:** with the key loaded, scripts 06–08 call Gemini and overwrite their saved (mock-based) results. To keep
> the recorded results, run those scripts without the key, or set `llm.provider: "mock"`. Script 11 writes only to its
> own folder.

---

## Project layout

| Folder | What's inside |
|---|---|
| `data/` | Generates the synthetic measurements; `data/generated/` holds the datasets |
| `equation_discovery/` | The regression baseline: numerical derivatives and sparse regression (SINDy) |
| `pinn/` | The physics-informed neural network: model, derivatives, losses, training loop |
| `llm/` | LLM providers (Gemini, mock), prompts, and the format every hypothesis must follow |
| `rag/` | Retrieval: the background library (`rag/corpus/`), chunking, search |
| `knowledge_graph/` | The knowledge graph: building it, querying it, saving it |
| `hypotheses/` | Turns equation text into something a network can test; checks, ranking, the search itself |
| `validation/` | Judging candidates: selection rules, out-of-sample checks, stability, conservation, forward solver |
| `evaluation/` | Benchmark datasets, question sets, adversarial cases, ablations, leakage audit |
| `experiments/` | One script per step (see the table above) |
| `tests/` | Automated tests |
| `results/` | All saved outputs (below) |
| `docs/` | Source of `Report.pdf` (`Report.tex`, flowcharts) and the development log |
| `config.yaml` | Every setting and every pass/fail threshold, fixed before the experiments were run |

---

## Where the results are

| What | Where |
|---|---|
| **Main discovery run**: candidates, trained models, selected law, predictions, report | `results/phase6/` |
| Regression baseline | `results/equations/` |
| Neural-network experiments | `results/pinn/` |
| Generated hypotheses | `results/hypotheses/` |
| Knowledge graph and retrieval results | `results/rag_kg/` |
| Benchmarks, adversarial cases, ablations, leakage audit | `results/evaluation/` |
| Second system | `results/evaluation/generalization/` |
| Real Gemini run and its comparison with the mock | `results/evaluation/real_llm_gemini/` |
| Conservation check | `results/evaluation/conservation/` |

---

## Documentation

| Document | Read it if you want… |
|---|---|
| [`Report.pdf`](Report.pdf) | The technical report: requirements, method, results, flowcharts and reproduction steps |
| [`results/final_report.md`](results/final_report.md) | The detailed final report, with every number traced to its source file |
| [`docs/development_log.md`](docs/development_log.md) | The build history, written stage by stage as the project grew |

---

## Limitations

- Tested on synthetic, one-dimensional data only. It recovers known laws; it doesn't claim new physics.
- Boundary and starting conditions are given to the system, not inferred.
- Uncertainty is estimated by resampling (how stable each term is), not a full probabilistic analysis.
- When a more complex equation contains the right one as a special case, both fit equally well. The system then
  prefers the simpler one by a stated rule, which is not a proof of uniqueness.
- The background library has only five short documents, and search is keyword-based.
- The regression baseline fails on noisy data; the neural-network route is what works there.

---

## Glossary

| Term | Meaning |
|---|---|
| **PDE** | Partial differential equation: a rule for how a quantity changes over space and time |
| **Diffusion / advection** | Spreading out / drifting sideways |
| **PINN** | Physics-informed neural network: a network trained to match the data *and* obey an equation |
| **Inverse problem** | Working out unknown constants (like $D$) from data |
| **RAG** | Retrieval-augmented generation: looking up relevant documents before asking the LLM |
| **Knowledge graph** | Facts stored as linked entities (mechanism → equation → constant), each with its source |
| **SINDy** | A regression method that finds equations by picking a few terms from a large list |
| **Out-of-distribution (OOD)** | Data from conditions the model wasn't trained on; here, later times |
| **Ablation** | Removing one component at a time to see how much it matters |
