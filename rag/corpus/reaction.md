---
document_id: doc_reaction
title: "Local Reaction Kinetics (Logistic Growth/Decay) -- Benchmark Fixture"
authors: []
source: "benchmark fixture"
year: null
domain: reaction
is_benchmark_fixture: true
declared_mechanism: reaction
declared_variables: ["u", "t"]
declared_parameters: ["r", "K"]
declared_equations: ["u_t = r*u*(1-u/K)"]
declared_assumptions: ["no spatial transport", "spatially uncoupled local dynamics", "logistic self-limiting growth/decay"]
---

# Local Reaction Kinetics

This is a benchmark fixture document summarizing well-known, generic
textbook physics for testing the Scientific Discovery Engine's
retrieval and knowledge-graph pipeline. It is not a reproduction of
any specific publication, and it does not state the parameter values
of any particular experiment.

## Overview

A local reaction term describes how a quantity grows, decays, or
saturates at a point due to local dynamics alone -- with no coupling
to neighboring spatial locations. A classic nonlinear form is logistic
growth/decay, where the rate of change depends on the current value
relative to a carrying capacity or equilibrium level.

## Governing equation

A standard local logistic reaction model (no spatial transport) is:

    u_t = r*u*(1 - u/K)

where u(t) is the local quantity, r is a growth/decay rate (units of
1/time), and K is a carrying capacity or saturation level (same units
as u). For r > 0 and u starting below K, the quantity grows and
asymptotically approaches K; for r < 0, it decays.

## Typical observational signatures

Systems governed purely by local reaction kinetics (with no spatial
coupling) typically show: (a) each spatial location evolves somewhat
independently -- there is no smoothing or sharpening driven by
neighboring values; (b) the rate of change at a point depends on the
local value itself, not on spatial derivatives; (c) the quantity tends
toward a steady state (zero or the carrying capacity) rather than
spreading or translating.

## Typical assumptions

A pure reaction model assumes no spatial transport term (no diffusion,
no advection) -- each location's evolution is governed only by its own
current value.
