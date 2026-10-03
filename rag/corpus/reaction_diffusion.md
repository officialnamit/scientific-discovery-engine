---
document_id: doc_reaction_diffusion
title: "Reaction-Diffusion Systems -- Benchmark Fixture"
authors: []
source: "benchmark fixture"
year: null
domain: reaction-diffusion
is_benchmark_fixture: true
declared_mechanism: reaction-diffusion
declared_variables: ["u", "x", "t"]
declared_parameters: ["D", "r", "K"]
declared_equations: ["u_t = D*u_xx + r*u*(1-u/K)"]
declared_assumptions: ["isotropic medium", "no advective transport", "logistic local reaction kinetics combined with diffusive spreading"]
---

# Reaction-Diffusion Systems

This is a benchmark fixture document summarizing well-known, generic
textbook physics for testing the Scientific Discovery Engine's
retrieval and knowledge-graph pipeline. It is not a reproduction of
any specific publication, and it does not state the parameter values
of any particular experiment.

## Overview

A reaction-diffusion system combines spatial spreading (diffusion)
with local growth, decay, or saturation (a reaction term). These
models are used across chemistry, ecology, and biology to describe
quantities that both spread spatially and grow/decay locally -- for
example a population that diffuses through a habitat while also
reproducing and competing for limited resources (logistic growth).

## Governing equation

A standard one-dimensional reaction-diffusion model combining Fick's
law with logistic growth is:

    u_t = D*u_xx + r*u*(1 - u/K)

where D is the diffusion coefficient, and r, K are the local reaction
rate and carrying capacity as in pure logistic kinetics. When r=0, this
equation reduces exactly to pure diffusion; when D=0, it reduces
exactly to pure local logistic reaction with no spatial coupling.

## Typical observational signatures

Systems governed by reaction-diffusion can show behavior that looks
like pure diffusion (spreading, smoothing, amplitude decay) PLUS an
amplitude trend that does not match a simple diffusive decay rate --
for instance, decay or growth that depends on the local magnitude of u
rather than following the uniform exponential-per-mode decay a pure
diffusion equation would predict for a given initial condition.
Distinguishing "pure diffusion" from "diffusion with a very weak
reaction term" from data alone can be difficult, since the reaction
term's effect vanishes continuously as r approaches zero.

## Typical assumptions

This model assumes an isotropic medium, no advective transport, and
that the only local nonlinearity is logistic self-limitation.
