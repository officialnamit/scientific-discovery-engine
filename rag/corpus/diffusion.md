---
document_id: doc_diffusion
title: "Diffusive Transport (Fick's Second Law) -- Benchmark Fixture"
authors: []
source: "benchmark fixture"
year: null
domain: diffusion
is_benchmark_fixture: true
declared_mechanism: diffusion
declared_variables: ["u", "x", "t"]
declared_parameters: ["D"]
declared_equations: ["u_t = D*u_xx"]
declared_assumptions: ["isotropic medium", "no advective transport", "no reaction/source terms", "constant diffusion coefficient"]
---

# Diffusive Transport

This is a benchmark fixture document summarizing well-known, generic
textbook physics for testing the Scientific Discovery Engine's
retrieval and knowledge-graph pipeline. It is not a reproduction of
any specific publication, and it does not state the parameter values
of any particular experiment.

## Overview

Diffusion is the net movement of a quantity (concentration, heat,
density, etc.) from regions of higher value to regions of lower value,
driven by random microscopic motion rather than any directed flow.
Over time, an initially uneven spatial distribution of the quantity
tends to spread out and smooth into a more uniform profile, while the
total integrated quantity is conserved in a closed system with no
sources or sinks.

## Governing equation

The standard one-dimensional model of diffusive transport is Fick's
second law:

    u_t = D * u_xx

where u(x,t) is the diffusing quantity, x is a spatial coordinate, t
is time, and D > 0 is the diffusion coefficient (units of
length^2/time), which characterizes how quickly the quantity spreads.

## Typical observational signatures

Systems governed by diffusion typically show: (a) the spatial profile
of the quantity smooths out over time -- sharp local features flatten
and multiple local extrema tend to merge into fewer, broader ones; (b)
the overall amplitude of spatial variation decreases over time if
there is no external forcing; (c) there is no bulk translation of the
profile across the domain -- diffusion spreads a feature in place
rather than carrying it sideways; (d) higher-spatial-frequency
features decay faster than lower-spatial-frequency features.

## Typical assumptions

A pure diffusion model assumes an isotropic medium, a constant
diffusion coefficient, no advective (directional) transport, and no
local reaction, growth, or decay term beyond the spreading itself.
