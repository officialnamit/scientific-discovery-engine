---
document_id: doc_advection_diffusion
title: "Advection-Diffusion Systems -- Benchmark Fixture"
authors: []
source: "benchmark fixture"
year: null
domain: advection-diffusion
is_benchmark_fixture: true
declared_mechanism: advection-diffusion
declared_variables: ["u", "x", "t"]
declared_parameters: ["c", "D"]
declared_equations: ["u_t + c*u_x = D*u_xx"]
declared_assumptions: ["constant transport velocity", "constant diffusion coefficient", "no reaction/source terms"]
---

# Advection-Diffusion Systems

This is a benchmark fixture document summarizing well-known, generic
textbook physics for testing the Scientific Discovery Engine's
retrieval and knowledge-graph pipeline. It is not a reproduction of
any specific publication, and it does not state the parameter values
of any particular experiment.

## Overview

An advection-diffusion system combines directional bulk transport
(advection) with spatial spreading (diffusion). These models describe
quantities that are both carried along by a flow and spread out
locally -- for example a pollutant carried by a river current while
also diffusing outward from its initial release point.

## Governing equation

The standard one-dimensional advection-diffusion equation is:

    u_t + c*u_x = D*u_xx

where c is the advection velocity and D is the diffusion coefficient.
When c=0, this reduces exactly to pure diffusion; when D=0, it reduces
exactly to pure advection.

## Typical observational signatures

Systems governed by advection-diffusion typically show both: (a) the
amplitude-weighted spatial centroid drifting over time (the advective
signature), and (b) spreading/flattening of local features and
amplitude decay (the diffusive signature). If the advective component
is very small, the observed behavior can look almost indistinguishable
from pure diffusion, since the advective term's effect vanishes
continuously as c approaches zero.

## Typical assumptions

This model assumes a constant transport velocity, a constant diffusion
coefficient, and no additional local reaction/source term.
