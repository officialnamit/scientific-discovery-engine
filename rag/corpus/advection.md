---
document_id: doc_advection
title: "Advective (Directional) Transport -- Benchmark Fixture"
authors: []
source: "benchmark fixture"
year: null
domain: advection
is_benchmark_fixture: true
declared_mechanism: advection
declared_variables: ["u", "x", "t"]
declared_parameters: ["c"]
declared_equations: ["u_t + c*u_x = 0"]
declared_assumptions: ["constant transport velocity", "no diffusive spreading", "no reaction/source terms"]
---

# Advective Transport

This is a benchmark fixture document summarizing well-known, generic
textbook physics for testing the Scientific Discovery Engine's
retrieval and knowledge-graph pipeline. It is not a reproduction of
any specific publication, and it does not state the parameter values
of any particular experiment.

## Overview

Advection is the transport of a quantity by the bulk motion of the
medium carrying it -- for example a dye carried along by a flowing
fluid at a constant velocity. Unlike diffusion, advection does not by
itself cause a profile to spread out or flatten; it translates the
profile through space while largely preserving its shape.

## Governing equation

The standard one-dimensional linear advection equation is:

    u_t + c*u_x = 0

where u(x,t) is the transported quantity and c is the (constant)
advection velocity (units of length/time). Its exact solution is any
function of the form u(x,t) = f(x - c*t): the initial profile f(x) is
simply translated by c*t without changing shape.

## Typical observational signatures

Systems governed by pure advection typically show: (a) the spatial
profile translates across the domain at a roughly constant apparent
speed; (b) the shape of local peaks/features is preserved rather than
flattening or spreading; (c) the overall amplitude of the profile does
not decay on its own (in the absence of boundary losses or additional
terms); (d) the amplitude-weighted spatial centroid of the quantity
moves roughly linearly in time.

## Typical assumptions

A pure advection model assumes a constant transport velocity, no
diffusive spreading term, and no local reaction/source term.
