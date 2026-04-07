# HQ Anchor and Route Protocol

This note locks the baseline naming and route roles for the instability-control project.

## Canonical Anchor

The canonical benchmark anchor is:

- `HQ_anchor_xgb1_unpruned`

Definition:

- unpruned mirror HQ feature bank
- evaluated with the mirror-style `XGB1` route
- rule override retained

This anchor stays fixed for the whole project.

## Why The Anchor Must Stay Fixed

The anchor is the benchmark continuity point.

It answers:

- does a new family or combo improve the benchmark model we already trust?

It should not change just because a pruned or transformed route is easier to integrate reasoning features into.

## Step 1 Route Roles

Step 1 now uses several route controls, not a single admission route:

- `anchor_xgb1_unpruned`
- `raw_lr_base`
- `raw_xgb1`
- `pls_lr_n6`
- `pls_mlp4_n6`

The Step 1 question is:

- what fails raw, and what only becomes usable after transformation?

It is not:

- which family gets admitted by a raw-gain gate?

In all Step 1 tables, every non-`HQ` row should be read as:

- `HQ + family`
- `HQ + combo`

not as a family-only model.

## XGB1 Plus Reasoning

`XGB1 + reasoning` should still be run.

Its role is:

- benchmark continuity check against the frozen anchor

That route is evidence, not permission to redefine the baseline.

## Route Variants

Pruned or transformed HQ variants should be named as route variants, for example:

- `HQ_pruned_*`
- `HQ_pls_*`

These variants are valid experimental routes.

They must never replace `HQ_anchor_xgb1_unpruned` as the canonical baseline.

## What Went Wrong Previously

The earlier drift was:

- prune HQ to make reasoning-family integration easier
- then treat the pruned HQ variant as the new baseline

That changed the question.

Instead of asking:

- does a reasoning unit improve the benchmark anchor?

it became:

- does a reasoning unit improve a route that has already been modified in response to the same instability problem?

That is a valid route experiment, but not a valid baseline definition.

## Working Rule

From this point on:

- `HQ_anchor_xgb1_unpruned` is the only canonical benchmark baseline
- raw and transformed HQ variants are route controls
- Step 1 uses raw controls plus transformed controls
- Step 1 does not use sequential admission as its governing logic
