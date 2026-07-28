# ADR-001: Vertical — Bearings / Rotating Equipment Only

**Status:** Accepted  
**Date:** 2026-07-28  
**Author:** LiamCarPer

## Context

Predictive maintenance is a broad domain covering everything from rotating machinery to
stationary assets, thermal imaging, and oil analysis. A general-purpose PdM platform
would require expertise across too many failure modes, sensor types, and domain models
for a single engineer to build credibly.

## Decision

AetherPdM will target **exactly one vertical**: bearings and rotating equipment
(motors, pumps, compressors with rolling-element bearings).

This means:
- All models operate on vibration signals (accelerometers)
- Fault taxonomy is limited to: normal, inner race, outer race, ball/rolling element
- Feature engineering is informed by bearing kinematics (BPFO, BPFI, BSF, FTF)
- Supported sensor types: accelerometers (single-axis or tri-axial)
- Supported data sources: CWRU, Paderborn, synthetic — all vibration-based

## Consequences

**Positive:**
- Focused feature engineering — every feature has a physical bearing interpretation
- Sharper portfolio narrative: "I understand bearings end-to-end"
- Measurable B2B metrics: false alarm rate directly affects maintenance trust
- Easier to reason about domain shift and transfer learning (same physics)

**Negative:**
- Does not cover gearboxes, belts, hydraulic systems, or process parameters
- Must reject or clearly scope out non-bearing datasets
- ForgeEdge / SignalOps must define their own verticals

## Compliance

Every dataset ingested must pass the "bearing check": does it contain accelerometer
data from a rotating machine with rolling-element bearings? If no, it is out of scope.
