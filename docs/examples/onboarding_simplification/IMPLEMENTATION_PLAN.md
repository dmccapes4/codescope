# Onboarding Simplification — Implementation Plan

> **Example** of what ProScope generates under `docs/onboarding_simplification/`.
> ProScope writes this automatically from `PROSCOPE_DOC` blocks in the model response.

## Product goal

New users reach the home screen in **under 60 seconds** with clear progress and minimal friction. This directly supports Day-1 retention and reduces support tickets about “stuck on signup.”

## Why now

Current onboarding has redundant welcome screens and scattered permission prompts. Analytics (if available) likely show drop-off before first value — we should validate with existing funnel events before cutting screens.

## Phased roadmap

| Phase | Outcome | Doc |
|-------|---------|-----|
| 1 | Inventory screens; remove or merge redundant steps | `STRATEGY_PHASE_1.md` |
| 2 | Single Compose screen for profile + permissions | `STRATEGY_PHASE_2.md` |
| 3 | Instrument time-to-home; A/B test copy | `STRATEGY_PHASE_3.md` |

## Success metrics

- Median time from app open → home screen &lt; 60s
- Onboarding completion rate (define event: `onboarding_complete`)
- Day-1 retention (7-day rolling)

## Risks

- Removing a screen that legal/compliance requires (permissions timing)
- Deep links to removed fragments may break until Navigation is updated

## Open questions

- Which permission prompts are mandatory on first launch vs deferrable?
- Do we have baseline funnel data, or do we need Phase 3 instrumentation first?

## Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| (ProScope session) | Phase 1 is audit-only, no code until plan approved | Reduces risk for non-technical stakeholders |
