# Phase 1 — Audit & remove redundant screens

## What we will build

A written map of every screen in the onboarding flow (Activities, Fragments, Compose destinations) and a short list of screens we recommend **removing or merging**. No user-facing change until you approve the list.

## Why this approach

We read the existing Navigation graph and entry Activities before changing UX. That way we do not break deep links, notification routes, or Play Store demo paths. Jetpack Navigation is the “map” of how screens connect — we follow that map instead of guessing.

## Alternatives considered

| Alternative | Why rejected |
|-------------|--------------|
| Rewrite onboarding in one sprint | Too risky without knowing what each screen does |
| Keep all screens, only change copy | Does not hit the 60-second goal |

## UX implications

- Fewer taps before first value
- Users may see fewer “welcome” moments — we should keep one clear value proposition screen if brand requires it

## Tech debt & future implications

- Merged screens may later need a shared **ViewModel** (state holder) so profile and permissions stay in sync
- Removing Fragments may leave unused layout XML until a cleanup pass

## Uncertainty / open questions

- We don't know yet which screens are legally required vs cosmetic — needs product/legal input
- Unclear if any onboarding step is tied to backend feature flags

## References

- Parent plan: `IMPLEMENTATION_PLAN.md`
- Code areas to inspect: Navigation XML, `MainActivity`, onboarding package (see ProScope context sections)
