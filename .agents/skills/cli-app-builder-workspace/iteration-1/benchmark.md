# Skill Benchmark: cli-app-builder

**Model**: <model-name>
**Date**: 2026-10-15T10:16:44Z
**Evals**: 1, 2, 3 (1 run each per configuration)

## Summary

| Metric | With Skill | Without Skill | Delta |
|--------|------------|---------------|-------|
| Pass Rate | 87% ± 12% | 60% ± 53% | +0.27 |
| Time | 549.3s ± 289.4s | 218.3s ± 107.0s | +331.0s |
| Tokens | 0 ± 0 | 0 ± 0 | +0 |

## Notes

- The with-skill configuration passed 86.7% of assertions versus 60.0% without the skill.
- The config-precedence baseline stopped before implementation; that raises the measured delta and is not a controlled quality comparison.
- Legacy-command assertions passed in both configurations, so they do not distinguish the skill's Typer/TOML guidance.
- With-skill runs averaged 549.3 seconds versus 218.3 seconds without the skill. Token counts were unavailable in task notifications.