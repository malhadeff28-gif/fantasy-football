# QB Wave 2 - adversarial pass (staged, not yet launched)

Wave 1 is producing KEEP verdicts. Before any of them enter the final briefing,
they get attacked. Agreement is not evidence; a finding that survives a genuine
attempt to break it is worth more than one that was never tested.

Launch condition: a free memory slot (>= 2 GB free, <= 2 concurrent agents).

## Targets, ranked by how much rides on them

| # | Finding | Why it deserves attack |
| --- | --- | --- |
| 1 | **QB-012**: QB2 worth +9.7 (elite QB1) / +30.1 (mid QB1) net of pick cost | Drives the actual draft recommendation. Rests on a measured pick-value curve and a replacement assumption. |
| 2 | **QB-016**: markets add +1.85 pp over a consensus proxy | The proxy is a proxy. If it is systematically weaker than a real projection, the increment is inflated. |
| 3 | **QB-011/013**: QB2 worth +12.0 pts/season, weekly decision +4.9 | Both depend on the streaming-replacement baseline, which is a modelling choice. |
| 4 | **QB-008**: wind, venue, home/away effects | Weather coverage is 38-75%. Missing-not-at-random would bias it. |

## Specific attacks to run

**Against QB-012**
- Replacement level is set at the weekly median starter. Re-run at the 40th and
  60th percentiles. The +9.7 figure should move; report how far.
- The pick-value curve is a mean over hit and miss. Re-run using the median, and
  separately using only players who reached a starting lineup.
- Survivorship: the ADP set only contains players who were drafted. Quantify
  what that omits.

**Against QB-016**
- Build a deliberately *stronger* consensus proxy (add prior-season efficiency,
  team continuity). If the market increment shrinks materially, the +1.85 pp is
  partly a weak-baseline artefact.
- The week 10-18 concentration was flagged exploratory. Re-test it as a
  pre-registered single hypothesis on a holdout of seasons.

**Against QB-011/013**
- Give the single-QB manager a *sharper* streamer (75th percentile, not median).
  Both gaps should shrink. Report the new numbers.
- Check whether the +12.0 is driven by a small number of extreme QB1 injuries.
  Trim the top and bottom 5% of pairs and re-measure.

**Against QB-008**
- Test whether weather-missing rows differ systematically from weather-present
  rows on observables. If they do, the wind estimate is biased.

## Replication target

Independently rebuild the **QB-012 economics** from raw sources - no reuse of
`draft_value.py` - and check the pick-value curve reproduces. This is the single
number most load-bearing on the final recommendation.

## Rule

Any finding that does not survive its attack gets downgraded to UNCERTAIN or
REJECT in the experiment log, with the attack recorded alongside it.
