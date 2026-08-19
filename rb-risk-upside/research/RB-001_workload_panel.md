# RB-001 — 2026 situation verification and the RB-week workload panel

**Verdict: KEEP.** The panel is built, the as-of guarantee is proved by a
runnable check, and the situation audit found one team assignment that memory
would have got wrong.

Reproduce: `./.venv/Scripts/python.exe rb-risk-upside/scripts/rb001_panel.py`

---

## 1. The 2026 situations, verified from data

Nothing below is from recall. Sources: `nfl_data.load_players()`,
`nflreadpy.load_rosters([2024,2025,2026])`, `nfl_data.load_depth_charts()`,
`markets.load_ffc_adp(2026)`, `markets.load_cbs_projections(2026,"RB")`.

| Player | 2025 team | **2026 team** | Depth slot | ADP (FFC 12-tm std) | Pos rank |
|---|---|---|---|---|---|
| D'Andre Swift | CHI | CHI | RB1 | 41.9 (sd 5.1) | RB20 |
| Quinshon Judkins | CLE | CLE | RB1 | 49.1 (sd 6.7) | RB22 |
| David Montgomery | **DET** | **HOU** | RB1 | 54.6 (sd 6.7) | RB24 |
| Tony Pollard | TEN | TEN | RB1 | 63.7 (sd 5.6) | RB26 |

**Montgomery changed teams: Detroit → Houston.** This is the finding that
justifies the instruction not to trust memory. It is confirmed independently by
three sources: the master player table's `latest_team`, the 2026 season roster,
and the March 2026 depth chart, and corroborated by FFC ADP listing him as HOU.
Every other assignment is unchanged from 2025.

ADP window 2026-08-11 to 2026-08-18, 1,466 drafts. All four sit in a tight band,
overall picks 42–64: rounds 4 through 6 of a 12-team draft.

### The room each one walks into (2026 roster + Mar-2026 depth chart)

| Player | Team | Competition that matters | Their 2025 usage |
|---|---|---|---|
| Swift | CHI | Kyle Monangai (2025 R7, pick 233) | 169 carries, 33.3% carry share, **12 goal-line carries vs Swift's 14** |
| Judkins | CLE | Dylan Sampson (2025 R4, pick 126) | 65 carries but 40 targets — a passing-down complement, 0 goal-line carries |
| Montgomery | HOU | Woody Marks (2025 R4, pick 116) | **196 carries, 43.6% carry share, 16 games** — an incumbent, not a backup |
| Pollard | TEN | Tyjae Spears (2023 R3) + Nicholas Singleton (2026 R5, pick 165) | Spears 72 carries / 50 targets |

**The one genuinely ambiguous situation is Montgomery.** The depth chart puts him
at RB1, but that snapshot is dated 2026-03-14, immediately after he signed and
before the 2026 draft. Woody Marks handled 196 carries and a 43.6% carry share
for Houston in 2025. A March depth chart written the week a veteran signs is a
weak signal about an August committee. RB-003 treats this as a distribution, not
a fact.

Swift's situation carries a quieter version of the same problem: Monangai took a
third of the carries and drew 12 goal-line carries to Swift's 14 in 2025. Swift
holds the role but not the scoring role exclusively.

### Consensus, rescored to this league

CBS publishes under PPR; components were rescored with `scoring.Scoring()`.

| Player | CBS games | rush att | rush yds | rush TD | rec | rec yds | **non-PPR pts** |
|---|---|---|---|---|---|---|---|
| Swift | 17 | 247 | 1117 | 9 | 36 | 325 | 202.2 |
| Judkins | 17 | 276 | 973 | 10 | 32 | 230 | 178.3 |
| Montgomery | 17 | 263 | 1012 | 9 | 27 | 211 | 172.3 |
| Pollard | 17 | 227 | 992 | 6 | 34 | 215 | 150.7 |

CBS projects **all four for a healthy 17 games**. That single assumption is worth
19–27 points per player and is the main reason RB-004 disagrees with consensus.

### 2023–2025 usage, from the panel

| Player | Season | Tm | G | Fin | Pts | Carries/g | Carry share | RZ carry share | GL carries/g | Team implied total |
|---|---|---|---|---|---|---|---|---|---|---|
| Swift | 2023 | PHI | 16 | RB23 | 160.3 | 14.3 | .470 | .349 | 0.88 | 25.5 |
| Swift | 2024 | CHI | 17 | RB19 | 172.5 | 14.9 | .587 | .517 | 0.59 | 20.3 |
| Swift | 2025 | CHI | 16 | RB14 | 194.6 | 13.9 | .474 | .444 | 0.88 | 22.7 |
| Montgomery | 2023 | DET | 14 | RB9 | 191.2 | 15.6 | .530 | .557 | 1.43 | 25.4 |
| Montgomery | 2024 | DET | 14 | RB17 | 185.7 | 13.2 | .421 | .505 | 1.29 | 27.8 |
| Montgomery | 2025 | DET | 17 | RB27 | 142.9 | **9.3** | **.357** | .380 | 1.00 | 26.9 |
| Judkins | 2025 | CLE | 14 | RB26 | 143.8 | 16.4 | **.657** | **.647** | 0.86 | **16.7** |
| Pollard | 2023 | DAL | 17 | RB18 | 167.6 | 14.8 | .536 | .569 | 1.06 | 26.0 |
| Pollard | 2024 | TEN | 16 | RB22 | 159.7 | 16.3 | .618 | .576 | 0.38 | 18.8 |
| Pollard | 2025 | TEN | 17 | RB23 | 152.8 | 14.2 | .640 | .422 | **0.24** | **17.0** |

Three things jump out and drive everything downstream:

1. **Montgomery's role collapsed in 2025** — 9.3 carries per game, a 35.7% carry
   share, RB27. He is not a lead back coming off a lead-back season; he is a
   committee back who changed teams.
2. **Judkins owns the largest share of anyone here** (65.7% of carries, 64.7% of
   red-zone carries as a rookie) but on **the worst offence** (16.7 implied team
   total). Share is elite, environment is poor.
3. **Pollard has almost no goal-line role** — 0.24 goal-line carries per game in
   2025, down from 1.06 in Dallas. He gets volume without the scoring plays,
   which caps his ceiling structurally rather than by luck.

---

## 2. The panel

`rb-risk-upside/src/rb_features.py` → `rb-risk-upside/data/rb_week_panel.parquet`

- **12,003 RB-weeks, 392 players, 2018–2025**, regular season only.
- Sources joined: `load_player_stats` (box score, rescored non-PPR via
  `scoring.skill_points`), `load_snap_counts` (via a pfr↔gsis crosswalk from
  `load_players`), play-by-play red-zone/goal-line counts, and
  `team_game_context` for spread / total / implied team total.
- Play-by-play is read one season at a time with 11 columns pruned out of 372,
  the same memory discipline as `qb-pairing/scripts/wave1_advanced_qb_vars.py`,
  because a whole pbp season does not fit alongside the rest at ~1.7 GB free.

### No look-ahead — how it is enforced and proved

Every rolling statistic is `.shift(1).rolling_*(...).over("player_id")`, copied
from `qb-pairing/src/qbpair/features/panel.py`. Market variables (spread, total,
implied team total) are closing lines that exist before kickoff and are used
unshifted, exactly as the QB panel does.

`rb_features._demo()` is the runnable proof and it asserts two things:

1. every player's first-ever panel game has **null** for all rolling features
   (`f_snap_share`, `f_fp`), so nothing is seeded from the future;
2. for a player whose career begins inside the panel, `f_carries` at game *k*
   equals the mean of his carries in games *k−8 … k−1* exactly, checked to 1e-9
   for every game of his career.

The test deliberately restricts case 2 to players whose careers start inside the
panel: `build_panel` loads one extra prior season for form history, so a veteran's
first panel row legitimately carries prior-season form. An earlier version of the
check failed on exactly this and the check was wrong, not the panel.

### Coverage

| Column group | Missing |
|---|---|
| `pbp_rush_td` | 10.2% — RB-games with zero rushing attempts; correct, not a defect |
| rolling `f_*` features | 2.4% — first career game of each player, by design |
| market context | 0.77% — a handful of games with no closing line |
| `snap_share` | 0.03% |

---

## 3. Shared-module repairs made along the way

Two real bugs in `shared/`, both found by using them rather than trusting them:

1. `nfl_data` had no `load_players` or `load_rosters_weekly` despite both being
   available in nflreadpy. Added, cached like every other loader.
2. **`markets.load_cbs_projections` was silently broken for every position except
   QB.** The column schema was hard-coded to CBS's QB table layout, so an RB
   request returned an empty frame. Added per-position schemas for RB/WR/TE; the
   QB path is byte-identical to before.

Also worth recording as a data limit for the next lab:
**`load_rosters_weekly` refuses any season after 2025** (`Season must be between
2002 and 2025`). Weekly roster membership for the upcoming season does not
exist. Use `load_rosters(seasons=[2026])` for team assignment.

---

## What this does not establish

The panel is a measurement instrument, not a finding. It says nothing yet about
which workload components are worth using — that is RB-002, and the answer there
is less flattering than the stability numbers alone suggest.
