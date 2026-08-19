# RB-003 — Workload scenarios from historical comparables

**Verdict: KEEP.** The scenario distributions are stable across every choice of
neighbourhood size tested and they separate the four backs far more sharply than
their eight-pick ADP spread does. The headline is a **negative on the project's
own premise**: nobody here has a strongly positively-skewed workload
distribution that consensus is missing. The one back whose distribution *is*
positively skewed is skewed up from a very low base.

Reproduce: `./.venv/Scripts/python.exe rb-risk-upside/scripts/rb003_scenarios.py`

---

## 1. Method, and what stops it cheating

For every RB in every season 2019–2025 a **pre-season situation vector** is built
from information that existed before that season kicked off:

| Feature | weight | source |
|---|---|---|
| prior touches per game | 2.0 | season *t−1* rollup |
| prior carries per game | 1.5 | season *t−1* |
| prior snap share | 1.0 | season *t−1* |
| prior RB target share | 0.5 | season *t−1* |
| prior games played | 0.5 | season *t−1* |
| years of experience | 0.5 | roster |
| **ADP positional rank** | 1.5 | FFC pre-season |
| **competition draft capital** | 1.0 | best draft slot among the other backs in the room |
| **competition prior touches/gm** | 1.0 | the most any teammate handled last year |
| team change | +0.6 distance penalty | roster *t* vs *t−1* |

Weights were **fixed in advance** to reflect what a drafter actually looks at.
They were not tuned against outcomes; no search was run over them, so there is no
multiple-testing debt to pay here.

Two guards, both asserted in `rb_simulate._demo()`:

1. The comparable pool for a target season contains **only outcome seasons
   strictly earlier** than the target's, so the whole procedure is usable
   walk-forward and is in fact backtested that way in RB-004.
2. Outcomes are joined **after** the k nearest neighbours have been selected.
   Selection cannot see what happened.

k = 30. Sensitivity to k is reported below and is the main robustness check.

---

## 2. The four situation vectors

| | Swift (CHI) | Pollard (TEN) | Judkins (CLE) | Montgomery (HOU) |
|---|---|---|---|---|
| ADP / pos rank | 41.9 / RB20 | 63.7 / RB26 | 49.1 / RB22 | 54.6 / RB24 |
| prior touches/gm | 16.9 | 16.7 | **19.0** | **11.0** |
| prior snap share | 0.58 | 0.62 | 0.51 | **0.37** |
| prior games | 16 | 17 | 14 | 17 |
| years exp | 6 | 7 | **1** | 7 |
| best other back's draft slot | 115 | 81 | 126 | 116 |
| best other back's prior touches/gm | 11.7 | 10.5 | **7.8** | **14.5** |
| changed team | no | no | no | **yes** |

Judkins has the biggest prior role *and* the weakest incumbent competition.
Montgomery has the smallest prior role, a team change, and the strongest
incumbent competition of the four — Woody Marks handled 14.5 touches per game for
Houston last year.

---

## 3. Comparables and what actually happened to them

**Swift** — nearest: Ezekiel Elliott 2022 (RB17), James Conner 2023 (RB15),
Raheem Mostert 2024 (RB60), Tony Pollard 2025 (RB23), Swift's own 2025 (RB14),
Aaron Jones 2023 (RB36), James Cook 2025 (RB5), Nick Chubb 2022 (RB5).
Role ratio next/prior: median **1.00**, p10 0.75, p90 1.26. Played under 13
games: **23%**.

**Pollard** — nearest: Mostert 2024 (RB60), Conner 2024 (RB10), Pollard 2025
(RB23), Mark Ingram 2020 (RB70), Conner 2023 (RB15), Rhamondre Stevenson 2025
(RB25), Conner 2025 (3 games, RB74), Mostert 2023 (RB2).
Role ratio: median **0.99**, p10 **0.52**, p90 1.18. Under 13 games: 23%.

**Judkins** — nearest: Marlon Mack 2019 (RB17), David Montgomery 2020 (RB6),
Antonio Gibson 2022 (RB35), Najee Harris 2023 (RB20), Jonathan Taylor 2021
(RB1), James Cook 2024 (RB7), Josh Jacobs 2020 (RB8), Leonard Fournette 2019
(RB13), Chase Brown 2025 (RB10).
Role ratio: median **1.00**, p10 0.71, p90 1.24. Under 13 games: **20%**.

**Montgomery** — nearest: Mark Ingram 2019 (RB8), Tevin Coleman 2019 (RB32),
Latavius Murray 2019 (RB30), D'Onta Foreman 2023 (RB43), Matt Breida 2020
(RB80), LeSean McCoy 2019 (RB41), David Johnson 2020 (RB19), Cam Akers 2023
(RB68), Zack Moss 2024 (RB55), Jordan Mason 2025 (RB33), Phillip Lindsay 2021
(RB74).
Role ratio: median **0.82**, p10 0.48, p90 1.37. Under 13 games: **33%**.

The Montgomery comp set is the profile the market is not pricing: an ageing back
coming off a reduced role who signs somewhere new. The median such back landed
around **RB40**. Two of thirty became lead backs (Ingram 2019, Johnson 2020) —
that is the real upside case, and it is roughly a one-in-fifteen event, not the
base case.

---

## 4. Role scenarios (Wilson 95% intervals, n = 30 each)

Bands are named for what a draft room calls them, not carved out of the outcome
distribution, so the probabilities are informative rather than 25/50/25 by
construction.

| Band (touches/gm) | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| **bellcow** (18+) | 0.43 [.27,.61] | 0.33 [.19,.51] | **0.47** [.30,.64] | **0.07** [.02,.21] |
| **lead** (13–18) | 0.47 [.30,.64] | 0.47 [.30,.64] | 0.43 [.27,.61] | 0.30 [.17,.48] |
| **committee** (9–13) | 0.03 [.01,.17] | 0.07 [.02,.21] | 0.10 [.04,.26] | **0.40** [.25,.58] |
| **backup** (<9) | 0.07 [.02,.21] | 0.13 [.05,.30] | **0.00** [.00,.11] | 0.23 [.12,.41] |
| median pts if bellcow | 212 | 209 | 185 | 199 |
| median pts if lead | 141 | 148 | 149 | 143 |

The intervals are wide — 30 comparables is 30 comparables, and every probability
here carries roughly ±0.17. **Swift, Pollard and Judkins overlap heavily.**
Montgomery does not overlap with any of them.

Note the one thing the bands reveal that a point projection cannot: Judkins has
**zero comparables who ended up a backup** and the smallest tail risk of losing
the job, while Swift and Pollard each have a real 7–13% chance of falling out of
the role entirely.

---

## 5. Downside / base / upside, and the skew question

Percentiles of the comparable set (20th / 50th / 80th):

| | touches/gm down–base–up | season points down–base–up | **skew** |
|---|---|---|---|
| Swift | 14.2 – 17.1 – 19.2 | 97 – 170 – 220 | 0.69 |
| Pollard | 13.3 – 16.7 – 18.8 | 70 – 157 – 208 | 0.58 |
| Judkins | 14.2 – 17.3 – 20.6 | 128 – 167 – 201 | 0.86 |
| Montgomery | 6.8 – 11.4 – 16.1 | 43 – 92 – 144 | **1.04** |

skew = (p80 − median) / (median − p20). Above 1 means the upside tail is longer
than the downside tail.

**This is a negative result on the project's stated hypothesis.** The edge was
supposed to come from finding a back whose workload distribution is more
positively skewed than consensus assumes. Three of the four are *negatively*
skewed — their downside tail is longer than their upside, which is the normal
shape for an established back whose role can collapse but cannot easily double.

The only positively-skewed distribution belongs to **Montgomery**, and it is
skewed up from a base of 92 points. A long right tail on a low median is a
lottery ticket, not a market inefficiency. Its expected value is still the worst
of the four by a wide margin.

Judkins's 0.86 is the interesting number: his distribution is the *tightest*
(128–201 covers the middle 60%), which is a floor story, not a ceiling story.

---

## 6. Robustness to k

Median comparable season points:

| k | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| 15 | 153 | 153 | 192 | 91 |
| 20 | 168 | 150 | 169 | 92 |
| 30 | 170 | 157 | 167 | 92 |
| 45 | 169 | 147 | 169 | 92 |
| 60 | 165 | 153 | 168 | 78 |

P(bellcow role, 18+ touches/gm):

| k | Swift | Pollard | Judkins | Montgomery |
|---|---|---|---|---|
| 15 | 0.33 | 0.20 | 0.53 | 0.00 |
| 20 | 0.35 | 0.20 | 0.45 | 0.00 |
| 30 | 0.43 | 0.33 | 0.47 | 0.07 |
| 45 | 0.47 | 0.29 | 0.47 | 0.07 |
| 60 | 0.45 | 0.32 | 0.47 | 0.07 |

The ordering never changes. Judkins ≥ Swift > Pollard ≫ Montgomery under every
k from 15 to 60. Swift's bellcow probability is the least stable figure (0.33 to
0.47) because his nearest neighbours are the most heterogeneous.

---

## Limitations

- **Thirty comparables.** Every probability in section 4 carries a ±0.15–0.18
  Wilson interval. Differences smaller than that between Swift, Pollard and
  Judkins should not be read as real.
- The pool is 2019–2025 only, so it inherits whatever the last seven years of
  NFL committee usage happened to look like.
- The competition feature uses the *best other back in the room* by draft slot
  and prior touches. It cannot see coaching intent, contract structure, or a
  training-camp report. Montgomery's depth-chart RB1 label from March 2026 is
  deliberately *not* used as a feature, because a March depth chart written the
  week a veteran signs is not evidence about an August committee — see RB-001.
- Comparables are matched, not identical. Judkins's is the weakest match in one
  respect: he is a second-year player and most of his neighbours are third- or
  fourth-year backs, because there are few second-year backs with a 65% carry
  share to compare him to.
