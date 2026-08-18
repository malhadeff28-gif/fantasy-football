"""Phase 1 gate: reconstruct one historical QB-week from pregame data only.

Proves the walking skeleton end to end -- source -> QB-week record -> projection
-> scored against the actual result -- and audits that no information dated at
or after kickoff was used to build the features.

Usage:  python scripts/walk_one_qb_week.py [SEASON] [WEEK] [QB NAME]
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbpair.data import loaders                      # noqa: E402
from qbpair.features import pregame                  # noqa: E402

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
WEEK = int(sys.argv[2]) if len(sys.argv) > 2 else 12
QB = sys.argv[3] if len(sys.argv) > 3 else "Josh Allen"

BAR = "=" * 74


def main() -> int:
    schedules = loaders.load_schedules()
    stats = loaders.load_player_stats(list(range(SEASON - 2, SEASON + 1))).filter(
        pl.col("season_type") == "REG"
    )

    print(BAR)
    print(f" WALKING SKELETON -- {QB}, {SEASON} Week {WEEK}")
    print(BAR)

    feats = pregame.build_qb_week(stats, schedules, SEASON, WEEK)
    row = feats.filter(pl.col("player_display_name") == QB)
    if row.is_empty():
        print(f"!! {QB} has no pregame record for {SEASON} wk {WEEK}")
        return 1
    r = row.row(0, named=True)

    kickoff = r["gameday"]
    print(f"\n[1] GAME CONTEXT  (published before kickoff {kickoff})")
    print(f"    {r['team']} {'vs' if r['is_home'] else '@'} {r['opponent']}   game_id={r['game_id']}")
    print(f"    spread (team persp.) : {r['team_spread']:+.1f}")
    print(f"    game total           : {r['total_line']:.1f}")
    print(f"    implied team total   : {r['implied_team_total']:.2f}")
    print(f"    implied opp total    : {r['implied_opp_total']:.2f}")
    print(f"    rest days / indoors  : {r['rest']} / {bool(r['indoors'])}")

    print(f"\n[2] QB FORM  (last {int(r['form_games'])} games, all strictly prior)")
    print(f"    fantasy pts / game   : {r['form_fp_mean']:.2f}  (sd {r['form_fp_sd']:.2f})")
    print(f"    pass att / rush att  : {r['form_pass_att']:.1f} / {r['form_rush_att']:.1f}")
    print(f"    rushing share of pts : {r['rush_fp_share']:.1%}")
    print(f"    career games observed: {int(r['career_games'])}")

    print(f"\n[3] OPPONENT ({r['opponent']}) QB DEFENCE  (prior games only)")
    print(f"    games observed       : {int(r['def_games'])}")
    print(f"    raw QB pts allowed   : {r['def_fp_allowed_raw']:.2f}")
    print(f"    shrunk toward league : {r['def_fp_allowed']:.2f}  (league {r['def_league_mean']:.2f})")
    print(f"    matchup adjustment   : {r['def_adj']:+.2f} pts")

    proj = r["form_fp_mean"] + r["def_adj"]
    print(f"\n[4] NAIVE PREGAME PROJECTION  = form {r['form_fp_mean']:.2f} "
          f"{r['def_adj']:+.2f} matchup = {proj:.2f}")

    # --- leakage audit ------------------------------------------------------
    hist = pregame.history(stats, SEASON, WEEK)
    used = (
        hist.join(schedules.select(["game_id", "gameday"]), on="game_id", how="left")
        .select(pl.col("gameday").max().alias("latest"))
        .item()
    )
    print(f"\n[5] LEAKAGE AUDIT")
    print(f"    latest game date in feature history : {used}")
    print(f"    target kickoff date                 : {kickoff}")
    ok = str(used) < str(kickoff)
    print(f"    strictly before kickoff             : {'PASS' if ok else 'FAIL'}")
    assert ok, "LOOK-AHEAD DETECTED"

    # --- reveal outcome (never used above) ----------------------------------
    actual = stats.filter(
        (pl.col("season") == SEASON)
        & (pl.col("week") == WEEK)
        & (pl.col("player_display_name") == QB)
    )
    print(f"\n[6] ACTUAL RESULT  (held out until now)")
    if actual.is_empty():
        print("    did not play")
        return 0
    a = actual.row(0, named=True)
    print(f"    {a['completions']}/{a['attempts']}, {a['passing_yards']} yd, "
          f"{a['passing_tds']} TD, {a['passing_interceptions']} INT, "
          f"{a['carries']} car / {a['rushing_yards']} rush yd, {a['rushing_tds']} rush TD")
    print(f"    actual fantasy points : {a['fantasy_points']:.2f}")
    print(f"    projection error      : {proj - a['fantasy_points']:+.2f}")
    print(f"\n{BAR}\n PHASE 1 GATE: source -> QB-week -> projection -> scored. OK\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
