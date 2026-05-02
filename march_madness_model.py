from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TOURNAMENT_PATH = Path(__file__).with_name("tournament_2026_teams.csv")
MODEL_PATH = Path(__file__).with_name("historical_model_weights.json")
REGION_ORDER = ["East", "West", "South", "Midwest"]
REGIONAL_SEED_ORDER = [
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]
FINAL_FOUR_PAIRINGS = [("East", "South"), ("West", "Midwest")]


@dataclass(frozen=True)
class Team:
    name: str
    region: str
    seed: int
    slot: str
    play_in: bool
    play_in_group: str
    games: int
    win_pct: float
    ppg: float
    opp_ppg: float
    scoring_margin: float
    fg_pct: float
    three_pt_pct: float
    ft_pct: float
    reb_pg: float
    ast_pg: float
    to_pg: float
    ast_to_ratio: float
    srs: float
    sos: float
    pace: float
    off_rtg: float
    ts_pct: float
    trb_pct: float
    ast_pct: float
    efg_pct: float
    tov_pct: float
    orb_pct: float
    power_rating: float


def load_learned_model(path: Path = MODEL_PATH) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_teams(path: Path = TOURNAMENT_PATH) -> list[Team]:
    df = pd.read_csv(path)
    rows = df.rename(columns={"team": "name"}).to_dict(orient="records")
    teams = [
        Team(
            name=row["name"],
            region=row["region"],
            seed=int(row["seed"]),
            slot=row["slot"],
            play_in=bool(row["play_in"]),
            play_in_group=row["play_in_group"] if isinstance(row["play_in_group"], str) else "",
            games=int(row["games"]),
            win_pct=float(row["win_pct"]),
            ppg=float(row["ppg"]),
            opp_ppg=float(row["opp_ppg"]),
            scoring_margin=float(row["scoring_margin"]),
            fg_pct=float(row["fg_pct"]),
            three_pt_pct=float(row["three_pt_pct"]),
            ft_pct=float(row["ft_pct"]),
            reb_pg=float(row["reb_pg"]),
            ast_pg=float(row["ast_pg"]),
            to_pg=float(row["to_pg"]),
            ast_to_ratio=float(row["ast_to_ratio"]),
            srs=float(row["srs"]),
            sos=float(row["sos"]),
            pace=float(row["pace"]),
            off_rtg=float(row["off_rtg"]),
            ts_pct=float(row["ts_pct"]),
            trb_pct=float(row["trb_pct"]),
            ast_pct=float(row["ast_pct"]),
            efg_pct=float(row["efg_pct"]),
            tov_pct=float(row["tov_pct"]),
            orb_pct=float(row["orb_pct"]),
            power_rating=float(row["power_rating"]),
        )
        for row in rows
    ]
    if len(teams) != 68:
        raise ValueError(f"Expected 68 tournament teams, found {len(teams)}")
    return teams


def learned_feature_gaps(team_a: Team, team_b: Team) -> dict[str, float]:
    return {
        "seed_gap": team_b.seed - team_a.seed,
        "scoring_margin_gap": team_a.scoring_margin - team_b.scoring_margin,
        "win_pct_gap": team_a.win_pct - team_b.win_pct,
        "fg_pct_gap": team_a.fg_pct - team_b.fg_pct,
        "three_pt_pct_gap": team_a.three_pt_pct - team_b.three_pt_pct,
        "ft_pct_gap": team_a.ft_pct - team_b.ft_pct,
        "reb_pg_gap": team_a.reb_pg - team_b.reb_pg,
        "ast_pg_gap": team_a.ast_pg - team_b.ast_pg,
        "tov_pg_gap": team_a.to_pg - team_b.to_pg,
        "ast_to_ratio_gap": team_a.ast_to_ratio - team_b.ast_to_ratio,
        "srs_gap": team_a.srs - team_b.srs,
        "sos_gap": team_a.sos - team_b.sos,
        "pace_gap": team_a.pace - team_b.pace,
        "off_rtg_gap": team_a.off_rtg - team_b.off_rtg,
        "ts_pct_gap": team_a.ts_pct - team_b.ts_pct,
        "trb_pct_gap": team_a.trb_pct - team_b.trb_pct,
        "ast_pct_gap": team_a.ast_pct - team_b.ast_pct,
        "efg_pct_gap": team_a.efg_pct - team_b.efg_pct,
        "tov_pct_rate_gap": team_a.tov_pct - team_b.tov_pct,
        "orb_pct_gap": team_a.orb_pct - team_b.orb_pct,
    }


def heuristic_matchup_score(team_a: Team, team_b: Team) -> float:
    # Composite edge: team quality, margin, shooting, ball security, rebounding, and seed prior.
    score = 0.0
    score += (team_a.power_rating - team_b.power_rating) * 0.60
    score += (team_a.scoring_margin - team_b.scoring_margin) * 0.85
    score += (team_a.win_pct - team_b.win_pct) * 0.18
    score += (team_a.fg_pct - team_b.fg_pct) * 0.28
    score += (team_a.three_pt_pct - team_b.three_pt_pct) * 0.18
    score += (team_a.ft_pct - team_b.ft_pct) * 0.08
    score += (team_a.reb_pg - team_b.reb_pg) * 0.12
    score += (team_a.ast_to_ratio - team_b.ast_to_ratio) * 1.10
    score += (team_b.to_pg - team_a.to_pg) * 0.10
    score += (team_b.seed - team_a.seed) * 0.55

    # Light historical upset priors.
    if team_a.seed == 12 and team_b.seed == 5:
        score += 0.75
    if team_a.seed == 11 and team_b.seed == 6:
        score += 0.45
    if team_a.seed == 10 and team_b.seed == 7:
        score += 0.20
    if team_a.seed == 13 and team_b.seed == 4:
        score += 0.18
    if team_a.seed == 9 and team_b.seed == 8:
        score += 0.10

    return score


def matchup_score(team_a: Team, team_b: Team, learned_model: dict[str, object] | None) -> float:
    if not learned_model:
        return heuristic_matchup_score(team_a, team_b) / 8.0

    feature_gaps = learned_feature_gaps(team_a, team_b)
    coefficients = learned_model["coefficients"]
    means = learned_model["means"]
    stds = learned_model["stds"]
    feature_names = learned_model.get("feature_names", [])
    score = float(learned_model["intercept"])

    for feature_name in feature_names:
        std = float(stds[feature_name]) or 1.0
        normalized_value = (feature_gaps[feature_name] - float(means[feature_name])) / std
        score += float(coefficients[feature_name]) * normalized_value

    return score


def win_probability(
    team_a: Team,
    team_b: Team,
    volatility: float = 1.0,
    learned_model: dict[str, object] | None = None,
) -> float:
    score = matchup_score(team_a, team_b, learned_model=learned_model) / max(volatility, 0.2)
    return 1.0 / (1.0 + math.exp(-score))


def pick_winner(
    team_a: Team,
    team_b: Team,
    rng: random.Random,
    volatility: float = 1.0,
    learned_model: dict[str, object] | None = None,
) -> Team:
    return team_a if rng.random() < win_probability(team_a, team_b, volatility, learned_model=learned_model) else team_b


def simulate_and_record_game(
    team_a: Team,
    team_b: Team,
    rng: random.Random,
    volatility: float,
    learned_model: dict[str, object] | None,
) -> tuple[Team, dict[str, Team]]:
    winner = pick_winner(team_a, team_b, rng, volatility, learned_model=learned_model)
    return winner, {"team_a": team_a, "team_b": team_b, "winner": winner}


def resolve_play_in_teams(
    teams: list[Team],
    rng: random.Random,
    volatility: float,
    learned_model: dict[str, object] | None,
) -> tuple[list[Team], list[dict[str, Team]]]:
    by_group: dict[str, list[Team]] = {}
    resolved: list[Team] = []
    play_in_games: list[dict[str, Team]] = []

    for team in teams:
        if team.play_in:
            by_group.setdefault(team.play_in_group, []).append(team)
        else:
            resolved.append(team)

    for _, contenders in sorted(by_group.items()):
        if len(contenders) != 2:
            raise ValueError(f"Play-in group must have 2 teams: {contenders}")
        winner, game = simulate_and_record_game(contenders[0], contenders[1], rng, volatility, learned_model=learned_model)
        resolved.append(winner)
        play_in_games.append(game)

    return resolved, play_in_games


def build_region_bracket(resolved_teams: list[Team]) -> dict[str, list[Team]]:
    regions: dict[str, list[Team]] = {}
    for region in REGION_ORDER:
        region_teams = [team for team in resolved_teams if team.region == region]
        if len(region_teams) != 16:
            raise ValueError(f"Region {region} should contain 16 resolved teams, found {len(region_teams)}")
        regions[region] = sorted(region_teams, key=lambda team: (team.seed, team.name))
    return regions


def play_region(
    region_teams: list[Team],
    rng: random.Random,
    volatility: float,
    learned_model: dict[str, object] | None,
) -> tuple[Team, dict[str, list[Team]], dict[str, list[dict[str, Team]]]]:
    by_seed = {team.seed: team for team in region_teams}
    rounds: dict[str, list[Team]] = {}
    games_by_round: dict[str, list[dict[str, Team]]] = {}

    current = []
    round_games = []
    for a, b in REGIONAL_SEED_ORDER:
        winner, game = simulate_and_record_game(by_seed[a], by_seed[b], rng, volatility, learned_model=learned_model)
        current.append(winner)
        round_games.append(game)
    rounds["Round of 32"] = current
    games_by_round["Round of 64"] = round_games

    next_round = []
    round_games = []
    for i in range(0, len(current), 2):
        winner, game = simulate_and_record_game(current[i], current[i + 1], rng, volatility, learned_model=learned_model)
        next_round.append(winner)
        round_games.append(game)
    current = next_round
    rounds["Sweet 16"] = current
    games_by_round["Round of 32"] = round_games

    next_round = []
    round_games = []
    for i in range(0, len(current), 2):
        winner, game = simulate_and_record_game(current[i], current[i + 1], rng, volatility, learned_model=learned_model)
        next_round.append(winner)
        round_games.append(game)
    current = next_round
    rounds["Elite 8"] = current
    games_by_round["Sweet 16"] = round_games

    winner, game = simulate_and_record_game(current[0], current[1], rng, volatility, learned_model=learned_model)
    current = [winner]
    rounds["Final Four"] = current
    games_by_round["Elite 8"] = [game]

    return current[0], rounds, games_by_round


def simulate_tournament(
    teams: list[Team],
    rng: random.Random,
    volatility: float = 1.0,
    learned_model: dict[str, object] | None = None,
) -> dict[str, object]:
    resolved, play_in_games = resolve_play_in_teams(teams, rng, volatility, learned_model=learned_model)
    regions = build_region_bracket(resolved)

    round_results: dict[str, list[Team]] = {
        "Round of 64": resolved,
        "Round of 32": [],
        "Sweet 16": [],
        "Elite 8": [],
        "Final Four": [],
        "Championship": [],
        "Champion": [],
    }
    regional_champions: dict[str, Team] = {}
    region_game_logs: dict[str, dict[str, list[dict[str, Team]]]] = {}

    for region in REGION_ORDER:
        champion, rounds, game_logs = play_region(regions[region], rng, volatility, learned_model=learned_model)
        regional_champions[region] = champion
        region_game_logs[region] = game_logs
        for round_name, winners in rounds.items():
            round_results[round_name].extend(winners)

    semifinal_winners = []
    final_four_games = []
    for region_a, region_b in FINAL_FOUR_PAIRINGS:
        semifinal_winner, game = simulate_and_record_game(
            regional_champions[region_a],
            regional_champions[region_b],
            rng,
            volatility,
            learned_model=learned_model,
        )
        semifinal_winners.append(semifinal_winner)
        final_four_games.append(game)

    round_results["Championship"] = semifinal_winners
    champion, championship_game = simulate_and_record_game(
        semifinal_winners[0],
        semifinal_winners[1],
        rng,
        volatility,
        learned_model=learned_model,
    )
    round_results["Champion"] = [champion]

    return {
        "resolved_teams": resolved,
        "play_in_games": play_in_games,
        "regional_champions": regional_champions,
        "region_game_logs": region_game_logs,
        "final_four_games": final_four_games,
        "championship_game": championship_game,
        "round_results": round_results,
        "champion": champion,
    }


def simulate_many(
    teams: list[Team],
    sims: int,
    seed: int,
    volatility: float,
    learned_model: dict[str, object] | None,
) -> dict[str, Counter]:
    rng = random.Random(seed)
    counters = {round_name: Counter() for round_name in ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship", "Champion"]}

    for _ in range(sims):
        result = simulate_tournament(teams, rng, volatility, learned_model=learned_model)
        for round_name, winners in result["round_results"].items():
            counters[round_name].update(team.name for team in winners)

    return counters


def print_title_odds(teams: list[Team], counters: dict[str, Counter], sims: int, top_n: int = 16) -> None:
    print("\nTOP TITLE ODDS")
    print("--------------")
    ranked = sorted(teams, key=lambda team: counters["Champion"][team.name], reverse=True)
    seen: set[str] = set()

    for team in ranked:
        if team.name in seen:
            continue
        seen.add(team.name)
        print(
            f"{team.name:<18} "
            f"{team.region:<8} "
            f"{team.seed:>2} "
            f"title {counters['Champion'][team.name] / sims:>6.1%} "
            f"final {counters['Championship'][team.name] / sims:>6.1%} "
            f"ff {counters['Final Four'][team.name] / sims:>6.1%}"
        )
        if len(seen) >= top_n:
            break


def print_algorithm_summary(learned_model: dict[str, object] | None) -> None:
    print("\nWINNER LOGIC")
    print("------------")
    if learned_model:
        print("1. Use coefficients fit on historical NCAA tournament games.")
        print("2. Compare teams on seed, scoring margin, win%, shooting, rebounding, assists, turnovers, and A/TO ratio.")
        print("3. Standardize each feature gap with historical means and standard deviations.")
        print("4. Convert the learned logit score to a win probability.")
        print("5. Simulate the bracket repeatedly and use round-advance rates as prediction odds.")
    else:
        print("1. Build a power rating from season stats: scoring margin, win%, shooting, rebounding, assists, turnovers.")
        print("2. Compare two teams with a weighted edge score plus a seed prior.")
        print("3. Add small historical upset bumps for 12/5, 11/6, 10/7, 13/4, and 9/8 spots.")
        print("4. Convert the edge score to a logistic win probability.")
        print("5. Simulate the bracket repeatedly and use round-advance rates as prediction odds.")


def print_sample_bracket(result: dict[str, object]) -> None:
    print("\nSAMPLE BRACKET")
    print("--------------")
    if result["play_in_games"]:
        print("\nFirst Four")
        for game in result["play_in_games"]:
            print(
                f"({game['team_a'].seed}) {game['team_a'].name} vs. "
                f"({game['team_b'].seed}) {game['team_b'].name} -> "
                f"{game['winner'].name}"
            )

    for region in REGION_ORDER:
        print(f"\n{region} Region")
        region_games = result["region_game_logs"][region]
        for round_name in ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]:
            print(round_name)
            for game in region_games[round_name]:
                print(
                    f"({game['team_a'].seed}) {game['team_a'].name} vs. "
                    f"({game['team_b'].seed}) {game['team_b'].name} -> "
                    f"{game['winner'].name}"
                )

    print("\nFinal Four")
    for game in result["final_four_games"]:
        print(
            f"({game['team_a'].seed}) {game['team_a'].name} vs. "
            f"({game['team_b'].seed}) {game['team_b'].name} -> "
            f"{game['winner'].name}"
        )

    championship_game = result["championship_game"]
    champion = result["champion"]
    print("\nChampionship")
    print(
        f"({championship_game['team_a'].seed}) {championship_game['team_a'].name} vs. "
        f"({championship_game['team_b'].seed}) {championship_game['team_b'].name} -> "
        f"{championship_game['winner'].name}"
    )
    print(f"\nChampion: {champion.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2026 March Madness bracket predictor")
    parser.add_argument("--sims", type=int, default=10000, help="Number of tournament simulations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--volatility",
        type=float,
        default=1.0,
        help="Lower means more decisive favorite edges; higher means more chaos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teams = load_teams()
    learned_model = load_learned_model()
    counters = simulate_many(teams, sims=args.sims, seed=args.seed, volatility=args.volatility, learned_model=learned_model)
    bracket = simulate_tournament(teams, random.Random(args.seed), volatility=args.volatility, learned_model=learned_model)

    print_algorithm_summary(learned_model)
    print_title_odds(teams, counters, sims=args.sims)
    print_sample_bracket(bracket)


if __name__ == "__main__":
    main()
