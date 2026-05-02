from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


BASE_HEADERS = {"User-Agent": "Mozilla/5.0"}
CACHE_DIR = Path(__file__).with_name("cache")
TRAINING_CSV_PATH = Path(__file__).with_name("historical_tournament_games.csv")
MODEL_PATH = Path(__file__).with_name("historical_model_weights.json")

FEATURE_NAMES = [
    "seed_gap",
    "scoring_margin_gap",
    "win_pct_gap",
    "fg_pct_gap",
    "three_pt_pct_gap",
    "ft_pct_gap",
    "reb_pg_gap",
    "ast_pg_gap",
    "tov_pg_gap",
    "ast_to_ratio_gap",
    "srs_gap",
    "sos_gap",
    "pace_gap",
    "off_rtg_gap",
    "ts_pct_gap",
    "trb_pct_gap",
    "ast_pct_gap",
    "efg_pct_gap",
    "tov_pct_rate_gap",
    "orb_pct_gap",
]


@dataclass
class FitResult:
    intercept: float
    coefficients: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    train_log_loss: float
    validation_log_loss: float
    validation_accuracy: float
    regularization: float


def fetch_text(url: str, cache_path: Path | None = None, force_refresh: bool = False) -> str:
    if cache_path and cache_path.exists() and not force_refresh:
        return cache_path.read_text(encoding="utf-8")

    request = Request(url, headers=BASE_HEADERS)
    with urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return text


def clean_name(name: str) -> str:
    cleaned = " ".join(name.replace("\xa0", " ").split())
    cleaned = cleaned.removesuffix(" NCAA").strip()
    return cleaned


def parse_basic_school_stats(season: int, force_refresh: bool = False) -> pd.DataFrame:
    url = f"https://www.sports-reference.com/cbb/seasons/men/{season}-school-stats.html"
    cache_path = CACHE_DIR / f"{season}_basic_school_stats.html"
    html = fetch_text(url, cache_path=cache_path, force_refresh=force_refresh)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "basic_school_stats"})
    if table is None:
        raise ValueError(f"Could not find basic school stats table for season {season}")

    rows: list[dict[str, object]] = []
    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError(f"Could not find tbody for season {season}")

    for tr in tbody.find_all("tr"):
        if "class" in tr.attrs and "thead" in tr["class"]:
            continue

        school_cell = tr.find("td", {"data-stat": "school_name"})
        games_cell = tr.find("td", {"data-stat": "g"})
        if school_cell is None or games_cell is None:
            continue

        team = clean_name(school_cell.get_text(" ", strip=True))
        games = float(games_cell.get_text(strip=True))
        if games <= 0:
            continue

        def get_float(stat: str) -> float:
            cell = tr.find("td", {"data-stat": stat})
            return float(cell.get_text(strip=True)) if cell and cell.get_text(strip=True) else 0.0

        pts = get_float("pts")
        opp_pts = get_float("opp_pts")
        ast = get_float("ast")
        tov = get_float("tov")

        rows.append(
            {
                "season": season,
                "team": team,
                "games": games,
                "win_pct": get_float("win_loss_pct") * 100.0,
                "ppg": pts / games,
                "opp_ppg": opp_pts / games,
                "scoring_margin": (pts - opp_pts) / games,
                "fg_pct": get_float("fg_pct") * 100.0,
                "three_pt_pct": get_float("fg3_pct") * 100.0,
                "ft_pct": get_float("ft_pct") * 100.0,
                "reb_pg": get_float("trb") / games,
                "ast_pg": ast / games,
                "to_pg": tov / games,
                "ast_to_ratio": ast / tov if tov else ast,
            }
        )

    return pd.DataFrame(rows)


def parse_advanced_school_stats(season: int, force_refresh: bool = False) -> pd.DataFrame:
    url = f"https://www.sports-reference.com/cbb/seasons/men/{season}-advanced-school-stats.html"
    cache_path = CACHE_DIR / f"{season}_advanced_school_stats.html"
    html = fetch_text(url, cache_path=cache_path, force_refresh=force_refresh)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "adv_school_stats"})
    if table is None:
        raise ValueError(f"Could not find advanced school stats table for season {season}")

    rows: list[dict[str, object]] = []
    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError(f"Could not find advanced tbody for season {season}")

    for tr in tbody.find_all("tr"):
        if "class" in tr.attrs and "thead" in tr["class"]:
            continue

        school_cell = tr.find("td", {"data-stat": "school_name"})
        if school_cell is None:
            continue

        def get_float(stat: str) -> float:
            cell = tr.find("td", {"data-stat": stat})
            return float(cell.get_text(strip=True)) if cell and cell.get_text(strip=True) else 0.0

        rows.append(
            {
                "season": season,
                "team": clean_name(school_cell.get_text(" ", strip=True)),
                "srs": get_float("srs"),
                "sos": get_float("sos"),
                "pace": get_float("pace"),
                "off_rtg": get_float("off_rtg"),
                "ts_pct": get_float("ts_pct") * 100.0,
                "trb_pct": get_float("trb_pct"),
                "ast_pct": get_float("ast_pct"),
                "efg_pct": get_float("efg_pct") * 100.0,
                "tov_pct_rate": get_float("tov_pct"),
                "orb_pct": get_float("orb_pct"),
            }
        )

    return pd.DataFrame(rows)


def parse_tournament_games(season: int, force_refresh: bool = False) -> pd.DataFrame:
    url = f"https://www.sports-reference.com/cbb/postseason/men/{season}-ncaa.html"
    cache_path = CACHE_DIR / f"{season}_ncaa_tournament.html"
    html = fetch_text(url, cache_path=cache_path, force_refresh=force_refresh)
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []
    brackets = soup.find_all("div", {"id": "bracket"})
    if not brackets:
        raise ValueError(f"Could not find brackets for season {season}")

    for bracket in brackets:
        bracket_classes = bracket.get("class", [])
        if "team4" in bracket_classes:
            round_names = ["Final Four", "Championship"]
        else:
            round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]

        for round_index, round_div in enumerate(bracket.find_all("div", recursive=False)):
            if "class" not in round_div.attrs or "round" not in round_div["class"]:
                continue
            round_name = round_names[min(round_index, len(round_names) - 1)]

            for game_div in round_div.find_all("div", recursive=False):
                team_divs = [child for child in game_div.find_all("div", recursive=False)]
                if len(team_divs) != 2:
                    continue

                matchup = []
                for team_div in team_divs:
                    seed_span = team_div.find("span")
                    team_link = team_div.find("a", href=lambda href: href and "/cbb/schools/" in href)
                    score_links = team_div.find_all("a", href=lambda href: href and "/cbb/boxscores/" in href)
                    if seed_span is None or team_link is None or not score_links:
                        matchup = []
                        break

                    matchup.append(
                        {
                            "seed": int(seed_span.get_text(strip=True)),
                            "team": clean_name(team_link.get_text(" ", strip=True)),
                            "score": int(score_links[-1].get_text(strip=True)),
                        }
                    )

                if len(matchup) != 2:
                    continue

                rows.append(
                    {
                        "season": season,
                        "round": round_name,
                        "team_a": matchup[0]["team"],
                        "seed_a": matchup[0]["seed"],
                        "score_a": matchup[0]["score"],
                        "team_b": matchup[1]["team"],
                        "seed_b": matchup[1]["seed"],
                        "score_b": matchup[1]["score"],
                        "team_a_won": int(matchup[0]["score"] > matchup[1]["score"]),
                    }
                )

    return pd.DataFrame(rows)


def build_training_frame(seasons: Iterable[int], force_refresh: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for season in seasons:
        if season == 2020:
            continue

        stats = parse_basic_school_stats(season, force_refresh=force_refresh)
        advanced = parse_advanced_school_stats(season, force_refresh=force_refresh)
        stats = stats.merge(advanced, on=["season", "team"], how="inner")
        stats = stats.set_index("team")
        games = parse_tournament_games(season, force_refresh=force_refresh)

        for game in games.to_dict(orient="records"):
            if game["team_a"] not in stats.index or game["team_b"] not in stats.index:
                continue

            team_a = stats.loc[game["team_a"]]
            team_b = stats.loc[game["team_b"]]

            base_row = {
                "season": season,
                "round": game["round"],
                "team_a": game["team_a"],
                "team_b": game["team_b"],
                "label": game["team_a_won"],
                "seed_gap": game["seed_b"] - game["seed_a"],
                "scoring_margin_gap": team_a["scoring_margin"] - team_b["scoring_margin"],
                "win_pct_gap": team_a["win_pct"] - team_b["win_pct"],
                "fg_pct_gap": team_a["fg_pct"] - team_b["fg_pct"],
                "three_pt_pct_gap": team_a["three_pt_pct"] - team_b["three_pt_pct"],
                "ft_pct_gap": team_a["ft_pct"] - team_b["ft_pct"],
                "reb_pg_gap": team_a["reb_pg"] - team_b["reb_pg"],
                "ast_pg_gap": team_a["ast_pg"] - team_b["ast_pg"],
                "tov_pg_gap": team_a["to_pg"] - team_b["to_pg"],
                "ast_to_ratio_gap": team_a["ast_to_ratio"] - team_b["ast_to_ratio"],
                "srs_gap": team_a["srs"] - team_b["srs"],
                "sos_gap": team_a["sos"] - team_b["sos"],
                "pace_gap": team_a["pace"] - team_b["pace"],
                "off_rtg_gap": team_a["off_rtg"] - team_b["off_rtg"],
                "ts_pct_gap": team_a["ts_pct"] - team_b["ts_pct"],
                "trb_pct_gap": team_a["trb_pct"] - team_b["trb_pct"],
                "ast_pct_gap": team_a["ast_pct"] - team_b["ast_pct"],
                "efg_pct_gap": team_a["efg_pct"] - team_b["efg_pct"],
                "tov_pct_rate_gap": team_a["tov_pct_rate"] - team_b["tov_pct_rate"],
                "orb_pct_gap": team_a["orb_pct"] - team_b["orb_pct"],
            }
            frames.append(pd.DataFrame([base_row]))

            # Mirror each example so the fit is not sensitive to team ordering.
            mirrored = base_row.copy()
            mirrored["team_a"] = base_row["team_b"]
            mirrored["team_b"] = base_row["team_a"]
            mirrored["label"] = 1 - base_row["label"]
            for feature in FEATURE_NAMES:
                mirrored[feature] = -base_row[feature]
            frames.append(pd.DataFrame([mirrored]))

    if not frames:
        raise ValueError("No historical tournament training rows were built")

    training = pd.concat(frames, ignore_index=True)
    training.to_csv(TRAINING_CSV_PATH, index=False)
    return training


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probs = np.clip(y_prob, 1e-9, 1 - 1e-9)
    return float(-np.mean((y_true * np.log(probs)) + ((1 - y_true) * np.log(1 - probs))))


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    regularization: float,
    iterations: int = 6000,
    learning_rate: float = 0.05,
) -> FitResult:
    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    x_train_scaled = (x_train - means) / stds
    x_valid_scaled = (x_valid - means) / stds

    coefficients = np.zeros(x_train.shape[1], dtype=float)
    intercept = 0.0

    for _ in range(iterations):
        logits = intercept + (x_train_scaled @ coefficients)
        probs = sigmoid(logits)
        errors = probs - y_train

        grad_intercept = errors.mean()
        grad_coef = (x_train_scaled.T @ errors) / len(x_train_scaled)
        grad_coef += regularization * coefficients

        intercept -= learning_rate * grad_intercept
        coefficients -= learning_rate * grad_coef

    train_probs = sigmoid(intercept + (x_train_scaled @ coefficients))
    valid_probs = sigmoid(intercept + (x_valid_scaled @ coefficients))
    valid_pred = (valid_probs >= 0.5).astype(int)

    return FitResult(
        intercept=float(intercept),
        coefficients=coefficients,
        means=means,
        stds=stds,
        train_log_loss=log_loss(y_train, train_probs),
        validation_log_loss=log_loss(y_valid, valid_probs),
        validation_accuracy=float((valid_pred == y_valid).mean()),
        regularization=regularization,
    )


def choose_regularization(training: pd.DataFrame, validation_seasons: set[int]) -> FitResult:
    train_df = training[~training["season"].isin(validation_seasons)].copy()
    valid_df = training[training["season"].isin(validation_seasons)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Need both training and validation seasons to tune regularization")

    x_train = train_df[FEATURE_NAMES].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=float)
    x_valid = valid_df[FEATURE_NAMES].to_numpy(dtype=float)
    y_valid = valid_df["label"].to_numpy(dtype=float)

    candidates = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
    best: FitResult | None = None
    for regularization in candidates:
        result = fit_logistic_regression(x_train, y_train, x_valid, y_valid, regularization=regularization)
        if best is None or result.validation_log_loss < best.validation_log_loss:
            best = result

    if best is None:
        raise ValueError("Failed to fit any model")
    return best


def refit_full_model(training: pd.DataFrame, regularization: float) -> FitResult:
    x = training[FEATURE_NAMES].to_numpy(dtype=float)
    y = training["label"].to_numpy(dtype=float)
    return fit_logistic_regression(x, y, x, y, regularization=regularization)


def save_model(
    model: FitResult,
    seasons: list[int],
    validation_seasons: list[int],
    total_games: int,
) -> None:
    payload = {
        "source": "sports-reference historical NCAA tournament games",
        "seasons": seasons,
        "validation_seasons": validation_seasons,
        "feature_names": FEATURE_NAMES,
        "intercept": model.intercept,
        "coefficients": {name: float(value) for name, value in zip(FEATURE_NAMES, model.coefficients)},
        "means": {name: float(value) for name, value in zip(FEATURE_NAMES, model.means)},
        "stds": {name: float(value) for name, value in zip(FEATURE_NAMES, model.stds)},
        "train_log_loss": model.train_log_loss,
        "validation_log_loss": model.validation_log_loss,
        "validation_accuracy": model.validation_accuracy,
        "regularization": model.regularization,
        "training_rows": total_games,
    }
    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train March Madness matchup weights from historical tournaments")
    parser.add_argument("--start-season", type=int, default=2010, help="First season end year to include")
    parser.add_argument("--end-season", type=int, default=2025, help="Last season end year to include")
    parser.add_argument(
        "--validation-seasons",
        type=str,
        default="2023,2024,2025",
        help="Comma-separated season end years reserved for validation",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached HTML and refetch source pages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = [season for season in range(args.start_season, args.end_season + 1) if season != 2020]
    validation_seasons = sorted({int(value.strip()) for value in args.validation_seasons.split(",") if value.strip()})

    training = build_training_frame(seasons, force_refresh=args.force_refresh)
    tuned = choose_regularization(training, validation_seasons=set(validation_seasons))
    final_model = refit_full_model(training, regularization=tuned.regularization)
    final_model.validation_log_loss = tuned.validation_log_loss
    final_model.validation_accuracy = tuned.validation_accuracy

    save_model(final_model, seasons, validation_seasons, total_games=len(training))

    print(f"Wrote training data to {TRAINING_CSV_PATH}")
    print(f"Wrote learned model to {MODEL_PATH}")
    print(f"Validation log loss: {tuned.validation_log_loss:.4f}")
    print(f"Validation accuracy: {tuned.validation_accuracy:.3%}")
    print(f"Chosen regularization: {tuned.regularization}")
    print("Learned coefficients:")
    for name, value in zip(FEATURE_NAMES, final_model.coefficients):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
