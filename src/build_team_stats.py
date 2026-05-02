from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup


BASE_URL = "https://www.ncaa.com"
MAIN_STATS_URL = f"{BASE_URL}/stats/basketball-men/d1/current/team/145"
OUTPUT_PATH = Path(__file__).with_name("current_season_team_stats.csv")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

DESIRED_STATS = {
    "Scoring Offense": "ppg",
    "Scoring Defense": "opp_ppg",
    "Field Goal Percentage": "fg_pct",
    "Three Point Percentage": "three_pt_pct",
    "Free Throw Percentage": "ft_pct",
    "Rebounds Per Game": "reb_pg",
    "Assists Per Game": "ast_pg",
    "Turnovers Per Game": "to_pg",
    "Assist/Turnover Ratio": "ast_to_ratio",
    "Winning Percentage": "win_pct",
}

COLUMN_CANDIDATES = {
    "ppg": ["PPG"],
    "opp_ppg": ["OPP PPG", "PPG"],
    "fg_pct": ["FG%"],
    "three_pt_pct": ["3FG%", "PCT"],
    "ft_pct": ["FT%"],
    "reb_pg": ["RPG"],
    "ast_pg": ["APG"],
    "to_pg": ["TOPG"],
    "ast_to_ratio": ["Ratio"],
    "win_pct": ["Pct"],
}


def fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_stat_urls(html: str) -> dict[str, str]:
    # NCAA renders stat choices as option tags with relative URLs.
    option_pattern = re.compile(r'<option[^>]+value="(?P<value>/stats/basketball-men/d1/current/team/\d+)"[^>]*>(?P<label>[^<]+)</option>')
    stat_urls: dict[str, str] = {}
    for match in option_pattern.finditer(html):
        label = re.sub(r"\s+", " ", match.group("label")).strip()
        stat_urls[label] = BASE_URL + match.group("value")
    return stat_urls


def normalize_team_name(name: str) -> str:
    cleaned = re.sub(r"^Image", "", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    if "Team" not in df.columns:
        raise ValueError("Expected 'Team' column in NCAA stats table")
    df["Team"] = df["Team"].astype(str).map(normalize_team_name)
    df = df[df["Team"].ne("Team")]
    df = df[df["Team"].ne("")]
    return df


def parse_first_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find a table in the NCAA stats page")

    header_cells = table.select("thead tr th")
    headers = [cell.get_text(" ", strip=True) for cell in header_cells]
    if not headers:
        first_row = table.find("tr")
        if first_row is None:
            raise ValueError("Could not find header row in NCAA stats table")
        headers = [cell.get_text(" ", strip=True) for cell in first_row.find_all(["th", "td"])]

    body_rows = []
    for row in table.select("tbody tr"):
        cells = row.find_all(["th", "td"])
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if values:
            body_rows.append(values)

    normalized_rows = []
    for values in body_rows:
        if len(values) < len(headers):
            values = values + [""] * (len(headers) - len(values))
        normalized_rows.append(values[: len(headers)])

    return pd.DataFrame(normalized_rows, columns=headers)


def find_metric_column(df: pd.DataFrame, target_key: str) -> str:
    for candidate in COLUMN_CANDIDATES[target_key]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find metric column for {target_key}: {df.columns.tolist()}")


def fetch_stat_table(stat_url: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    seen_teams: set[str] = set()

    for page_number in range(1, 20):
        page_url = stat_url if page_number == 1 else f"{stat_url}/p{page_number}"
        try:
            html = fetch_html(page_url)
        except HTTPError as exc:
            if exc.code == 404 and page_number > 1:
                break
            raise
        except URLError:
            if page_number > 1:
                break
            raise

        try:
            table = clean_table(parse_first_table(html))
        except ValueError:
            if page_number > 1:
                break
            raise
        new_rows = table[~table["Team"].isin(seen_teams)]
        if new_rows.empty:
            break

        seen_teams.update(new_rows["Team"].tolist())
        rows.append(new_rows)

    if not rows:
        raise ValueError(f"No rows fetched from {stat_url}")

    return pd.concat(rows, ignore_index=True)


def merge_stat_tables(stat_urls: dict[str, str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for stat_name, output_name in DESIRED_STATS.items():
        if stat_name not in stat_urls:
            raise ValueError(f"Stat '{stat_name}' not found in NCAA stats dropdown")

        table = fetch_stat_table(stat_urls[stat_name])
        metric_column = find_metric_column(table, output_name)
        subset_columns = ["Team", metric_column]
        if "GM" in table.columns:
            subset_columns.insert(1, "GM")
        subset = table[subset_columns].copy()
        rename_map = {"Team": "team", metric_column: output_name}
        if "GM" in subset.columns:
            rename_map["GM"] = "games"
        subset = subset.rename(columns=rename_map)

        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on=["team"], how="outer")
            if "games_x" in merged.columns and "games_y" in merged.columns:
                merged["games"] = merged["games_x"].fillna(merged["games_y"])
                merged = merged.drop(columns=["games_x", "games_y"])
            elif "games_x" in merged.columns:
                merged = merged.rename(columns={"games_x": "games"})
            elif "games_y" in merged.columns:
                merged = merged.rename(columns={"games_y": "games"})

    if merged is None:
        raise ValueError("No stat tables were merged")

    numeric_columns = [column for column in merged.columns if column not in {"team"}]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    return merged.sort_values("team").reset_index(drop=True)


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    try:
        main_html = fetch_html(MAIN_STATS_URL)
        stat_urls = extract_stat_urls(main_html)
        merged = merge_stat_tables(stat_urls)
        write_csv(merged, OUTPUT_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build current season team stats CSV: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(merged)} rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
