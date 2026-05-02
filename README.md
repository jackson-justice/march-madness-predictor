# March Madness Predictor

## Overview

This project builds a data-driven model to predict NCAA March Madness tournament outcomes using historical data and Monte Carlo simulation.

The goal is to estimate matchup probabilities and simulate entire tournament brackets to generate title odds and advancement probabilities for each team.

---

## Approach

### 1. Data Collection
- Scraped historical NCAA team statistics and tournament results
- Built a dataset of tournament matchups with team-level features

### 2. Feature Engineering
For each matchup, computed differences between teams:
- Seed gap
- Scoring margin
- Win percentage
- Shooting efficiency (FG%, 3PT%, FT%)
- Rebounding and assists
- Turnovers and assist-to-turnover ratio
- Advanced metrics (SRS, SOS, offensive rating, etc.)

---

### 3. Model

Trained a logistic regression model to estimate:

> P(team A beats team B)

- Features are standardized using historical means and standard deviations
- Model is trained on past NCAA tournaments
- Regularization is tuned using validation seasons

---

### 4. Simulation

- Simulates the entire tournament bracket thousands of times
- Each game outcome is sampled using predicted win probabilities
- Aggregates results to estimate:
  - Championship probability
  - Final Four probability
  - Round advancement rates

---

## Example Output

- Title odds by team
- Final Four probabilities
- Simulated bracket results

---

## Files

- `src/train_model.py` – trains logistic regression on historical data  
- `src/march_madness_model.py` – runs tournament simulations  
- `src/build_team_stats.py` – scrapes current season team stats  
- `models/historical_model_weights.json` – learned model parameters  
- `data/` – input datasets  

---

## Tools Used

- Python
- pandas
- numpy
- BeautifulSoup (web scraping)

---

## Future Improvements

- Incorporate player-level data
- Add injury adjustments
- Improve model with ensemble methods
- Calibrate probabilities using cross-validation

---

## Notes

This project focuses on combining statistical modeling with simulation to replicate real-world bracket prediction and uncertainty.
