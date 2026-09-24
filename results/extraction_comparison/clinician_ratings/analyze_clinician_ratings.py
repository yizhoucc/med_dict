#!/usr/bin/env python3
"""Fit the adjusted clinician-preference analysis used in the workshop draft."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from openpyxl import load_workbook
from scipy.special import expit
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable


HERE = Path(__file__).resolve().parent
OUTPUT_CSV = HERE / "adjusted_gee_results.csv"
OUTPUT_MD = HERE / "ADJUSTED_ANALYSIS.md"

RATER_FILES = {
    "Oncologist 01": HERE / "simo_breast_pdac_blind_scores_20260921.csv",
    "Oncologist 02": HERE / "kevin_breast_pdac_blind_scores_20260911.csv",
    "Oncologist 03": HERE / "bolun_breast_blind_scores_20260921.xlsx",
}

BREAST_FIELDS = [
    "current_meds",
    "stage",
    "distant_met",
    "metastasis",
    "response",
    "type_receptor",
    "genetic_results",
    "genetic_plan",
    "supportive_meds",
    "procedure_plan",
    "imaging_plan",
    "lab_plan",
    "medication_plan",
    "recent_changes",
]
PDAC_FIELDS = [field for field in BREAST_FIELDS if field != "type_receptor"]


@dataclass(frozen=True)
class GeeResult:
    scope: str
    directional_judgments: int
    harness_preferences: int
    baseline_preferences: int
    clusters: int
    adjusted_probability: float
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float


def read_ratings() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for evaluator, path in RATER_FILES.items():
        if path.suffix == ".xlsx":
            workbook = load_workbook(path, read_only=True, data_only=True)
            rows = list(workbook["blind_scores"].values)
            frame = pd.DataFrame(rows[1:], columns=rows[0])
        else:
            frame = pd.read_csv(path)
        frame["evaluator"] = evaluator
        frames.append(frame)

    ratings = pd.concat(frames, ignore_index=True)
    ratings["cancer"] = ratings["sample"].str[0].map(
        {"b": "Breast cancer", "p": "PDAC"}
    )
    required = (
        (ratings["cancer"] == "Breast cancer")
        & ratings["field"].isin(BREAST_FIELDS)
    ) | ((ratings["cancer"] == "PDAC") & ratings["field"].isin(PDAC_FIELDS))
    ratings = ratings[required].copy()
    ratings["cluster"] = ratings["sample"]
    ratings["preference_value"] = ratings["score"].map(
        {"A": 1, "B": -1, "TIE": 0}
    )
    return ratings


def fit_gee(scope: str, directional: pd.DataFrame) -> GeeResult:
    directional = directional.copy()
    directional["harness_preferred"] = (directional["score"] == "A").astype(int)
    formula = "harness_preferred ~ C(evaluator, Sum)"
    if directional["cancer"].nunique() > 1:
        formula += " + C(cancer, Sum)"

    model = smf.gee(
        formula,
        groups="cluster",
        data=directional,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )
    fitted = model.fit(maxiter=200)

    design = model.exog
    coefficients = fitted.params.to_numpy()
    covariance = fitted.cov_params().to_numpy()
    probabilities = expit(design @ coefficients)
    adjusted_probability = float(probabilities.mean())

    probability_gradient = (
        (probabilities * (1 - probabilities))[:, None] * design
    ).mean(axis=0)
    log_odds_gradient = probability_gradient / (
        adjusted_probability * (1 - adjusted_probability)
    )
    log_odds_se = float(
        np.sqrt(log_odds_gradient @ covariance @ log_odds_gradient)
    )
    log_odds = math.log(adjusted_probability / (1 - adjusted_probability))
    z_score = log_odds / log_odds_se

    return GeeResult(
        scope=scope,
        directional_judgments=len(directional),
        harness_preferences=int((directional["score"] == "A").sum()),
        baseline_preferences=int((directional["score"] == "B").sum()),
        clusters=directional["cluster"].nunique(),
        adjusted_probability=adjusted_probability,
        odds_ratio=math.exp(log_odds),
        ci_low=math.exp(log_odds - 1.96 * log_odds_se),
        ci_high=math.exp(log_odds + 1.96 * log_odds_se),
        p_value=float(2 * norm.sf(abs(z_score))),
    )


def two_sided_sign_p(positive: int, negative: int) -> float:
    total = positive + negative
    tail = min(positive, negative)
    probability = 2 * sum(math.comb(total, i) for i in range(tail + 1)) / 2**total
    return min(1.0, probability)


def note_sign_summaries(ratings: pd.DataFrame) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for (evaluator, cancer), group in ratings.groupby(["evaluator", "cancer"]):
        margins = group.groupby("sample")["preference_value"].sum()
        positive = int((margins > 0).sum())
        negative = int((margins < 0).sum())
        tied = int((margins == 0).sum())
        summaries.append(
            {
                "evaluation": f"{evaluator}, {cancer}",
                "positive": positive,
                "negative": negative,
                "tied": tied,
                "p_value": two_sided_sign_p(positive, negative),
            }
        )

    pooled = ratings.groupby(["cancer", "sample"])["preference_value"].sum()
    positive = int((pooled > 0).sum())
    negative = int((pooled < 0).sum())
    tied = int((pooled == 0).sum())
    summaries.append(
        {
            "evaluation": "All clinicians pooled within note",
            "positive": positive,
            "negative": negative,
            "tied": tied,
            "p_value": two_sided_sign_p(positive, negative),
        }
    )
    return summaries


def write_results(
    gee_results: list[GeeResult], sign_results: list[dict[str, object]]
) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(GeeResult.__annotations__),
            lineterminator="\n",
        )
        writer.writeheader()
        for result in gee_results:
            writer.writerow(result.__dict__)

    lines = [
        "# Adjusted clinician-preference analysis",
        "",
        "Updated: 2026-09-23",
        "",
        "## Primary model",
        "",
        "The primary analysis excludes ties and models whether a directional rating favors the inference harness. It uses a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator is included as a fixed effect, and the overall model also includes cancer type. This approach accounts for repeated ratings of fields and evaluators within a note without treating all 520 directional ratings as independent.",
        "",
        "| Scope | Directional judgments | Harness | Baseline | Clusters | Adjusted harness probability | Odds ratio | 95% CI | p value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in gee_results:
        lines.append(
            f"| {result.scope} | {result.directional_judgments} | "
            f"{result.harness_preferences} | {result.baseline_preferences} | "
            f"{result.clusters} | {result.adjusted_probability:.1%} | "
            f"{result.odds_ratio:.2f} | {result.ci_low:.2f}-{result.ci_high:.2f} | "
            f"{result.p_value:.3g} |"
        )

    lines.extend(
        [
            "",
            "The odds ratio compares the adjusted probability of a harness preference with the probability of a baseline preference among non-tie judgments. It is not an odds ratio for clinical correctness.",
            "",
            "## Note-level sensitivity analysis",
            "",
            "For each evaluator-cancer combination, field ratings were reduced to one harness-minus-baseline margin per note. Tied note margins were excluded from the exact two-sided sign test.",
            "",
            "| Evaluation | Positive notes | Negative notes | Tied notes | Exact p value |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in sign_results:
        lines.append(
            f"| {result['evaluation']} | {result['positive']} | "
            f"{result['negative']} | {result['tied']} | {result['p_value']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The observed-note result is statistically strong, but three oncologists remain too few for a precise estimate of variation across the broader oncologist population. Evaluator-level generalization should therefore remain a pilot claim.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ratings = read_ratings()
    directional = ratings[ratings["score"].isin(["A", "B"])].copy()
    gee_results = [
        fit_gee("Overall", directional),
        fit_gee("Breast cancer", directional[directional["cancer"] == "Breast cancer"]),
        fit_gee("PDAC", directional[directional["cancer"] == "PDAC"]),
    ]
    sign_results = note_sign_summaries(ratings)
    write_results(gee_results, sign_results)
    for result in gee_results:
        print(
            f"{result.scope}: OR {result.odds_ratio:.2f}, "
            f"95% CI {result.ci_low:.2f}-{result.ci_high:.2f}, "
            f"p={result.p_value:.3g}"
        )
    print(f"Wrote {OUTPUT_CSV.name} and {OUTPUT_MD.name}")


if __name__ == "__main__":
    main()
