#!/usr/bin/env python3
"""Fit the adjusted physician-preference analysis used in the workshop draft."""

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
from patsy import build_design_matrices


HERE = Path(__file__).resolve().parent
OUTPUT_CSV = HERE / "adjusted_gee_results.csv"
LOO_OUTPUT_CSV = HERE / "leave_one_physician_out_gee.csv"
CORE_FIELD_OUTPUT_CSV = HERE / "core_field_adjusted_gee.csv"
OUTPUT_MD = HERE / "ADJUSTED_ANALYSIS.md"

RATER_FILES = {
    "Physician 01": [HERE / "simo_breast_pdac_blind_scores_20260921.csv"],
    "Physician 02": [HERE / "kevin_breast_pdac_blind_scores_20261005.csv"],
    "Physician 03": [
        HERE / "bolun_breast_blind_scores_20260921.xlsx",
        HERE / "bolun_pdac_blind_scores_and_letter_selection_20261006.xlsx",
    ],
    "Physician 04": [HERE / "zhengrui_breast_pdac_blind_scores_20261005.csv"],
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

CORE_FIELD_LABELS = {
    "current_meds": "Active anticancer medications",
    "stage": "Stage",
    "response": "Treatment response",
    "distant_met": "Distant metastasis",
    "genetic_results": "Completed molecular or genetic results",
    "metastasis": "Regional or overall metastasis",
    "type_receptor": "Breast cancer type and receptors",
}


@dataclass(frozen=True)
class GeeResult:
    scope: str
    directional_judgments: int
    harness_preferences: int
    baseline_preferences: int
    clusters: int
    adjusted_probability: float
    probability_ci_low: float
    probability_ci_high: float
    difference_from_50: float
    difference_ci_low: float
    difference_ci_high: float
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class DescriptiveResult:
    total_judgments: int
    harness_preferences: int
    baseline_preferences: int
    ties: int
    directional_judgments: int
    crude_directional_probability: float
    crude_directional_ci_low: float
    crude_directional_ci_high: float
    net_preference_margin: float
    net_margin_ci_low: float
    net_margin_ci_high: float


@dataclass(frozen=True)
class LeaveOnePhysicianOutResult:
    excluded_physician: str
    remaining_physicians: int
    directional_judgments: int
    harness_preferences: int
    baseline_preferences: int
    clusters: int
    adjusted_probability: float
    probability_ci_low: float
    probability_ci_high: float
    difference_from_50: float
    difference_ci_low: float
    difference_ci_high: float
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class CoreFieldGeeResult:
    field: str
    field_label: str
    applicable_judgments: int
    directional_judgments: int
    harness_preferences: int
    baseline_preferences: int
    ties: int
    adjusted_probability: float
    probability_ci_low: float
    probability_ci_high: float
    converged: bool


def read_ratings() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for evaluator, paths in RATER_FILES.items():
        for path in paths:
            if path.suffix == ".xlsx":
                workbook = load_workbook(path, read_only=True, data_only=True)
                rows = list(workbook["blind_scores"].values)
                frame = pd.DataFrame(rows[1:], columns=rows[0])
                workbook.close()
            else:
                frame = pd.read_csv(path)
            frame["evaluator"] = evaluator
            frame["source_file"] = path.name
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
    duplicate = ratings.duplicated(["evaluator", "sample", "field"], keep=False)
    if duplicate.any():
        conflicts = ratings.loc[
            duplicate, ["evaluator", "sample", "field", "source_file"]
        ].to_dict("records")
        raise ValueError(f"Duplicate required judgments across source files: {conflicts}")
    ratings["cluster"] = ratings["sample"]
    ratings["preference_value"] = ratings["score"].map(
        {"A": 1, "B": -1, "TIE": 0}
    )
    return ratings


def _fit_gee(
    scope: str, directional: pd.DataFrame
) -> tuple[GeeResult, bool, int, tuple[str, ...]]:
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
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise AssertionError(
            f"Rank-deficient design for {scope}: "
            f"rank {np.linalg.matrix_rank(design)} of {design.shape[1]}."
        )
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
    probability_ci_low = float(expit(log_odds - 1.96 * log_odds_se))
    probability_ci_high = float(expit(log_odds + 1.96 * log_odds_se))

    result = GeeResult(
        scope=scope,
        directional_judgments=len(directional),
        harness_preferences=int((directional["score"] == "A").sum()),
        baseline_preferences=int((directional["score"] == "B").sum()),
        clusters=directional["cluster"].nunique(),
        adjusted_probability=adjusted_probability,
        probability_ci_low=probability_ci_low,
        probability_ci_high=probability_ci_high,
        difference_from_50=adjusted_probability - 0.5,
        difference_ci_low=probability_ci_low - 0.5,
        difference_ci_high=probability_ci_high - 0.5,
        odds_ratio=math.exp(log_odds),
        ci_low=math.exp(log_odds - 1.96 * log_odds_se),
        ci_high=math.exp(log_odds + 1.96 * log_odds_se),
        p_value=float(2 * norm.sf(abs(z_score))),
    )
    iterations = max(0, len(fitted.fit_history.get("params", [])) - 1)
    columns = tuple(model.exog_names)
    return result, bool(fitted.converged), iterations, columns


def fit_gee(scope: str, directional: pd.DataFrame) -> GeeResult:
    result, _, _, _ = _fit_gee(scope, directional)
    return result


def leave_one_physician_out_results(
    ratings: pd.DataFrame,
) -> list[LeaveOnePhysicianOutResult]:
    physicians = sorted(ratings["evaluator"].dropna().unique())
    if len(physicians) != 4:
        raise AssertionError(f"Expected four physicians, found {len(physicians)}.")

    full_directional = ratings[ratings["score"].isin(["A", "B"])].copy()
    results: list[LeaveOnePhysicianOutResult] = []
    for excluded in physicians:
        directional = full_directional[
            full_directional["evaluator"] != excluded
        ].copy()
        remaining = sorted(directional["evaluator"].unique())
        if len(remaining) != 3:
            raise AssertionError(
                f"Excluding {excluded} should leave three physicians, found {remaining}."
            )
        if set(directional["cancer"].unique()) != {"Breast cancer", "PDAC"}:
            raise AssertionError(f"Both cancer types must remain after excluding {excluded}.")

        gee, converged, iterations, columns = _fit_gee(
            f"Excluding {excluded}", directional
        )
        expected_columns = 1 + (len(remaining) - 1) + 1
        if len(columns) != expected_columns:
            raise AssertionError(
                f"Unexpected design for {excluded}: {columns}; expected "
                f"{expected_columns} columns."
            )
        if not converged:
            raise RuntimeError(f"GEE did not converge after excluding {excluded}.")
        if gee.directional_judgments != (
            gee.harness_preferences + gee.baseline_preferences
        ):
            raise AssertionError(f"Directional counts do not reconcile for {excluded}.")
        if gee.clusters != ratings["cluster"].nunique():
            raise AssertionError(f"A note cluster was lost after excluding {excluded}.")
        if not (
            0
            < gee.probability_ci_low
            < gee.adjusted_probability
            < gee.probability_ci_high
            < 1
        ):
            raise AssertionError(f"Invalid probability interval for {excluded}.")
        if not math.isclose(
            gee.odds_ratio,
            gee.adjusted_probability / (1 - gee.adjusted_probability),
            rel_tol=1e-12,
        ):
            raise AssertionError(f"Probability and odds disagree for {excluded}.")
        if not math.isclose(
            gee.difference_from_50,
            gee.adjusted_probability - 0.5,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Difference from 50% is inconsistent for {excluded}.")

        results.append(
            LeaveOnePhysicianOutResult(
                excluded_physician=excluded,
                remaining_physicians=len(remaining),
                directional_judgments=gee.directional_judgments,
                harness_preferences=gee.harness_preferences,
                baseline_preferences=gee.baseline_preferences,
                clusters=gee.clusters,
                adjusted_probability=gee.adjusted_probability,
                probability_ci_low=gee.probability_ci_low,
                probability_ci_high=gee.probability_ci_high,
                difference_from_50=gee.difference_from_50,
                difference_ci_low=gee.difference_ci_low,
                difference_ci_high=gee.difference_ci_high,
                odds_ratio=gee.odds_ratio,
                ci_low=gee.ci_low,
                ci_high=gee.ci_high,
                p_value=gee.p_value,
                converged=converged,
                iterations=iterations,
            )
        )

    if len(results) != 4 or {result.excluded_physician for result in results} != set(
        physicians
    ):
        raise AssertionError("Leave-one-physician-out analysis is incomplete.")
    return results


def descriptive_results(ratings: pd.DataFrame) -> DescriptiveResult:
    directional = ratings[ratings["score"].isin(["A", "B"])].copy()
    directional["harness_preferred"] = (directional["score"] == "A").astype(int)

    crude_model = smf.glm(
        "harness_preferred ~ 1",
        data=directional,
        family=sm.families.Binomial(),
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": directional["cluster"], "use_correction": False},
    )
    crude_log_odds = float(crude_model.params.iloc[0])
    crude_log_odds_se = float(crude_model.bse.iloc[0])

    net_model = smf.ols("preference_value ~ 1", data=ratings).fit(
        cov_type="cluster",
        cov_kwds={"groups": ratings["cluster"], "use_correction": False},
    )
    net_ci_low, net_ci_high = (float(value) for value in net_model.conf_int().iloc[0])

    return DescriptiveResult(
        total_judgments=len(ratings),
        harness_preferences=int((ratings["score"] == "A").sum()),
        baseline_preferences=int((ratings["score"] == "B").sum()),
        ties=int((ratings["score"] == "TIE").sum()),
        directional_judgments=len(directional),
        crude_directional_probability=float(directional["harness_preferred"].mean()),
        crude_directional_ci_low=float(
            expit(crude_log_odds - 1.96 * crude_log_odds_se)
        ),
        crude_directional_ci_high=float(
            expit(crude_log_odds + 1.96 * crude_log_odds_se)
        ),
        net_preference_margin=float(net_model.params.iloc[0]),
        net_margin_ci_low=net_ci_low,
        net_margin_ci_high=net_ci_high,
    )


def core_field_gee_results(ratings: pd.DataFrame) -> list[CoreFieldGeeResult]:
    """Fit one exploratory shared-effects GEE and standardize within each field."""

    field_order = list(CORE_FIELD_LABELS)
    core = ratings[ratings["field"].isin(field_order)].copy()
    directional = core[core["score"].isin(["A", "B"])].copy()
    directional["harness_preferred"] = (directional["score"] == "A").astype(int)
    directional["field"] = pd.Categorical(
        directional["field"], categories=field_order, ordered=True
    )
    if len(core) != 1040 or len(directional) != 495:
        raise AssertionError(
            f"Core-field counts changed: {len(core)} applicable and "
            f"{len(directional)} directional."
        )

    formula = (
        "harness_preferred ~ C(field, Sum) + C(evaluator, Sum) + C(cancer, Sum)"
    )
    model = smf.gee(
        formula,
        groups="cluster",
        data=directional,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )
    fitted = model.fit(maxiter=200)
    if not fitted.converged:
        raise RuntimeError("Joint exploratory core-field GEE did not converge.")
    if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
        raise AssertionError("Joint exploratory core-field GEE design is rank deficient.")

    coefficients = fitted.params.to_numpy()
    covariance = fitted.cov_params().to_numpy()
    design_info = model.data.orig_exog.design_info
    results: list[CoreFieldGeeResult] = []
    for field in field_order:
        observed = directional[directional["field"] == field].copy()
        design = np.asarray(
            build_design_matrices([design_info], observed, return_type="dataframe")[0]
        )
        probabilities = expit(design @ coefficients)
        probability = float(probabilities.mean())
        gradient = (
            (probabilities * (1 - probabilities))[:, None] * design
        ).mean(axis=0)
        logit_gradient = gradient / (probability * (1 - probability))
        logit_se = float(np.sqrt(logit_gradient @ covariance @ logit_gradient))
        logit = math.log(probability / (1 - probability))
        applicable = core[core["field"] == field]
        results.append(
            CoreFieldGeeResult(
                field=field,
                field_label=CORE_FIELD_LABELS[field],
                applicable_judgments=len(applicable),
                directional_judgments=len(observed),
                harness_preferences=int((applicable["score"] == "A").sum()),
                baseline_preferences=int((applicable["score"] == "B").sum()),
                ties=int((applicable["score"] == "TIE").sum()),
                adjusted_probability=probability,
                probability_ci_low=float(expit(logit - 1.96 * logit_se)),
                probability_ci_high=float(expit(logit + 1.96 * logit_se)),
                converged=bool(fitted.converged),
            )
        )

    expected_counts = {
        "current_meds": (132, 3, 25),
        "stage": (69, 7, 84),
        "response": (62, 12, 86),
        "distant_met": (23, 5, 132),
        "genetic_results": (34, 10, 116),
        "metastasis": (74, 26, 60),
        "type_receptor": (24, 14, 42),
    }
    expected_probabilities = {
        "current_meds": (0.977, 0.911, 0.994),
        "stage": (0.894, 0.776, 0.954),
        "response": (0.817, 0.701, 0.894),
        "distant_met": (0.785, 0.550, 0.916),
        "genetic_results": (0.775, 0.611, 0.883),
        "metastasis": (0.744, 0.659, 0.814),
        "type_receptor": (0.624, 0.386, 0.815),
    }
    for result in results:
        assert (
            result.harness_preferences,
            result.baseline_preferences,
            result.ties,
        ) == expected_counts[result.field]
        expected = expected_probabilities[result.field]
        actual = (
            result.adjusted_probability,
            result.probability_ci_low,
            result.probability_ci_high,
        )
        if any(abs(left - right) > 0.0007 for left, right in zip(actual, expected)):
            raise AssertionError(
                f"Unexpected standardized estimate for {result.field}: "
                f"{actual}, expected approximately {expected}."
            )
    return results


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

    for cancer, group in ratings.groupby("cancer"):
        pooled = group.groupby("sample")["preference_value"].sum()
        positive = int((pooled > 0).sum())
        negative = int((pooled < 0).sum())
        tied = int((pooled == 0).sum())
        summaries.append(
            {
                "evaluation": f"All physicians pooled, {cancer}",
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
            "evaluation": "All physicians pooled, both cancers",
            "positive": positive,
            "negative": negative,
            "tied": tied,
            "p_value": two_sided_sign_p(positive, negative),
        }
    )
    return summaries


def write_results(
    gee_results: list[GeeResult],
    descriptive: DescriptiveResult,
    sign_results: list[dict[str, object]],
    loo_results: list[LeaveOnePhysicianOutResult],
    core_field_results: list[CoreFieldGeeResult],
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

    with LOO_OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(LeaveOnePhysicianOutResult.__annotations__),
            lineterminator="\n",
        )
        writer.writeheader()
        for result in loo_results:
            writer.writerow(result.__dict__)

    with CORE_FIELD_OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CoreFieldGeeResult.__annotations__),
            lineterminator="\n",
        )
        writer.writeheader()
        for result in core_field_results:
            writer.writerow(result.__dict__)

    lines = [
        "# Adjusted physician-preference analysis",
        "",
        "Updated: 2026-10-13",
        "",
        "## Primary model",
        "",
        "The primary estimand is P(harness preference | directional judgment). The analysis excludes ties and uses a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator is included as a fixed effect, and the overall model also includes cancer type. Adjusted probabilities are standardized over the observed covariate distribution. Delta-method confidence limits are calculated on the logit scale and transformed back to probabilities. The absolute effect is the adjusted probability minus 50 percentage points; the odds ratio is the same standardized estimate on the odds scale.",
        "",
        "| Scope | Directional judgments | Harness | Baseline | Clusters | Adjusted probability (95% CI) | Difference from 50% (95% CI), pp | Odds ratio (95% CI) | p value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in gee_results:
        lines.append(
            f"| {result.scope} | {result.directional_judgments} | "
            f"{result.harness_preferences} | {result.baseline_preferences} | "
            f"{result.clusters} | {result.adjusted_probability:.2%} "
            f"({result.probability_ci_low:.2%}-{result.probability_ci_high:.2%}) | "
            f"{result.difference_from_50 * 100:+.2f} "
            f"({result.difference_ci_low * 100:+.2f} to "
            f"{result.difference_ci_high * 100:+.2f}) | "
            f"{result.odds_ratio:.3f} ({result.ci_low:.3f}-{result.ci_high:.3f}) | "
            f"{result.p_value:.3g} |"
        )

    lines.extend(
        [
            "",
            "## Leave-one-physician-out sensitivity analysis",
            "",
            "The overall model was refitted four times after excluding one physician at a time. Each fit rebuilt the fixed-effect coding for the three remaining physicians. All fits converged. Adjusted harness-preference probabilities ranged from 77.08% to 83.53%, and every 95% confidence interval remained above 50%.",
            "",
            "| Excluded physician | Directional judgments | Harness | Baseline | Notes | Adjusted probability (95% CI) | Difference from 50% (95% CI), pp | Odds ratio (95% CI) | p value | Converged |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in loo_results:
        lines.append(
            f"| {result.excluded_physician} | {result.directional_judgments} | "
            f"{result.harness_preferences} | {result.baseline_preferences} | "
            f"{result.clusters} | {result.adjusted_probability:.2%} "
            f"({result.probability_ci_low:.2%}-{result.probability_ci_high:.2%}) | "
            f"{result.difference_from_50 * 100:+.2f} "
            f"({result.difference_ci_low * 100:+.2f} to "
            f"{result.difference_ci_high * 100:+.2f}) | "
            f"{result.odds_ratio:.3f} ({result.ci_low:.3f}-{result.ci_high:.3f}) | "
            f"{result.p_value:.3g} | {'Yes' if result.converged else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Exploratory core-field model",
            "",
            "One joint exploratory GEE used the 495 directional core-field judgments. Field, physician, and cancer type were fixed effects; repeated judgments were clustered by note with an exchangeable working correlation. Field-specific probabilities were standardized over the physician and cancer distribution observed for that field. The breast type and receptor field applies only to breast-cancer notes. These estimates describe preference among directional judgments, not extraction accuracy. Ties were excluded. No pairwise field tests or multiplicity claims were made.",
            "",
            "| Core field | Applicable judgments | Harness | Baseline | Tie | Directional judgments | Adjusted probability (95% CI) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in core_field_results:
        lines.append(
            f"| {result.field_label} | {result.applicable_judgments} | "
            f"{result.harness_preferences} | {result.baseline_preferences} | "
            f"{result.ties} | {result.directional_judgments} | "
            f"{result.adjusted_probability:.1%} "
            f"({result.probability_ci_low:.1%}-{result.probability_ci_high:.1%}) |"
        )

    lines.extend(
        [
            "",
            "The odds ratio compares the adjusted probability of a harness preference with the probability of a baseline preference among non-tie judgments. It is not an odds ratio for clinical correctness.",
            "",
            "## Descriptive and tie-inclusive effects",
            "",
            f"The raw three-category distribution was {descriptive.harness_preferences}/{descriptive.total_judgments} harness preferences ({descriptive.harness_preferences / descriptive.total_judgments:.2%}), {descriptive.ties}/{descriptive.total_judgments} ties ({descriptive.ties / descriptive.total_judgments:.2%}), and {descriptive.baseline_preferences}/{descriptive.total_judgments} baseline preferences ({descriptive.baseline_preferences / descriptive.total_judgments:.2%}).",
            "",
            f"The crude harness share was {descriptive.crude_directional_probability:.2%} among {descriptive.directional_judgments} directional judgments (note-cluster robust 95% CI {descriptive.crude_directional_ci_low:.2%}-{descriptive.crude_directional_ci_high:.2%}). The tie-inclusive net preference margin, calculated as the mean of harness=1, tie=0, and baseline=-1 over all required judgments, was {descriptive.net_preference_margin * 100:+.2f} percentage points (note-cluster robust 95% CI {descriptive.net_margin_ci_low * 100:+.2f} to {descriptive.net_margin_ci_high * 100:+.2f}). Both intervals use sandwich standard errors clustered by note; they are not bootstrap intervals.",
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
            "The observed-note result is statistically strong, but four physicians remain too few for a precise estimate of variation across the broader physician population. Evaluator-level generalization should therefore remain a pilot claim.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ratings = read_ratings()
    directional = ratings[ratings["score"].isin(["A", "B"])].copy()
    descriptive = descriptive_results(ratings)
    loo_results = leave_one_physician_out_results(ratings)
    core_field_results = core_field_gee_results(ratings)
    gee_results = [
        fit_gee("Overall", directional),
        fit_gee("Breast cancer", directional[directional["cancer"] == "Breast cancer"]),
        fit_gee("PDAC", directional[directional["cancer"] == "PDAC"]),
    ]
    sign_results = note_sign_summaries(ratings)
    write_results(
        gee_results, descriptive, sign_results, loo_results, core_field_results
    )
    for result in gee_results:
        print(
            f"{result.scope}: adjusted probability {result.adjusted_probability:.2%} "
            f"({result.probability_ci_low:.2%}-{result.probability_ci_high:.2%}), "
            f"OR {result.odds_ratio:.3f} "
            f"({result.ci_low:.3f}-{result.ci_high:.3f}), "
            f"p={result.p_value:.3g}"
        )
    print(
        f"Crude directional: {descriptive.crude_directional_probability:.2%} "
        f"({descriptive.crude_directional_ci_low:.2%}-"
        f"{descriptive.crude_directional_ci_high:.2%}); "
        f"tie-inclusive net margin {descriptive.net_preference_margin * 100:+.2f} pp "
        f"({descriptive.net_margin_ci_low * 100:+.2f} to "
        f"{descriptive.net_margin_ci_high * 100:+.2f})"
    )
    for result in loo_results:
        print(
            f"Excluding {result.excluded_physician}: "
            f"adjusted probability {result.adjusted_probability:.2%} "
            f"({result.probability_ci_low:.2%}-{result.probability_ci_high:.2%}), "
            f"OR {result.odds_ratio:.3f} "
            f"({result.ci_low:.3f}-{result.ci_high:.3f}), "
            f"p={result.p_value:.3g}, converged={result.converged}"
        )
    for result in core_field_results:
        print(
            f"{result.field_label}: {result.adjusted_probability:.1%} "
            f"({result.probability_ci_low:.1%}-{result.probability_ci_high:.1%})"
        )
    print(
        f"Wrote {OUTPUT_CSV.name}, {LOO_OUTPUT_CSV.name}, "
        f"{CORE_FIELD_OUTPUT_CSV.name}, and {OUTPUT_MD.name}"
    )


if __name__ == "__main__":
    main()
