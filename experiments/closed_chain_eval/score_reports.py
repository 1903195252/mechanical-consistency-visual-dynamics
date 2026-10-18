"""Apply the current composite to planar reports, including historical measurements.

Measurement dictionaries and prediction files are preserved. All public scoring
and presentation entry points call this adapter instead of averaging pass rates.
"""
from .continuous import SCORING_VERSION, aggregate_continuous
from .rescore_continuous import PLANAR_FIELDS, UNITS, make_card


def planar_report(report, limits, calibration_id):
    rows, cards = [], []
    for original in report["rows"]:
        row = dict(original)
        card = make_card({k: row.get("metrics", {}).get(v) for k, v in PLANAR_FIELDS.items()},
                         {k: limits[v] for k, v in PLANAR_FIELDS.items()}, UNITS, calibration_id,
                         (row["category"], row["condition"], row["case_id"]))
        row["continuous_components_100"] = {
            field: card["continuous"]["metrics"][k]["score_100"] for k, field in PLANAR_FIELDS.items()}
        row["continuous_score_100"] = card["continuous"]["score_100"]
        cards.append(card)
        rows.append(row)
    result = aggregate_continuous(cards, [c["case_key"] for c in cards], calibration_id=calibration_id)
    return dict(report, rows=rows, scoring_version=SCORING_VERSION,
                composite_score_100=result["continuous"]["score_100"],
                continuous_components_100=result["continuous"]["components_100"])
