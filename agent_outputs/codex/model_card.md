# Model Card: Leakage-Audited Flight Delay Classifier

## Model Overview

- Model name: Task 5 Leakage-Audited Flight Delay Classifier
- Model family: Histogram Gradient Boosting Classifier
- Prediction target: whether `ARRIVAL_DELAY > 15` minutes
- Primary artifact: the Task 5 audited model pipeline derived from `task5_model_audit.py`
- Intended prediction timing: strictly before departure

## Intended Use

This model is intended to support operational risk awareness before a flight departs. Suitable use cases include:

- identifying flights with elevated pre-departure delay risk
- prioritizing customer messaging for potentially disrupted itineraries
- ranking planned flights by expected delay exposure
- supporting schedule buffer decisions for business travel bookings

This model is not intended to:

- explain causality for individual delays
- replace airline operations control systems
- make safety-critical decisions
- produce legally binding customer compensation decisions

## Training and Evaluation Data

- Source dataset: US DOT aviation flight records in `data/flights.csv`
- Additional repository context: airport and airline reference files are used elsewhere in the pipeline, but the final audited model is trained from the raw flight file with engineered pre-departure features
- Sampling strategy: fixed random sampling with `RANDOM_SEED = 42`
- Label definition: `1` if `ARRIVAL_DELAY > 15`, else `0`
- Exclusions:
  - cancelled flights
  - diverted flights
  - rows with missing `ARRIVAL_DELAY`

## Features Used in the Final Audited Model

Only pre-departure or schedule-known features were retained. Examples include:

- calendar features: `MONTH`, `DAY`, `DAY_OF_WEEK`
- schedule timing: scheduled departure hour/minute, scheduled duration
- route structure: origin, destination, route, airline-route grouping
- airline identity
- flight number bucket
- distance and engineered distance bins
- weekend and seasonal flags

## Features Explicitly Removed for Leakage

The following features were removed during the audit because they are not available before departure:

- `DEPARTURE_DELAY`
- `DEPARTURE_DELAY_CLIPPED`
- `DEPARTURE_DELAY_15_PLUS`

These variables encode information observed after the aircraft should already be departing, so using them for a pre-departure arrival-delay model would create look-ahead leakage and falsely inflate reported performance.

## Performance

### Final audited Task 5 metrics

- Accuracy: `0.6432`
- Precision: `0.2683`
- Recall: `0.5836`
- F1-score: `0.3676`
- ROC-AUC: `0.6647`

### Leakage comparison

- Task 4 optimized ROC-AUC: `0.9332`
- Task 5 honest ROC-AUC: `0.6647`
- Exact AUC drop after leakage removal: `0.2685`

This gap is the central audit finding: earlier performance was materially overstated because post-departure information had entered the feature set.

## Strengths

- audited against look-ahead leakage
- reproducible through pinned dependencies and fixed random seed
- appropriate for pre-departure ranking of delay risk
- captures non-linear structure through boosted trees and engineered route/schedule features

## Limitations

- performance is modest after leakage removal, so the model should be treated as a risk-ranking tool rather than a high-certainty classifier
- the model was trained on historical data and may drift as airline schedules, airport congestion, weather exposure, or operating policies change
- route and airline groupings were bucketed for tractability, which may smooth over rare but meaningful local effects
- the model excludes cancelled and diverted flights from training because the arrival-delay target is structurally undefined there

## Ethical and Operational Considerations

- The model should not be used to penalize airlines, crews, airports, or passengers without broader operational context.
- Delay predictions may indirectly reflect structural congestion, regional weather patterns, or network effects rather than controllable airline behavior alone.
- The model may underperform on unusual operational events, shocks, or schedule changes not represented in the training sample.
- Users should communicate clearly that the model estimates probability of delay risk, not certainty of delay occurrence.

## Monitoring Recommendations

- track ROC-AUC, recall, and calibration over time
- compare live performance by airline, airport, route, and month
- re-run the leakage audit whenever new features are proposed
- retrain when major schedule or traffic-pattern changes occur

## Deployment Recommendation

This model is acceptable for pre-departure decision support, prioritization, and stakeholder reporting, provided the business uses the audited Task 5 metrics as the true performance reference. Any production deployment should use only the pre-departure feature set audited in Task 5 and should never reintroduce post-departure operational variables.
