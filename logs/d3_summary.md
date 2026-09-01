# D3 leakage audit and frozen feature set

## Prediction-time definition

The task is conditional severity prediction after defining that a collision occurs, using only collision-table conditions that existed at or before the event. Post-impact consequences, emergency response, identifiers, exact geography, alternative severity outcomes and DfT severity-adjustment outputs are excluded.

## Frozen feature set

- Raw included fields: **15** (14 categorical, 1 numeric).
- Derived fields: **2** (`month`, `hour`).
- Final model feature count before encoding: **17**.
- Target order: Slight < Serious < Fatal.
- Source scope: collision table only; vehicle and casualty tables are not joined.

Direct fields: `day_of_week`, `first_road_class`, `road_type`, `junction_detail`, `junction_control`, `second_road_class`, `pedestrian_crossing`, `light_conditions`, `weather_conditions`, `road_surface_conditions`, `special_conditions_at_site`, `carriageway_hazards`, `urban_or_rural_area`, `trunk_road_flag`, `speed_limit`.

## Decision counts

- DERIVE_THEN_DROP: 2
- EXCLUDE_EXACT_GEOGRAPHY: 9
- EXCLUDE_HIGH_CARDINALITY: 2
- EXCLUDE_HISTORIC_DUPLICATE: 4
- EXCLUDE_IDENTIFIER: 2
- EXCLUDE_OUTCOME: 1
- EXCLUDE_POST_COLLISION: 2
- EXCLUDE_TARGET_DERIVED: 3
- GROUP_ONLY: 1
- INCLUDE: 15
- REPORTING_AUDIT_ONLY: 1
- SPLIT_ONLY: 1
- TARGET: 1

## High unknown/not-applicable rates among included fields

- `junction_control`: 44.1%
- `special_conditions_at_site`: 10.6%

These values are not silently deleted. Categorical unknown/not-applicable codes remain explicit; speed-limit -1 is converted to missing inside the modelling pipeline.

## Non-negotiable leakage exclusions

`number_of_casualties`, `enhanced_severity_collision`, `collision_adjusted_severity_serious`, and `collision_adjusted_severity_slight` encode outcomes or severity-adjustment results. `did_police_officer_attend_scene_of_accident` is a post-collision response. None may enter training, preprocessing, feature selection or SHAP analysis.

`collision_injury_based` is reserved for reporting-system sensitivity analysis and must not enter the feature matrix.
