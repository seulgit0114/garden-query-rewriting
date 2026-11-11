# Rule-Statistic Calibration Report

- min_support: 5
- eligible rule-feature pairs: 8

| Rule | Feature | N | Regression Rate | Improvement Rate | Common Levels |
|---|---:|---:|---:|---:|---|
| PROJECT_JOIN_TRANSPOSE | join_width | 6 | 0.500 | 0.000 | high:14=2, medium:4=1, high:9=1 |
| PROJECT_JOIN_TRANSPOSE | input_cardinality | 6 | 0.500 | 0.000 | high=4, medium=2 |
| JOIN_PROJECT_LEFT_TRANSPOSE_INCLUDE_OUTER | join_width | 6 | 0.500 | 0.000 | high:14=2, medium:4=1, high:9=1 |
| PROJECT_FILTER_TRANSPOSE | join_width | 5 | 0.400 | 0.000 | high:14=4, medium:4=1 |
| PROJECT_FILTER_TRANSPOSE | input_cardinality | 5 | 0.400 | 0.000 | high=4, medium=1 |
| AGGREGATE_PROJECT_MERGE | input_cardinality | 158 | 0.380 | 0.025 | high=106, medium=52 |
| FILTER_INTO_JOIN | join_width | 193 | 0.311 | 0.021 | high:8=48, medium:5=40, medium:7=28 |
| FILTER_REDUCE_EXPRESSIONS | join_width | 22 | 0.045 | 0.000 | high:8=9, medium:6=5, medium:7=3 |
