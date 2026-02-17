# Known Issues

## Ignored (won't fix for now)

1. `downsample_behavior_data()` assumes a `TIME (S)` column is first in the dataset. (Input data is fixed in current workflow.)
2. `process_data()` slices by index labels, which can misbehave if the index is unsorted or irregular. (Current datasets are sorted, so this is acceptable.)
3. `TRIAL_TYPE["RT"]` is sized from `LIGHT_ONSET["LC"]`, which may mismatch RT timing. (RT uses LC timing in current experiments.)
