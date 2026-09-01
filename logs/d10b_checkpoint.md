# D10b tree-count sensitivity checkpoint

## Status

**PASS - the pre-frozen temporal boundary diagnostic completed.**

## Scope

- One fixed C03 model was trained on 2018-2022 and evaluated on 2023 only.
- The fit used 4 CPU threads, no early stopping and checkpoints at 1200, 1500 and 2000 trees.
- No random-reference test row and no 2024 row was predicted or evaluated.
- D10b is diagnostic only. D11 remains locked to the D10 C03 model with 1200 trees.

## Reproduction control

- The 1200-tree checkpoint reproduced the frozen D10 temporal validation IDs, labels, probabilities and reported metrics within the prespecified numerical tolerance.

## Observed validation results

- 1200 trees: log loss **0.87480894**, Macro-F1 **0.35727**, QWK **0.11629**, Fatal recall **0.38765**.
- 1500 trees: log loss **0.86032835**, Macro-F1 **0.35961**, QWK **0.11577**, Fatal recall **0.35151**.
- 2000 trees: log loss **0.84324476**, Macro-F1 **0.36278**, QWK **0.11533**, Fatal recall **0.31932**.
- Log-loss change, 1200 to 1500: **-0.01448059**.
- Log-loss change, 1500 to 2000: **-0.01708359**.
- No post-hoc numerical threshold was used to label the curve as plateaued or not plateaued.
- Full 2000-tree fit time: **65.58 s**.
