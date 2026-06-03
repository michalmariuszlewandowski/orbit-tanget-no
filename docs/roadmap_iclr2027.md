# ICLR 2027 roadmap

Assumed external deadline: around September 20, 2026. Internal freeze: September 15, 2026.

## May 28--June 10: viability gate

Deliverables:

1. smoke test passes;
2. exact 1D advection data generated;
3. baseline and orbit FNO trained;
4. signs of translation action verified;
5. first figure: equivariance defect versus training epoch.

Gate:

```text
Continue if orbit consistency reduces equivariance defect and orbit OOD error.
Pivot if the defect does not move after transform and loss-scale debugging.
```

## June 11--June 24: Burgers gate

Deliverables:

1. Galilean identity unit test passes;
2. Burgers baseline, augmentation, orbit, and aug_orbit variants trained;
3. label-fraction pilot at 5%, 10%, 25%, 100%;
4. first timing table.

Gate:

```text
Continue if orbit improves over baseline or improves label efficiency.
Pivot to unlabeled adaptation if augmentation wins but orbit lowers equivariance defect.
```

## June 25--July 15: 2D gate

Deliverables:

1. 2D vorticity data generated;
2. FNO2d baseline and orbit variants trained;
3. translation and D4 defects reported separately if needed;
4. memory and latency reported.

Gate:

```text
Continue if the 2D result has the same direction as 1D.
Narrow the paper to 1D + adaptation if 2D is inconclusive but adaptation is strong.
```

## July 16--August 1: ablations

Deliverables:

1. lambda sweep;
2. data-fraction sweep;
3. epsilon/transform-magnitude sweep;
4. seed sweep for selected configurations;
5. compute table.

## August 2--August 15: stress tests

Deliverables:

1. approximate symmetry or boundary mask experiment;
2. unlabeled adaptation script and pilot result;
3. optional lightweight canonicalization baseline.

## August 16--August 25: theory freeze

Deliverables:

1. finalized first-order Lie-defect proposition;
2. orbit OOD decomposition;
3. appendix proof for connected-group propagation;
4. method pseudocode.

## August 26--September 5: result freeze

Deliverables:

1. final tables with three seeds for core results;
2. final plots;
3. result summary with pass/fail interpretation;
4. paper skeleton complete.

## September 6--September 15: paper freeze

Deliverables:

1. introduction and framing;
2. related work integrated with `paper/related_work.bib`;
3. method section polished;
4. theory checked;
5. final reproducibility checklist.
