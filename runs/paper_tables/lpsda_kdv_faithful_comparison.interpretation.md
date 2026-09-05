# LPSDA-aligned KdV comparison: interpretation

## Evidence being compared

- Primary published source: https://proceedings.mlr.press/v162/brandstetter22a/brandstetter22a.pdf
- The two **published-reference** rows are the 64-sample KdV 40s FNO(AR) entries copied from Brandstetter et al., Table 3. They are not local reruns. Their reported uncertainty is a bootstrap $\pm2$ SD interval, not an across-training-seed standard deviation.
- The two **ours (independent protocol)** rows are five matched local seeds on one cached dataset. Both use the same 10,856,084-parameter FNO. LOCO changes training only.
- The local implementation is aligned to the released 20-to-20-step autoregressive FNO protocol, not a bitwise reproduction. Local uncertainty is the sample standard deviation across five training seeds.
- Local LOCO constrains only the Galilean generator $g_3$ with $|\epsilon|\leq0.2$. Published full LPSDA augments with $g_1g_2g_3g_4$. Cross-block comparisons are therefore contextual rather than controlled method-only comparisons.

## Supported interpretation

- Against the matched local FNO calibration, LOCO changes ID NMSE from 0.1205 to 0.1199. The paired mean change (LOCO minus FNO) is -0.000548, or -0.45%. Its two-sided 95% paired t-interval is [-0.001690, 0.000595] (p=0.254). With five seeds, this is consistent with preserved ID accuracy, not evidence of an ID improvement.
- The local LOCO absolute Galilean metrics are 0.5341 orbit-OOD NMSE and 0.5724 equivariance defect. No matched five-seed local FNO or published LPSDA values exist for these metrics, so this benchmark cannot establish a symmetry-robustness gain over either comparator.
- Published full LPSDA reduces ID NMSE by 51.4% within the source paper (0.1248 to 0.0606). Local LOCO ID NMSE is 1.98x the published full-LPSDA value. Current Galilean-only LOCO therefore does not reproduce the published full-symmetry LPSDA label-efficiency gain.
- The local FNO calibration is 3.5% below the correct published FNO(AR) baseline and lies inside its reported interval. This supports broad protocol alignment, while the independent implementation and different uncertainty estimands still preclude a controlled cross-block method attribution.
- The published 40s augmentation ladder is informative but not a Galilean-only control: $g_1$ gives 0.0674, $g_1g_2$ gives 0.0673, $g_1g_2g_3$ gives 0.0550, and $g_1g_2g_3g_4$ gives 0.0606. Much of the gain appears before adding $g_3$, so the table cannot isolate LOCO versus supervised augmentation for the same generator.
- FNO and LOCO have identical parameter counts and inference graphs. Their measured latency means (11.90 and 10.15 ms/sample) must not be interpreted as a LOCO speedup; the defensible conclusion is that LOCO adds no inference-time module or theoretical inference cost.
- Training wall time increases from 4.29 to 7.44 hours per seed on average (1.74x), so unchanged inference cost must not be confused with unchanged training cost.

## Bottom line

The external KdV run shows that five-seed, Galilean-only LOCO under an independent LPSDA-aligned FNO(AR) protocol preserves local ID accuracy with an unchanged inference architecture. It does **not** reproduce full LPSDA's published label-efficiency gain, does **not** provide a controlled same-generator LOCO-versus-LPSDA comparison, and does **not** provide a matched symmetry-metric comparison on this benchmark.
