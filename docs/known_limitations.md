# Known limitations

This repository contains paper-grade infrastructure and cached paper-facing evidence. Additional training runs may refine the empirical claim, but the included tables and figures define the submitted artifact snapshot.

The two-dimensional Navier--Stokes solver is a compact pseudo-spectral RK4 implementation intended for pilots and symmetry debugging. Final paper numbers should include solver-validation reports, and possibly a higher-resolution or smaller-step reference check.

The baseline suite includes supervised FNO, supervised Lie augmentation, orbit consistency, augmentation plus orbit consistency, unlabeled adaptation, and a lightweight analytic canonicalization baseline. External G-FNO or large third-party canonicalization implementations are not vendored. Add those as external baselines if the main result depends on comparisons against architecture-level equivariance.

The current transforms cover periodic translations, Burgers Galilean boosts, and D4 vorticity symmetries. Boundary-broken or geometry-broken symmetries should use explicit validity masks and should be reported separately from exact-symmetry results.

The quality gate checks code integrity, config validity, solver identities, and smoke execution. It does not certify that hyperparameters are tuned or that the method improves over baselines.
