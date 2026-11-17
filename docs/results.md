# Results log

Rule: an entry exists only if it was measured, with ≥ 5 seeds where learning
is involved, and it records **date, git commit, config file, exact command, n**.
Single-seed observations from the build are in `build_notes.md`, not here.

| date | stage | claim | numbers (mean ± 95% CI, n) | commit | command / config | figure |
|------|-------|-------|----------------------------|--------|------------------|--------|
| | 1 | C tabular learner reproduces the Python Q16.16 reference bit for bit on the corridor | 15/15 seed×ε cases identical | | `make test` | — |
