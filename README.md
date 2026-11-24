# ML CI/CD Pipeline (P10)

Every push to `main` runs a GitHub Actions pipeline:

1. **Lint**: flake8
2. **Unit tests**: pytest
3. **Train + quality gate**: Logistic Regression on the Breast Cancer dataset; the job fails if accuracy < `ACCURACY_THRESHOLD` (default 0.90)
4. **Docker**: build the image with the trained model and push it to GitHub Container Registry (`ghcr.io/<owner>/<repo>`)

The pipeline stops at the first failed stage, so a model that fails the gate is never packaged.

Deliberate failure: Actions → ML CI/CD Pipeline → Run workflow → set threshold to `0.99`.
