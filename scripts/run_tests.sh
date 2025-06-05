#!/bin/bash
uv run --frozen pytest \
  --junitxml=pytest.xml \
  --cov-report=term-missing:skip-covered \
  --cov=. \
  tests/ | tee pytest-coverage.txt
