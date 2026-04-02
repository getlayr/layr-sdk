# Contributing to Layr SDK

## Setup

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -e ".[dev]"`

## Workflow

- Add or update tests for every behavior change.
- Keep public methods fully typed and documented.
- Run:
  - `pytest`
  - `python -m mypy layr`

## Pull Requests

- Keep PRs focused and small.
- Include rationale and test evidence.
- Update `CHANGELOG.md` for user-visible changes.
