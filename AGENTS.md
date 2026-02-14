# AGENTS.md

Guide for agentic coding tools working in `C:\Users\PCC\Desktop\steps_actu`.

## Scope and precedence

- Applies to the whole repository.
- Follow user instructions first, then this file.
- Keep edits focused and avoid unrelated refactors.

## Project reality (important)

- Python script-based repo; no package build system.
- Main runtime: `live_trading_bot.py`.
- Core ML/trading modules are in `src/`.
- Root `test_*.py` files are script-style tests with `main()`, not pytest unit tests.
- Typical environment is `btc_trading_env` virtualenv.

## Cursor/Copilot rule files

- Checked for `.cursorrules`, `.cursor/rules/`, and `.github/copilot-instructions.md`.
- None were found during analysis.
- If added later, treat them as required and update this document.

## Environment setup

Run from repo root:

```bash
python -m venv btc_trading_env
btc_trading_env\Scripts\activate
pip install -r requirements.txt
```

## Build / lint / test commands

There is no formal build command (`pyproject.toml`, `setup.py`, `Makefile`, CI pipeline not present in root).

### Build-equivalent checks

```bash
python -m compileall src
python -m compileall .
```

### Lint/format checks

No committed lint config. Use safe syntax checks:

```bash
python -m py_compile src\*.py
python -m py_compile *.py
```

If you add Ruff/Black/Mypy locally, do not do a repo-wide reformat unless explicitly requested.

### Tests

Run one test script (preferred single-test workflow):

```bash
python test_realtime_720d.py
```

Run any specific test script:

```bash
python test_short_only.py
```

Run all root test scripts (PowerShell):

```powershell
Get-ChildItem test_*.py | ForEach-Object { python $_.FullName }
```

Single-test note:

- Current repo does not expose pytest-style `::test_name` targets.
- If pytest tests are added later, use `python -m pytest path/to/file.py::test_name`.

## Code style conventions

Follow local file style and nearby patterns before external opinions.

### Imports

- Order: stdlib, third-party, local imports.
- Prefer explicit imports; avoid wildcard imports.
- Keep imports stable and minimal when editing existing files.
- Avoid introducing new `sys.path.insert(...)` hacks unless unavoidable.

### Formatting

- Preserve existing formatting style in each file.
- Do not submit formatting-only churn unless requested.
- Keep comments short and useful for non-obvious trading/math logic.

### Types

- Add type hints for new or changed functions where practical.
- Continue existing use of `dataclass` and `typing` aliases.
- Prefer concrete annotations in signatures (`pd.DataFrame`, `pd.Series`, `Dict[str, Any]`).
- Do not enforce strict typing migration across untouched code.

### Naming

- `snake_case` for functions/variables.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for constants/config blocks.
- Reuse domain vocabulary consistently: `session`, `horizon`, `target`, `proba`, `thr_short`, `thr_long`.

### Data and ML safety

- Keep time index sorted before rolling/shift operations.
- Avoid look-ahead leakage; favor lagged/shifted predictors.
- Respect session boundaries when creating targets/signals.
- Handle `NaN`/`inf` explicitly before model inference.

### Error handling and logging

- Raise clear exceptions for missing data/models.
- Guard exchange/network calls with `try/except` and useful logs.
- Log actionable context (symbol, time, thresholds, exit reasons).
- Avoid silent failures.

### Secrets and credentials

- Never hardcode real API keys/tokens.
- Prefer environment variables for credentials.
- Do not commit `.env` or credential dumps.
- `start_bot.bat` currently contains key-like values; treat as sensitive and do not replicate.

## Testing expectations for code changes

- Run at least one relevant `test_*.py` script for any behavior change.
- For live trading logic, validate via offline/backtest scripts first.
- In handoff notes, include commands run and short result summary.

## Change management

- Keep diffs small and localized.
- Prefer modifying existing modules over introducing duplicates.
- Update docs/comments when behavior changes materially.
- If adding new tooling, document exact commands in this file.

## Git policy (owner-approved)

- Repository owner allows autonomous commits without additional confirmation.
- Push target: `main` branch directly.
- Commit only significant changes: logic updates that materially affect behavior/results.
- Do not commit minor cosmetic edits (small text tweaks, trivial formatting, non-functional micro-fixes).
- Pre-commit test/lint execution is optional unless the current task explicitly requires it.
- Keep baseline security rules: never commit secrets (`.env`, tokens, API keys, credential dumps).
- If a change mixes significant and minor edits, split/stage only the significant part when practical.

## Pre-handoff checklist

- `python -m compileall src` passes.
- Relevant test script(s) executed.
- No secrets added.
- Output paths remain consistent with repo conventions (`data/`, `models/`, `results/`, `logs/`).
