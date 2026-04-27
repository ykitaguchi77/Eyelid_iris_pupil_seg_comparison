# Repository Guidelines

## Project Structure & Module Organization
- Source notebooks live at repo root (e.g., `train.ipynb`, `ablation_*.ipynb`).
- Python helpers and utilities: `tools/` and `model/`.
- Data and images: `Images/` (input samples); intermediate artifacts: `cache/`; experiment outputs: `results/`.
- Papers/notes: `Article/`, logs and experiment summaries: `*.md`.
- Dependencies: `requirements.txt`. Avoid editing `venv/` directly.

## Build, Test, and Development Commands
- Create env: `python -m venv venv && venv\\Scripts\\activate` (Windows) or `source venv/bin/activate` (Unix).
- Install deps: `pip install -r requirements.txt`.
- Launch notebooks: `jupyter lab` (or `jupyter notebook`).
- Run a script: `python tools/<script>.py` (e.g., dataset prep or evaluation).
- Quick sanity tests (if present): `pytest -q`.

## Coding Style & Naming Conventions
- Python 3; PEP 8; 4‑space indentation.
- Files: `snake_case.py`; notebooks: `verb_subject_detail.ipynb` (e.g., `train_rf-detr.ipynb`).
- Functions/vars: `snake_case`; Classes: `PascalCase`.
- Prefer pure functions in `tools/` and model code in `model/`.
- Formatting: use `black` and `isort` if installed; otherwise follow PEP 8.

## Testing Guidelines
- Use `pytest`; place tests under `tests/` mirroring package structure.
- Name tests `test_*.py`; one behavior per test; include minimal fixtures.
- For metrics code, add small deterministic samples under `Images/` or a dedicated `tests/data/`.

## Commit & Pull Request Guidelines
- Commits: present tense, concise scope prefix when helpful (e.g., `data:`, `train:`, `tools:`). Example: `tools: add YOLO mask post‑proc util`.
- PRs: include purpose, key changes, how to run (commands), sample output/metrics, and screenshots when UI/plots change.
- Link related issues/experiments (`Experiment.md`, notes, or result paths under `results/`).

## Security & Configuration Tips
- Do not commit private or patient‑identifying data. Keep large models and datasets outside the repo; reference paths instead.
- Store secrets/config in environment variables or ignored files; respect `.gitignore`.
