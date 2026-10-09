# Contributing to TB Detection Model

We welcome contributions! This guide helps new contributors get up to speed with the project’s current state and the **AGENTS.md** / **RULES.md** policies.

---

## 📋 Quick start
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/TB_Detection_Model.git
   cd TB_Detection_Model
   ```
2. **Install dependencies**
   ```bash
   make requirements   # installs from requirements.txt
   pip install -e .   # install the package in editable mode
   ```
3. **Run the data extraction pipeline**
   ```bash
   python src/data/extract.py   # extracts all ZIPs into data/interim/
   python src/data/create_interim.py   # creates a random sampled interim set
   ```
   > Note: the `make data` target is a stub; invoke the scripts directly.
4. ** lint the code**
   ```bash
   make lint   # runs flake8 on the src/ directory
   ```
5. **Explore notebooks**
   Open any notebook in `notebooks/` (e.g., `01_data_exploration.ipynb`) with Jupyter.

---

## 🛠️ Development workflow
- **Follow `AGENTS.md`** for the recommended build, data‑pipeline, and workflow commands.
- **Adhere to `RULES.md`** – never commit generated files, add unit tests for new code, run `make lint` before committing, and use a fixed random seed when sampling data.
- **Add or update tests** in a `tests/` directory and run them with `pytest` (or `make test` when added).
- **Documentation updates**
  - Keep `architecture.md`, `AGENTS.md`, and this `CONTRIBUTING.md` in sync when the pipeline, model, or data layout changes.
  - Use clear headings and concise bullet points.

---

## 🤝 Pull‑request process
1. **Branch** off `main` with a descriptive name (e.g., `feat/add‑training‑script`).
2. **Make changes** following the project conventions.
3. **Run lint and tests** locally:
   ```bash
   make lint
   pytest  # after adding a test suite
   ```
4. **Commit** with a meaningful message; the commit linting is enforced by `make lint`.
5. **Open a PR** targeting `main`.
6. **Review** – a maintainer will review code style, test coverage, and adherence to sticky rules.
7. **Merge** – once approved, the PR will be merged and the CI (if configured) will run the linting step.

---

## 📚 Resources
- **AGENTS.md** – project‑wide instructions and workflow.
- **RULES.md** – sticky rules that must never be violated.
- **architecture.md** – high‑level overview of the repository structure.
- **Makefile** – useful shortcuts (`make requirements`, `make lint`).

---

## 🐞 Reporting issues
If you encounter bugs or have questions about the current scaffold, open an issue with:
- A clear title.
- Steps to reproduce (if applicable).
- Expected vs. actual behavior.

---

We look forward to your contributions! 🎉