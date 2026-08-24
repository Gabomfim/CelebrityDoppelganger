# Contributing to Celebrity Twin

Thanks for helping improve Celebrity Twin. Contributions should preserve its privacy-first design,
reproducible ML workflow, and cross-browser photobooth experience.

Before starting, read [`AGENTS.md`](AGENTS.md) for the project invariants that apply equally to
people and automated coding agents.

## Setup

Prerequisites:

- Git
- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Docker for container checks
- DVC/AWS access only when the task needs versioned data or production assets

```bash
git clone https://github.com/Gabomfim/CelebrityDoppelganger.git
cd CelebrityDoppelganger
uv sync --locked
cp .env.example .env
uv run pytest -q
```

The application can start locally once model and prototype assets referenced by `.env` are
available:

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open `http://127.0.0.1:8000`. Browsers allow camera access on localhost, but HTTPS is required on
non-local origins.

## Data and model assets

The canonical dataset lives at `data/source_dataset` and is tracked by DVC. Never add its images
directly to Git.

```bash
uv run dvc pull
```

Checkpoints, prototype databases, and celebrity photos are generated/versioned artifacts. Do not
commit them. See `README.md` for training on Modal, W&B logging, and prototype generation.

## Making a change

1. Create a focused branch from the latest `main`.
2. Keep changes small and add tests for new behavior or regressions.
3. Preserve the privacy guarantees and the analytics opt-out.
4. For camera changes, verify capture and upload flows at mobile and desktop breakpoints. Test
   Safari manually when possible and retain upload fallback messaging.
5. Update documentation and `.env.example` when configuration or behavior changes.

## Quality checks

Run the same core checks as CI:

```bash
uv run ruff format --check train.py build_prototypes.py refresh_metadata.py modal_train.py api.py tests
uv run ruff check train.py build_prototypes.py refresh_metadata.py modal_train.py api.py tests
uv run pytest -q
docker build --tag celebrity-twin:test .
```

Also run `git diff --check`. Validate `infra/aws/cloudformation.yml` with AWS CloudFormation when it
changes.

## Pull requests

Describe:

- what changed and why;
- privacy, data, model, and infrastructure impact;
- tests and browsers used for verification;
- screenshots for visible UI changes;
- migrations or deployment steps, if any.

Do not place credentials, personal images, biometric data, production logs, or copied datasets in
issues or pull requests. CI must pass before merge. Production deployment is performed by GitHub
Actions through AWS OIDC after merging to `main`.

## Reporting problems

For camera bugs, include the browser name/version, operating system, device type, whether the
camera permission appeared, whether the preview was black, and whether uploading three images
worked. Do not attach a real selfie; use a non-sensitive test image when reproduction requires a
file.
