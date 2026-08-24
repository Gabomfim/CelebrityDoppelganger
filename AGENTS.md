# Celebrity Twin agent policies

These instructions apply to every automated coding agent working in this repository. User
instructions take precedence when they explicitly change a requirement.

## Product invariants

- Use `data/source_dataset` as the canonical face dataset. It must remain versioned through DVC,
  not committed to Git.
- The recognition model is an InceptionResnetV1 backbone pretrained on VGGFace2 and fine-tuned
  with supervised contrastive learning and classification loss.
- Use a stratified validation split for model selection. Select the deployable checkpoint from
  validation performance, not training performance or the final epoch.
- After model selection, create one normalized class prototype from the mean embeddings of all
  available images for that identity, including the former validation images.
- Inference creates a temporary prototype from exactly three user images and retrieves only the
  three nearest class prototypes using cosine distance.
- Keep display names human-readable and free of dataset underscores. Celebrity records should
  include profession, why the person is famous, what they are most famous for, and Wikipedia URL.

## Privacy and security

- Never persist, cache, log, upload, or include user selfies or temporary user embeddings in
  analytics, exceptions, fixtures, screenshots, or debugging artifacts.
- Process inference images only in memory and discard their byte buffers and temporary prototype
  immediately after the request.
- Do not log visitor IP addresses, raw user-agent strings, celebrity matches, biometric data, or
  persistent user identifiers. Product analytics must remain allowlisted and anonymous.
- Preserve the per-browser `?analytics=off` opt-out behavior.
- Never commit credentials or populated `.env` files. Use `.env.example` for names and safe
  placeholders, local AWS profiles for development, GitHub OIDC for deployment, ECS task roles at
  runtime, and Modal secrets/OIDC for training.
- If a credential is exposed, do not reproduce or store it. Recommend revocation and replacement.

## User experience

- Keep the visual language responsive, accessible, playful, and based on crimson, gold, and red
  carpet styling.
- Support both a three-shot photobooth and exactly three uploaded images.
- Maintain camera compatibility across Safari, Chrome, Firefox, Edge, and mobile/in-app browsers,
  especially LinkedIn. Always preserve upload as the fallback.
- Tell users that images are processed in memory and never stored. Instruct them to remove
  sunglasses and keep exactly one face in frame.
- Result cards must show at most three people and link to their Wikipedia pages.

## Development workflow

- Manage Python and dependencies exclusively with `uv`; keep `uv.lock` synchronized.
- Write production workflows as Python scripts, not notebooks.
- Format and lint Python with Ruff. Run `uv run ruff format` and `uv run ruff check` on changed
  Python files.
- Add or update pytest coverage for behavior changes. Before handoff, run `uv run pytest -q` and
  relevant frontend/container/infrastructure checks.
- Use Git for code, DVC for datasets, and W&B Artifacts plus immutable S3 keys/Modal volumes for
  model versioning. Do not commit generated datasets, checkpoints, LanceDB data, or secrets.
- Keep infrastructure reproducible in `infra/aws/cloudformation.yml` and containers reproducible
  in `Dockerfile`/`compose.yaml`.
- Changes should pass GitHub Actions before merge and deploy through the existing OIDC-based
  workflow. Do not weaken CI, IAM, privacy, or tests to make a change pass.

## Working safely

- Inspect existing code and repository status before editing. Preserve unrelated user changes.
- Prefer small, reviewable changes with clear commit messages and documented verification.
- Do not perform destructive data, Git, AWS, DVC, W&B, or Modal operations without explicit user
  authorization and a verified target.
- Update README and `.env.example` when setup, public behavior, configuration, or architecture
  changes.

## Agent quick start

```bash
uv sync --locked
cp .env.example .env
uv run dvc pull
uv run ruff format --check train.py build_prototypes.py refresh_metadata.py modal_train.py api.py tests
uv run ruff check train.py build_prototypes.py refresh_metadata.py modal_train.py api.py tests
uv run pytest -q
```

Read `README.md` for architecture and operations and `CONTRIBUTING.md` for the contribution and
review workflow before making changes.
