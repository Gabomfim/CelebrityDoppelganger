# Celebrity Twin

> A privacy-first, end-to-end face-recognition application that finds your three closest
> celebrity lookalikes from a temporary three-selfie prototype.

[![CI/CD](https://github.com/Gabomfim/CelebrityDoppelganger/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Gabomfim/CelebrityDoppelganger/actions/workflows/ci-cd.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-supervised%20contrastive-EE4C2C)](https://pytorch.org/)
[![AWS](https://img.shields.io/badge/AWS-ECS%20%7C%20S3%20%7C%20CloudFront-FF9900)](https://aws.amazon.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/)

**[Try Celebrity Twin](https://gabomfim.github.io/CelebrityDoppelganger/)** ·
**[CloudFront application](https://d3tc24jtdkcl2p.cloudfront.net/)** ·
**[Gabriel Silveira on LinkedIn](https://www.linkedin.com/in/gabrielabsilveira/)**

The model uses an InceptionResnetV1 backbone pretrained on VGGFace2 and fine-tuned with
supervised contrastive learning plus cross-entropy classification. The validation-selected model
achieved **92.76% validation accuracy**.

The project gateway will be available at
[`gabomfim.github.io/CelebrityDoppelganger`](https://gabomfim.github.io/CelebrityDoppelganger/)
and forwards visitors to the AWS-hosted application. The current CloudFront URL is
[`d3tc24jtdkcl2p.cloudfront.net`](https://d3tc24jtdkcl2p.cloudfront.net/).

## How it works

```mermaid
flowchart LR
    A[Three selfies or uploads] --> B[MTCNN face detection]
    B --> C[Training-equivalent preprocessing]
    C --> D[InceptionResnetV1 embeddings]
    D --> E[Temporary mean user prototype]
    E --> F[Cosine search in LanceDB]
    G[Mean celebrity prototypes on S3] --> F
    F --> H[Three closest celebrities]
```

1. The browser captures or accepts exactly three images.
2. FastAPI processes them only in memory and requires one clear face in every image.
3. The fine-tuned network produces normalized embeddings whose mean is the temporary user
   prototype.
4. LanceDB compares it by cosine distance with one mean prototype per celebrity class.
5. The frontend displays at most three matches with photos, occupations, biographies, and
   Wikipedia links.

## Technology

| Area | Tools |
| --- | --- |
| Model | PyTorch, facenet-pytorch, InceptionResnetV1, supervised contrastive learning |
| Training | Modal GPU, Weights & Biases, stratified validation, early stopping |
| Data and models | DVC, Git, W&B Artifacts, AWS S3, LanceDB |
| API and UI | FastAPI, responsive HTML/CSS/JavaScript photobooth |
| Platform | Docker, AWS ECS Fargate, ECR, CloudFront, CloudWatch |
| Delivery | GitHub Actions, AWS OIDC, Ruff, pytest, uv |

## Quick start

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), Git, and DVC credentials for the
versioned dataset and model assets.

```bash
git clone https://github.com/Gabomfim/CelebrityDoppelganger.git
cd CelebrityDoppelganger
uv sync --locked
cp .env.example .env
uv run dvc pull
uv run pytest -q
```

Automated coding agents should read [`AGENTS.md`](AGENTS.md) before editing. Human contributors
and agent operators can follow [`CONTRIBUTING.md`](CONTRIBUTING.md) for environment setup, quality
checks, privacy rules, and pull-request expectations.

## Reproducible training

The dataset must use ImageFolder-style directories (`one_directory_per_identity/image.jpg`).
The raw, tracked DVC dataset is expected at `data/source_dataset`. Resize, crop, normalization, and
training augmentation are applied at load time, leaving the versioned source images unchanged.

```bash
uv sync
uv run dvc pull                 # after a DVC pointer and remote are configured
uv run python train.py --offline --epochs 1  # local smoke run without W&B credentials
```

To create the initial pointer from this repository's current parent-level dataset (make sure the
cloud-synced files are downloaded locally first):

```bash
mkdir -p data
uv run dvc add --out data/source_dataset ../source_dataset
```

For normal tracked training, set `WANDB_API_KEY` and omit `--offline`. Every epoch is saved as a
checkpoint, the best checkpoint is copied to `models/celebrity_face_classifier.pt`, and the final
model is logged as a versioned W&B model artifact.

## Modal GPU training

Authenticate once and create the W&B secret without putting the key in Git. AWS access uses Modal
OIDC to assume `CelebrityTwinModalTrainingRole`; no long-lived AWS key is stored in Modal:

```bash
uv run modal secret create wandb-secret WANDB_API_KEY=YOUR_KEY
uv run modal run modal_train.py --epochs 30 --run-name supcon-v1
uv run modal volume get celebrity-doppelganger-models supcon-v1/celebrity_face_classifier.pt models/celebrity_face_classifier.pt
```

The Modal job uses an L4 GPU and saves checkpoints into the persistent
`celebrity-doppelganger-models` Volume. After training, it also builds
the `person_prototypes` LanceDB table at the S3 URI in `PROTOTYPE_DATABASE_URI`, containing the
normalized mean embedding and enriched person metadata for every dataset identity.

To rebuild prototypes locally from a versioned checkpoint:

```bash
uv run python build_prototypes.py \
  --checkpoint models/celebrity_face_classifier.pt \
  --database-uri s3://YOUR_BUCKET/celebrity-doppelganger/prototypes.lancedb \
  --photo-s3-prefix s3://YOUR_BUCKET/celebrity-doppelganger/photos
```

Metadata is cached in the database directory as `person_metadata.json`, so interrupted enrichment
resumes without repeating completed Wikidata requests. Use `--offline-metadata` when network access
is unavailable; unresolved fields remain null and are never guessed.

The AWS trust policy is restricted to the configured Modal workspace, app, and `train_remote`
function. The assumed role can access only the project bucket.

## Data and model versioning

Git versions source code and the DVC pointer. DVC versions the image dataset. W&B Artifacts versions
trained models; the Modal Volume retains all physical checkpoint files by run name. Configure a DVC
remote before collaborating or backing up data:

```bash
uv run dvc remote add -d storage s3://YOUR_BUCKET/celebrity-doppelganger
uv run dvc push
```

Run quality checks with `uv run ruff format --check .`, `uv run ruff check .`, and
`uv run pytest`.

## Private selfie web service

The photobooth captures or accepts exactly three JPEG, PNG, or WebP images. FastAPI processes each
image directly in memory, detects and crops its face, applies the training evaluation transform,
and averages the three normalized embeddings into a temporary user prototype. That prototype is
queried against the three nearest class prototypes. Images and the user prototype are never written
to disk or retained.

The training pipeline keeps a stratified validation split during fine-tuning. Training runs for up
to 60 epochs with early stopping after 12 epochs without validation improvement, and deployment
uses the checkpoint with the highest validation accuracy rather than the final epoch. Once that
checkpoint is selected, all available class images are used only to calculate the mean celebrity
prototype embeddings.

```bash
MODEL_CHECKPOINT=models/celebrity_face_classifier.pt \
PROTOTYPE_DATABASE_URI=s3://YOUR_BUCKET/celebrity-doppelganger/prototypes.lancedb \
SOURCE_DATASET_DIR=data/source_dataset \
LINKEDIN_URL=https://www.linkedin.com/in/gabrielabsilveira/ \
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

## Privacy and analytics

- Selfies are decoded, processed, and discarded in memory; they are never written to storage.
- The temporary user embedding is discarded immediately after the nearest-neighbor query.
- Analytics contains only allowlisted interaction events and coarse browser, device, and OS
  categories—never photos, embeddings, matches, raw user agents, or persistent user identifiers.
- Uvicorn access logging is disabled in production, preventing visitor IP addresses from being
  written to the application CloudWatch log.

To persistently disable product analytics for one browser, open:

```text
https://d3tc24jtdkcl2p.cloudfront.net/?analytics=off
```

To opt that browser back in, open the same URL with `?analytics=on`.

## Docker

Copy `.env.example` to `.env`, fill in the non-secret resource URLs, and use AWS credentials from
your shell for local development. The production container runs as a non-root user and reads the
model, prototype table, and celebrity reference photos from S3.

```bash
docker compose up --build
```

Do not place long-lived AWS keys in `.env` in production. The published ECS task should use a
least-privilege IAM task role and AWS Secrets Manager for `LINKEDIN_URL` or any future secrets.

## CI/CD and AWS

GitHub Actions runs locked-environment Ruff checks, tests, and a production Docker build for pull
requests and `main`. Successful `main` builds use GitHub OIDC to publish an immutable commit-SHA
image to ECR and perform a monitored ECS rolling deployment.

Configure these GitHub production-environment variables after provisioning AWS:
`AWS_ROLE_ARN`, `AWS_REGION`, `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`,
`ECS_TASK_DEFINITION`, and `ECS_CONTAINER_NAME`.

Infrastructure is defined in `infra/aws/cloudformation.yml`. It provisions the network, public
HTTPS CloudFront endpoint, load balancer, ECS Fargate service, least-privilege task roles, and the
GitHub OIDC deployment role. The service intentionally starts with zero tasks; scale it to one only
after the trained checkpoint and prototype database have been published to S3.

## Project structure

```text
api.py                         FastAPI inference and anonymous analytics API
train.py                       Supervised contrastive training pipeline
modal_train.py                 Modal GPU entry point
build_prototypes.py            Mean class-prototype and LanceDB builder
refresh_metadata.py            Person metadata enrichment
web/static/                    Responsive photobooth frontend
tests/                         API, privacy, camera, and training tests
infra/aws/cloudformation.yml   Reproducible AWS infrastructure and dashboard
.github/workflows/             CI/CD and GitHub Pages workflows
```

## Author

Built by [Gabriel Silveira](https://www.linkedin.com/in/gabrielabsilveira/) with supervised
contrastive prototype learning and Codex-assisted software development.
