# CelebrityDoppelganger

Find your celebrity lookalike. This repository includes a PyTorch training pipeline that
fine-tunes a VGGFace2-pretrained InceptionResnetV1 using supervised contrastive learning plus
cross-entropy classification.

The project gateway will be available at
[`gabomfim.github.io/CelebrityDoppelganger`](https://gabomfim.github.io/CelebrityDoppelganger/)
and forwards visitors to the AWS-hosted application. The current CloudFront URL is
[`d3tc24jtdkcl2p.cloudfront.net`](https://d3tc24jtdkcl2p.cloudfront.net/).

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

The FastAPI service accepts JPEG, PNG, or WebP bytes directly into memory, rejects requests over
10 MB, detects and crops a face, applies the training evaluation transform, generates an embedding,
and queries the three nearest class prototypes. User images are never
written to disk or retained.

```bash
MODEL_CHECKPOINT=models/celebrity_face_classifier.pt \
PROTOTYPE_DATABASE_URI=s3://YOUR_BUCKET/celebrity-doppelganger/prototypes.lancedb \
SOURCE_DATASET_DIR=data/source_dataset \
LINKEDIN_URL=https://www.linkedin.com/in/gabrielabsilveira/ \
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

### Docker

Copy `.env.example` to `.env`, fill in the non-secret resource URLs, and use AWS credentials from
your shell for local development. The production container runs as a non-root user with a read-only
filesystem and reads the model, prototype table, and celebrity reference photos from S3.

```bash
docker compose up --build
```

Do not place long-lived AWS keys in `.env` in production. The published ECS task should use a
least-privilege IAM task role and AWS Secrets Manager for `LINKEDIN_URL` or any future secrets.

### CI/CD

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
