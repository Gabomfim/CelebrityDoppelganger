# CelebrityDoppelganger

Find your celebrity lookalike. This repository includes a PyTorch training pipeline that
fine-tunes a VGGFace2-pretrained InceptionResnetV1 using supervised contrastive learning plus
cross-entropy classification.

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

Authenticate once with `uv run modal setup`, then create the W&B secret without putting the key in
Git:

```bash
uv run modal secret create wandb-secret WANDB_API_KEY=YOUR_KEY
uv run modal secret create aws-secret \
  AWS_ACCESS_KEY_ID=YOUR_KEY_ID \
  AWS_SECRET_ACCESS_KEY=YOUR_SECRET \
  AWS_DEFAULT_REGION=us-east-1 \
  PROTOTYPE_DATABASE_URI=s3://YOUR_BUCKET/celebrity-doppelganger/prototypes.lancedb
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
  --database-uri s3://YOUR_BUCKET/celebrity-doppelganger/prototypes.lancedb
```

Metadata is cached in the database directory as `person_metadata.json`, so interrupted enrichment
resumes without repeating completed Wikidata requests. Use `--offline-metadata` when network access
is unavailable; unresolved fields remain null and are never guessed.

For Modal, create an `aws-secret` containing `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`, and `PROTOTYPE_DATABASE_URI`. Prefer temporary credentials or an IAM role with
access restricted to the selected S3 prefix.

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
