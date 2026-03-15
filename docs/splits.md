# Split Generation

Use one dated split file for both segmentation and classification runs.

The current recommended label space is `downstream_5class`, not the raw metadata class. This matters because cases without targets are remapped to class `0` for downstream classification.

## Generate A New Dated Split

```bash
python tools/generate_splits.py \
  --metadata data/aligned_v2/metadata.json \
  --output data/splits/2026-03-15.yaml \
  --label-space downstream_5class
```

The command also writes:

```text
data/splits/2026-03-15_summary.json
```

## What The Summary Contains

The JSON summary includes:

- selected label space
- total case count
- `train`, `val`, and `test` case counts
- per-split downstream label histograms

Use that file to verify the split is sensible before starting long runs.

## Ratios, Seed, And Stratification

Override the default `0.7,0.15,0.15` split ratios:

```bash
python tools/generate_splits.py \
  --metadata data/aligned_v2/metadata.json \
  --output data/splits/2026-03-15.yaml \
  --ratios 0.8,0.1,0.1 \
  --seed 20260315 \
  --label-space downstream_5class
```

Disable stratification only if you have a specific reason:

```bash
python tools/generate_splits.py \
  --metadata data/aligned_v2/metadata.json \
  --output data/splits/2026-03-15.yaml \
  --no-stratify
```

## Label Space Options

`downstream_5class`

- recommended default
- stratifies by the effective classification label
- maps no-target cases to class `0`

`original`

- stratifies by the raw metadata class
- not recommended for downstream classification paper runs

## Checked-In Smoke Splits

- [smoke_3case.yaml](../data/splits/smoke_3case.yaml): minimal CPU smoke validation
- [smoke_5case.yaml](../data/splits/smoke_5case.yaml): small end-to-end workflow validation

## What To Read Next

- Use [train.md](train.md) after the split file exists
- Use [research.md](research.md) if you want split generation embedded in the end-to-end runner
