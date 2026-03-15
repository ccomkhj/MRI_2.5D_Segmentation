# Data Import And Validation

The modular pipeline expects a repository-local aligned dataset at `data/aligned_v2`.

The current source of truth is the sibling repository output:

```text
/Users/huijokim/personal/tcia-handler/data/aligned_v2
```

## Default Import Command

Create a repository-local symlink to the source dataset:

```bash
python tools/dataset/import_tcia_aligned.py \
  --source /Users/huijokim/personal/tcia-handler/data/aligned_v2 \
  --dest data/aligned_v2 \
  --mode link
```

Use `--mode copy` if you need a physical copy:

```bash
python tools/dataset/import_tcia_aligned.py \
  --source /Users/huijokim/personal/tcia-handler/data/aligned_v2 \
  --dest data/aligned_v2 \
  --mode copy
```

## Useful Options

- `--dry-run`: validate source and report the planned action without changing `data/aligned_v2`
- `--validate-files`: validate sample-level PNG files, not only case directories
- `--force`: replace an existing destination if it does not match the source
- `--manifest-output <path>`: write the import manifest to a custom JSON path

Example:

```bash
python tools/dataset/import_tcia_aligned.py \
  --source /Users/huijokim/personal/tcia-handler/data/aligned_v2 \
  --dest data/aligned_v2 \
  --mode link \
  --validate-files \
  --dry-run
```

## Dataset Contract

The import tool validates:

- `metadata.json` exists and is readable
- every case referenced in metadata has a directory under `data/aligned_v2/<case_id>`
- modality directories exist when metadata says they should exist
- mask directories exist when metadata says prostate or target slices are present
- optional sample-level PNG files exist when `--validate-files` is enabled

Expected case subdirectories include:

- `t2`
- `adc` when `has_adc` is true
- `calc` when `has_calc` is true
- `mask_prostate` when prostate slices exist
- `mask_target1` when target slices exist

## Output Artifact

Every import writes a JSON manifest, by default:

```text
data/aligned_v2_import_manifest.json
```

That manifest records:

- source path
- destination path
- import mode
- action taken: `create`, `replace`, or `reuse`
- case count
- sample count
- metadata hash

## Common Failure Modes

`Destination already exists and does not match source`

- rerun with `--force` if replacement is intended

`Aligned dataset validation failed`

- confirm the source repo finished producing `aligned_v2`
- rerun with `--validate-files` only if you want file-level validation

## What To Read Next

- Use [splits.md](splits.md) after `data/aligned_v2/metadata.json` is in place
- Use [research.md](research.md) if you want the import step embedded in the full workflow
