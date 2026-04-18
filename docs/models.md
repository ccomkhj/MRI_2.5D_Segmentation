# Model Families

This page lists the segmentation and classification model names you can put under `model.name` in a task config, where their config presets live, and what `model.params` keys they accept.

The registry lives in [`mri/models/registry.py`](../mri/models/registry.py). Builders live in [`mri/models/seg/monai_models.py`](../mri/models/seg/monai_models.py), [`mri/models/seg/simple_unet.py`](../mri/models/seg/simple_unet.py), and [`mri/models/cls/monai_models.py`](../mri/models/cls/monai_models.py).

## Registered Names

### Segmentation (`task.name: segmentation`)

| `model.name` | Backing class | Spatial | Notes |
|---|---|---|---|
| `simple_unet` | In-repo `SimpleUNet` (small 2D) | 2D | Lightweight baseline. Current precision leader recipe. |
| `segresnet` | MONAI `SegResNet` | 2D or 3D | Default baseline; checked-in preset is 2D. |
| `dynunet` | MONAI `DynUNet` | 2D or 3D | Dynamic kernel/stride U-Net. |
| `unet` | MONAI `UNet` | 2D or 3D | Plain U-Net. |
| `vnet` | MONAI `VNet` | 3D | Volumetric. |

### Classification (`task.name: classification`)

| `model.name` | Backing class | Spatial | Notes |
|---|---|---|---|
| `resnet101` | MONAI `resnet101` | 3D | Default checked-in preset. |
| `swin` | MONAI `SwinTransformer` | 3D | Attention-based; not always installed. |
| `vit` | MONAI `ViT` | 3D | Plain transformer. |
| `resnext101` | MONAI `resnext101_32x8d` | 3D | Wider ResNet. |
| `densenet121` | MONAI `DenseNet` | 3D | `num_classes` is auto-mapped to `out_channels`. |
| `efficientnetb7` | MONAI `EfficientNetBN` (`efficientnet-b7`) | 3D | |

## Where Configs Live

Reusable per-model config presets:

- [`mri/config/model/segmentation/segresnet.yaml`](../mri/config/model/segmentation/segresnet.yaml)
- [`mri/config/model/classification/swin.yaml`](../mri/config/model/classification/swin.yaml)
- [`mri/config/model/classification/resnet101.yaml`](../mri/config/model/classification/resnet101.yaml)

Other registered families have no checked-in preset — set `model.name` and `model.params` directly in your task config or `extends:` from a new file you add under `mri/config/model/{segmentation,classification}/`.

> **Naming gotcha.** Both `mri/config/model/segmentation/` and an empty legacy `mri/config/model/seg/` exist (likewise `classification/` and `cls/`). Use the long form (`segmentation/`, `classification/`).

## `model.params` Keys

Whatever you put under `model.params` is forwarded to the backing class. Unsupported keys are dropped with a `RuntimeWarning` (see `filter_model_kwargs` in the registry) — this is intentional so you can switch model families via `extends:` without param-conflict errors.

Common keys:

- **`simple_unet`** (in-repo): `in_channels` (default `5`), `out_channels` (default `2`). 2D only.
- **`segresnet`** / **`unet`** / **`dynunet`** / **`vnet`**: pass through to the MONAI class — see the corresponding MONAI docs for the full signature. The checked-in `segresnet.yaml` uses `in_channels: 7`, `out_channels: 2`, `spatial_dims: 2`.
- **`swin`**: `in_channels`, `num_classes`, `spatial_dims` (3D in the preset).
- **`resnet101`** / **`resnext101`**: `spatial_dims`, `n_input_channels`, `num_classes`.
- **`densenet121`**: `spatial_dims`, `in_channels`, `out_channels` (or `num_classes`, auto-mapped).
- **`efficientnetb7`**: `spatial_dims`, `in_channels`, `num_classes` (uses `efficientnet-b7` by default).

### Segmentation logit calibration (optional)

Segmentation builders extract these calibration keys before constructing the model and wrap the output in `LogitCalibrationWrapper`:

- `logit_temperature_init` (float)
- `learn_logit_temperature` (bool)
- `logit_bias_init` (float or per-channel list)
- `learn_logit_bias` (bool)

If none of these change the logits, the wrapper is skipped with a warning.

## Picking A Model

- **Segmentation baseline**: `segresnet` with the checked-in preset.
- **Sparse / positive-only target**: `simple_unet` — what the current precision leader uses (see [current_leader.md](current_leader.md)).
- **Dynamic kernel stacks**: `dynunet`.
- **Classification baseline**: `resnet101` — fastest and best-studied.
- **Attention-heavy 3D**: `swin` (verify your MONAI build includes `SwinTransformer`).

## Adding A New Family

1. Implement a builder in `mri/models/seg/` or `mri/models/cls/` and decorate it with `@register_segmentation_model("<name>")` or `@register_classification_model("<name>")`.
2. Make sure the module is imported (registry decorators only run when the module loads — check `mri/models/{seg,cls}/__init__.py`).
3. Add a preset under `mri/config/model/{segmentation,classification}/<name>.yaml` so other configs can `extends:` it.
4. Reference it from a task config with `model.name: <name>` and the params your builder expects.

## Common Failure Modes

- `KeyError: Unknown segmentation model: <name>` — the builder module wasn't imported, or the name doesn't match the `@register_*` argument exactly.
- `RuntimeWarning: Ignoring unsupported params for model '...'` — your `model.params` keys don't match the backing class signature. Fine when switching families; otherwise check the param name.
- `ImportError: MONAI not installed ...` — install requirements: `pip install -r requirements.txt`.
- Checkpoint/config mismatch (e.g. loading a `simple_unet` checkpoint with a `segresnet` config) — the config defines the architecture before the checkpoint loads. Use the config that matches the checkpoint.

## See Also

- [configuration.md](configuration.md) — how `extends:` composes presets onto task configs.
- [inference.md](inference.md) — segmentation→classification data contract.
- [current_leader.md](current_leader.md) — current best segmentation run and recipe.
