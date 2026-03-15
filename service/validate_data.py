#!/usr/bin/env python3
"""
Data Validation Script

Visually validates that data is correctly converted and ready for training.
Shows sample images, 2.5D stacks, masks (if available), and data statistics.

Usage:
    python service/validate_data.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def validate_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load and validate manifest file."""
    print(f"Loading manifest: {manifest_path}")

    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}")
        return None

    df = pd.read_csv(manifest_path)

    print(f"✓ Manifest loaded: {len(df)} rows")
    print(f"\nColumns: {', '.join(df.columns)}")

    # Check for required columns
    required = ["case_id", "series_uid", "slice_idx", "image_path"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"✗ Missing required columns: {missing}")
        return None

    print(f"✓ All required columns present")

    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Total slices: {len(df)}")
    print(f"  Unique cases: {df['case_id'].nunique()}")
    print(f"  Unique series: {df['series_uid'].nunique()}")

    # Check for masks
    if "mask_path" in df.columns:
        has_masks = df["mask_path"].notna() & (df["mask_path"] != "")
        num_with_masks = has_masks.sum()
        coverage = num_with_masks / len(df) * 100

        print(f"  Slices with masks: {num_with_masks} ({coverage:.1f}%)")

        if num_with_masks == 0:
            print(f"\n⚠ WARNING: No masks found in dataset!")
            print(f"  This means:")
            print(f"    • Training will skip all batches (loss = 0)")
            print(f"    • You need segmentation masks for supervised learning")
            print(f"\n  To add masks, run:")
            print(f"    python tools/preprocessing/process_overlay_aligned.py")
    else:
        print(f"  Mask column: Not present")

    return df


def validate_image_files(df: pd.DataFrame, num_samples: int = 5):
    """Check if image files exist and are readable."""
    print(f"\nValidating image files (checking {num_samples} samples)...")

    sample_paths = df["image_path"].head(num_samples).tolist()

    valid_count = 0
    for i, path in enumerate(sample_paths):
        path = Path(path)
        if path.exists():
            try:
                img = Image.open(path)
                img_array = np.array(img)
                print(
                    f"  ✓ [{i+1}] {path.name}: {img_array.shape}, dtype={img_array.dtype}, range=[{img_array.min()}, {img_array.max()}]"
                )
                valid_count += 1
            except Exception as e:
                print(f"  ✗ [{i+1}] {path.name}: Error reading - {e}")
        else:
            print(f"  ✗ [{i+1}] {path.name}: File not found")

    print(f"\n✓ Valid images: {valid_count}/{len(sample_paths)}")
    return valid_count > 0


def visualize_single_slice(df: pd.DataFrame, output_dir: Path):
    """Visualize individual slices."""
    print(f"\nCreating single slice visualizations...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Select samples from different series
    unique_series = df.groupby("series_uid").first().reset_index()
    num_samples = min(6, len(unique_series))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(num_samples):
        row = unique_series.iloc[i]
        img_path = Path(row["image_path"])

        if img_path.exists():
            img = Image.open(img_path).convert("L")
            img_array = np.array(img)

            axes[i].imshow(img_array, cmap="gray")
            axes[i].set_title(f"Case {row['case_id']}\nSlice {row['slice_idx']}")
            axes[i].axis("off")
        else:
            axes[i].text(
                0.5,
                0.5,
                "Image not found",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].axis("off")

    # Hide unused subplots
    for i in range(num_samples, 6):
        axes[i].axis("off")

    plt.tight_layout()
    output_path = output_dir / "single_slices.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {output_path}")


def visualize_2d5_stack(df: pd.DataFrame, output_dir: Path, stack_depth: int = 5):
    """Visualize 2.5D slice stacks."""
    print(f"\nCreating 2.5D stack visualizations (stack_depth={stack_depth})...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find a series with enough slices
    series_groups = df.groupby("series_uid")

    for series_uid, group in series_groups:
        if len(group) >= stack_depth:
            group = group.sort_values("slice_idx")

            # Take middle slices
            start_idx = len(group) // 2 - stack_depth // 2
            stack_rows = group.iloc[start_idx : start_idx + stack_depth]

            # Load stack
            stack = []
            for _, row in stack_rows.iterrows():
                img_path = Path(row["image_path"])
                if img_path.exists():
                    img = Image.open(img_path).convert("L")
                    stack.append(np.array(img))

            if len(stack) == stack_depth:
                # Create visualization
                fig, axes = plt.subplots(1, stack_depth, figsize=(20, 4))

                for i, img_array in enumerate(stack):
                    axes[i].imshow(img_array, cmap="gray")
                    axes[i].set_title(
                        f"Slice {stack_rows.iloc[i]['slice_idx']}\n{'[Central]' if i == stack_depth//2 else ''}"
                    )
                    axes[i].axis("off")

                plt.suptitle(
                    f"2.5D Stack (Case {stack_rows.iloc[0]['case_id']})", fontsize=14
                )
                plt.tight_layout()

                output_path = output_dir / "2d5_stack_example.png"
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()

                print(f"✓ Saved: {output_path}")
                print(f"  Case: {stack_rows.iloc[0]['case_id']}")
                print(f"  Series: {series_uid[:30]}...")
                print(
                    f"  Central slice: {stack_rows.iloc[stack_depth//2]['slice_idx']}"
                )

                # Also visualize as a single image showing the stack concept
                fig = plt.figure(figsize=(15, 6))
                gs = gridspec.GridSpec(
                    2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1]
                )

                # Top row: show stack slices
                ax1 = fig.add_subplot(gs[0, :])
                combined = np.concatenate(stack, axis=1)
                ax1.imshow(combined, cmap="gray")
                ax1.set_title(
                    f"2.5D Stack: {stack_depth} consecutive slices concatenated"
                )
                ax1.axis("off")

                # Bottom row: show central slice and explain
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.imshow(stack[stack_depth // 2], cmap="gray")
                ax2.set_title("Central Slice\n(Target for prediction)")
                ax2.axis("off")

                ax3 = fig.add_subplot(gs[1, 1:])
                ax3.text(
                    0.1,
                    0.5,
                    "2.5D Approach:\n\n"
                    f"• Stack {stack_depth} consecutive slices\n"
                    f"• Treat as {stack_depth}-channel 2D image\n"
                    "• Provides 3D context efficiently\n"
                    "• Central slice is prediction target\n"
                    "• Neighbors provide spatial context",
                    transform=ax3.transAxes,
                    fontsize=12,
                    verticalalignment="center",
                )
                ax3.axis("off")

                plt.tight_layout()
                output_path = output_dir / "2d5_stack_explained.png"
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()

                print(f"✓ Saved: {output_path}")

                break  # Only visualize one example

    print()


def check_processed_seg_masks(df: pd.DataFrame) -> dict:
    """Check for masks in data/processed_seg/ directory."""
    print(f"\nChecking for masks in data/processed_seg/...")

    # Get first row to determine class
    first_row = df.iloc[0]
    case_id = first_row["case_id"]
    series_uid = first_row["series_uid"]

    # Determine class from image path
    img_path_str = str(first_row["image_path"])
    class_match = None
    for i in range(1, 5):
        if f"/class{i}/" in img_path_str:
            class_match = f"class{i}"
            break

    if not class_match:
        print("  ⚠ Could not determine class from image path")
        return {}

    # Check processed_seg directory
    processed_seg_dir = Path("data/processed_seg") / class_match

    if not processed_seg_dir.exists():
        print(f"  ⚠ No processed_seg directory: {processed_seg_dir}")
        return {}

    # Find cases with masks
    cases_with_masks = list(processed_seg_dir.glob("case_*"))

    if not cases_with_masks:
        print(f"  ⚠ No cases found in {processed_seg_dir}")
        return {}

    # Count masks
    total_prostate = 0
    total_target1 = 0
    total_target2 = 0
    total_cases = len(cases_with_masks)

    for case_dir in cases_with_masks:
        # Find series directories
        series_dirs = [
            d for d in case_dir.iterdir() if d.is_dir() and d.name != "biopsies.json"
        ]

        for series_dir in series_dirs:
            prostate_dir = series_dir / "prostate"
            target1_dir = series_dir / "target1"
            target2_dir = series_dir / "target2"

            if prostate_dir.exists():
                total_prostate += len(list(prostate_dir.glob("*.png")))
            if target1_dir.exists():
                total_target1 += len(list(target1_dir.glob("*.png")))
            if target2_dir.exists():
                total_target2 += len(list(target2_dir.glob("*.png")))

    print(f"  ✓ Found masks in processed_seg/{class_match}/")
    print(f"    Cases with masks: {total_cases}")
    print(f"    Prostate masks: {total_prostate} slices")
    if total_target1 > 0:
        print(f"    Target1 masks: {total_target1} slices")
    if total_target2 > 0:
        print(f"    Target2 masks: {total_target2} slices")

    return {
        "has_masks": True,
        "total_cases": total_cases,
        "total_prostate": total_prostate,
        "total_target1": total_target1,
        "total_target2": total_target2,
        "class": class_match,
    }


def visualize_with_masks(df: pd.DataFrame, output_dir: Path):
    """Visualize images with masks (if available)."""
    print(f"\nChecking for mask visualizations...")

    # First check processed_seg directory
    mask_info = check_processed_seg_masks(df)

    if not mask_info or not mask_info.get("has_masks"):
        print("⚠ No masks found in processed_seg/")
        print("\nTo add masks:")
        print("  1. Make sure overlay data exists in data/overlay/")
        print("  2. Run: python tools/preprocessing/process_overlay_aligned.py")
        return

    # Find matching case/series with masks
    first_row = df.iloc[0]
    case_id = first_row["case_id"]
    series_uid = first_row["series_uid"]
    class_name = mask_info["class"]

    processed_seg_base = Path("data/processed_seg") / class_name
    # Format case_id with leading zeros (4 digits)
    case_dir = processed_seg_base / f"case_{int(case_id):04d}"

    if not case_dir.exists():
        print(f"⚠ Case directory not found: {case_dir}")
        return

    print(f"✓ Found masks in processed_seg/{class_name}/")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find series directory with masks
    series_dirs = [d for d in case_dir.iterdir() if d.is_dir()]

    if not series_dirs:
        print(f"⚠ No series directories in {case_dir}")
        return

    # Use first series with masks
    series_dir = series_dirs[0]

    # Define structure colors (RGB normalized to [0, 1])
    structure_colors = {
        "prostate": (1.0, 1.0, 0.0),  # Yellow
        "target1": (1.0, 0.0, 0.0),  # Red
        "target2": (0.0, 0.0, 1.0),  # Blue
    }

    # Find available structures
    available_structures = {}
    for structure_name in structure_colors.keys():
        structure_dir = series_dir / structure_name
        if structure_dir.exists():
            mask_files = sorted(list(structure_dir.glob("*.png")))
            if mask_files:
                available_structures[structure_name] = mask_files

    if not available_structures:
        print(f"⚠ No masks found in {series_dir}")
        return

    # Get slice numbers from the first available structure
    first_structure = list(available_structures.keys())[0]
    sample_mask_files = available_structures[first_structure][:6]

    # Match with images from processed/
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    num_samples = 0
    for mask_file in sample_mask_files:
        # Get slice number from mask filename
        slice_num = int(mask_file.stem)

        # Find matching image in df
        matching_row = df[
            (df["case_id"] == case_id)
            & (df["series_uid"] == series_dir.name)
            & (df["slice_idx"] == slice_num)
        ]

        if len(matching_row) > 0:
            row = matching_row.iloc[0]
            img_path = Path(row["image_path"])

            if img_path.exists():
                img = np.array(Image.open(img_path).convert("L"))

                # Show base image
                axes[num_samples].imshow(img, cmap="gray")

                # Overlay each available structure with its color
                structures_present = []
                for structure_name, color in structure_colors.items():
                    if structure_name in available_structures:
                        structure_dir = series_dir / structure_name
                        mask_path = structure_dir / f"{slice_num:04d}.png"

                        if mask_path.exists():
                            mask = np.array(Image.open(mask_path).convert("L"))

                            # Create colored mask (RGBA)
                            if mask.max() > 0:
                                mask_rgba = np.zeros((*mask.shape, 4))
                                mask_rgba[..., 0] = color[0]  # R
                                mask_rgba[..., 1] = color[1]  # G
                                mask_rgba[..., 2] = color[2]  # B
                                mask_rgba[..., 3] = 0.5 * (mask / 255.0)  # Alpha

                                axes[num_samples].imshow(mask_rgba)
                                structures_present.append(structure_name.capitalize())

                # Set title with structure info
                title = f"Case {case_id} - Slice {slice_num}"
                if structures_present:
                    title += f"\n({', '.join(structures_present)})"
                axes[num_samples].set_title(title, fontsize=9)
                axes[num_samples].axis("off")
                num_samples += 1

                if num_samples >= 6:
                    break

    # Hide unused subplots
    for i in range(num_samples, 6):
        axes[i].axis("off")

    # Add legend for structure colors
    from matplotlib.patches import Patch

    legend_elements = []
    for structure_name in available_structures.keys():
        color = structure_colors[structure_name]
        legend_elements.append(
            Patch(facecolor=color, edgecolor="black", label=structure_name.capitalize())
        )

    # Add legend to the figure
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=len(legend_elements),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 0.98),
    )

    plt.suptitle(
        f"Images with Masks from processed_seg/{class_name}/", fontsize=14, y=0.93
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = output_dir / "masks_overlay.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {output_path}")
    print(
        f"  Available structures: {', '.join([s.capitalize() for s in available_structures.keys()])}"
    )
    print(f"  Colors: Prostate=Yellow, Target1=Red, Target2=Orange")
    print(f"\n✓ Masks ARE available in data/processed_seg/!")
    print(f"  Use these for training with a custom dataset class")


def show_data_distribution(df: pd.DataFrame, output_dir: Path):
    """Show data distribution statistics."""
    print(f"\nAnalyzing data distribution...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by series and count slices
    series_counts = df.groupby("series_uid").size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of slices per series
    axes[0].hist(series_counts, bins=20, edgecolor="black")
    axes[0].set_xlabel("Number of slices")
    axes[0].set_ylabel("Number of series")
    axes[0].set_title(
        f"Slices per Series\n(mean: {series_counts.mean():.1f}, median: {series_counts.median():.1f})"
    )
    axes[0].grid(True, alpha=0.3)

    # Slices per case
    case_counts = df.groupby("case_id").size()
    axes[1].hist(case_counts, bins=20, edgecolor="black", color="orange")
    axes[1].set_xlabel("Number of slices")
    axes[1].set_ylabel("Number of cases")
    axes[1].set_title(
        f"Slices per Case\n(mean: {case_counts.mean():.1f}, median: {case_counts.median():.1f})"
    )
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "data_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {output_path}")

    # Print statistics
    print(f"\nDistribution Statistics:")
    print(
        f"  Slices per series: min={series_counts.min()}, max={series_counts.max()}, mean={series_counts.mean():.1f}"
    )
    print(
        f"  Slices per case: min={case_counts.min()}, max={case_counts.max()}, mean={case_counts.mean():.1f}"
    )


if __name__ == "__main__":
    print_banner("DATA VALIDATION")

    # Find available manifests
    processed_dir = Path("data/processed")

    if not processed_dir.exists():
        print(f"✗ Processed data directory not found: {processed_dir}")
        print("\nPlease run the required preprocessing tools first.")
        sys.exit(1)

    manifests = list(processed_dir.glob("class*/manifest.csv"))

    if not manifests:
        print(f"✗ No manifest files found in {processed_dir}")
        print("\nPlease run the required preprocessing tools first.")
        sys.exit(1)

    print(f"Found {len(manifests)} manifest file(s):")
    for i, m in enumerate(manifests, 1):
        print(f"  {i}. {m}")

    # Process each manifest
    for manifest_path in manifests:
        class_name = manifest_path.parent.name

        print_banner(f"Validating {class_name}")

        # Load and validate manifest
        df = validate_manifest(manifest_path)

        if df is None:
            continue

        # Validate image files
        if not validate_image_files(df):
            print(f"⚠ Skipping visualizations for {class_name} (image file issues)")
            continue

        # Create output directory
        output_dir = Path("validation_results") / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualizations
        visualize_single_slice(df, output_dir)
        visualize_2d5_stack(df, output_dir)
        visualize_with_masks(df, output_dir)
        show_data_distribution(df, output_dir)

        print(f"\n✓ Validation complete for {class_name}")
        print(f"  Results saved to: {output_dir}/")

    # Final summary
    print_banner("VALIDATION SUMMARY")

    print("✓ Data validation complete!")
    print(f"\nResults saved to: validation_results/")
    print("\nGenerated visualizations:")
    print("  • single_slices.png - Sample individual slices")
    print("  • 2d5_stack_example.png - Example 2.5D stack")
    print("  • 2d5_stack_explained.png - Explanation of 2.5D approach")
    print("  • data_distribution.png - Data statistics")
    print("  • masks_overlay.png - Masks (if available)")

    print("\nNext steps:")
    print("  1. Review the visualizations to confirm data quality")
    print(
        "  2. If no masks found, run: python tools/preprocessing/process_overlay_aligned.py"
    )
    print("  3. Once masks are available, train: python mri/cli/train.py --config mri/config/task/segmentation.yaml")

    print("\n" + "=" * 80 + "\n")
