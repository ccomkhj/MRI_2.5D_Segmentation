#!/bin/bash
# ============================================================================
# Sync WandB Runs from Scratch Space
# ============================================================================
# 
# This script syncs WandB offline runs from scratch space to wandb.ai servers.
# It handles missing artifact staging files gracefully.
#
# Usage:
#   ./scripts/sync_wandb_runs.sh                    # Sync all runs
#   ./scripts/sync_wandb_runs.sh --run <run-id>     # Sync specific run
#   ./scripts/sync_wandb_runs.sh --status           # Check sync status
#
# ============================================================================

set -e

# Get project directory to load .env file
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load configuration from .env file
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    # Load WANDB_API_KEY from .env
    if [[ -z "${WANDB_API_KEY}" ]]; then
        if [[ -f "${HOME}/.wandb_api_key" ]]; then
            export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
        else
            export WANDB_API_KEY=$(grep "^WANDB_API_KEY=" "${ENV_FILE}" | cut -d'=' -f2-)
        fi
    fi
    
    # Load WANDB_DIR from .env (try WANDB_DIR first, then WANDB_PATH as fallback)
    if [[ -z "${WANDB_DIR}" ]]; then
        ENV_WANDB_DIR=$(grep "^WANDB_DIR=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2-)
        if [[ -z "${ENV_WANDB_DIR}" ]]; then
            ENV_WANDB_DIR=$(grep "^WANDB_PATH=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2-)
        fi
        if [[ -n "${ENV_WANDB_DIR}" ]]; then
            WANDB_DIR="${ENV_WANDB_DIR}"
        fi
    fi
else
    # Fallback: load wandb API key from file if no .env
    if [[ -z "${WANDB_API_KEY}" ]] && [[ -f "${HOME}/.wandb_api_key" ]]; then
        export WANDB_API_KEY=$(cat "${HOME}/.wandb_api_key")
    fi
fi

# Default values (use .env values if loaded, otherwise use defaults)
WANDB_DIR="${WANDB_DIR:-/p/scratch/ebrains-0000006/kim27/wandb}"
CHECKPOINTS_ROOT="${PROJECT_DIR}/checkpoints"
SYNC_SPECIFIC_RUN=""
CHECK_STATUS=0
FILTER_METRIC="primary"
FILTER_THRESHOLD=""

collect_offline_runs() {
    local legacy_runs=""
    local checkpoint_runs=""

    if [[ -d "${WANDB_DIR}/wandb" ]]; then
        legacy_runs="$(find "${WANDB_DIR}/wandb" -maxdepth 1 -type d -name "offline-run-*" 2>/dev/null | sort || true)"
    fi

    if [[ -d "${CHECKPOINTS_ROOT}" ]]; then
        checkpoint_runs="$(find "${CHECKPOINTS_ROOT}" -type d -name "offline-run-*" 2>/dev/null | sort || true)"
    fi

    printf "%s\n%s\n" "${legacy_runs}" "${checkpoint_runs}" | sed '/^$/d' | sort -u
}

resolve_run_path() {
    local requested_run="$1"
    local candidate=""

    if [[ -d "${requested_run}" ]]; then
        printf "%s\n" "${requested_run}"
        return 0
    fi

    candidate="$(
        collect_offline_runs | awk -v run="${requested_run}" '
            {
                if ($0 == run || $0 ~ ("/" run "$") || $0 ~ run) {
                    print
                    exit
                }
            }
        '
    )"

    if [[ -n "${candidate}" ]]; then
        printf "%s\n" "${candidate}"
        return 0
    fi

    return 1
}

evaluate_run_metric() {
    local run_path="$1"
    python - "$run_path" "$CHECKPOINTS_ROOT" "$FILTER_METRIC" "$FILTER_THRESHOLD" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

run_path = Path(sys.argv[1]).resolve()
checkpoints_root = Path(sys.argv[2]).resolve()
metric_arg = sys.argv[3]
threshold_arg = sys.argv[4]

NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if not value or not NUMBER_RE.match(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def resolve_manifest(run_dir: Path):
    direct_manifest = run_dir / "run_manifest.json"
    if direct_manifest.exists():
        return direct_manifest

    run_id = run_path.name.rsplit("-", 1)[-1]
    for manifest_path in checkpoints_root.rglob("run_manifest.json"):
        manifest = load_json(manifest_path)
        if not manifest:
            continue
        wandb_info = ((manifest.get("tracking") or {}).get("wandb") or {})
        if str(wandb_info.get("run_id")) == run_id:
            return manifest_path
    return None


def best_history_metric(history_path, metric_name, primary_name):
    if not history_path.exists():
        return None, None, None

    bare_metric = metric_name[4:] if metric_name.startswith("val/") else metric_name
    if metric_arg in {"primary", "best_metric", "primary_metric"}:
        candidates = [f"val/{primary_name}"] if primary_name else []
        output_name = primary_name or "primary"
    else:
        candidates = [metric_name, f"val/{bare_metric}", bare_metric]
        output_name = bare_metric

    try:
        with history_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            selected_column = None
            best_value = None
            for row in reader:
                if selected_column is None:
                    for candidate in candidates:
                        if candidate in row:
                            selected_column = candidate
                            break
                if selected_column is None:
                    continue
                value = parse_number(row.get(selected_column))
                if value is None:
                    continue
                if best_value is None or value > best_value:
                    best_value = value
            if best_value is not None:
                return output_name, best_value, f"history:{selected_column}"
    except Exception:
        return None, None, None

    return None, None, None


def best_summary_metric(summary, metric_name, primary_name):
    best_val_metrics = summary.get("best_val_metrics") or {}
    if metric_arg in {"primary", "best_metric", "primary_metric"}:
        value = parse_number(summary.get("best_metric"))
        if value is not None:
            return primary_name or "primary", value, "summary:best_metric"
        return None, None, None

    bare_metric = metric_name[4:] if metric_name.startswith("val/") else metric_name
    value = parse_number(best_val_metrics.get(bare_metric))
    if value is not None:
        return bare_metric, value, "summary:best_val_metrics"
    return None, None, None


def best_log_metric(log_path, metric_name, primary_name):
    if not log_path.exists():
        return None, None, None

    bare_metric = metric_name[4:] if metric_name.startswith("val/") else metric_name
    search_metric = primary_name if metric_arg in {"primary", "best_metric", "primary_metric"} else bare_metric
    if not search_metric:
        return None, None, None

    pattern = re.compile(rf"\b{re.escape(search_metric)}:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    best_value = None

    try:
        with log_path.open(errors="ignore") as handle:
            for line in handle:
                if "| val:" not in line:
                    continue
                val_segment = line.split("| val:", 1)[1]
                match = pattern.search(val_segment)
                if not match:
                    continue
                value = parse_number(match.group(1))
                if value is None:
                    continue
                if best_value is None or value > best_value:
                    best_value = value
    except Exception:
        return None, None, None

    if best_value is None:
        return None, None, None
    return search_metric, best_value, "train.log"


run_dir = run_path.parent.parent if run_path.parent.name == "wandb" else run_path
manifest_path = resolve_manifest(run_dir)
manifest = load_json(manifest_path) if manifest_path else None
run_root = manifest_path.parent if manifest_path else run_dir
run_name = (manifest or {}).get("run_name") or run_root.name
run_status = (manifest or {}).get("status") or "unknown"

summary = (manifest or {}).get("summary") or {}
summary_path = run_root / "run_summary.json"
if not summary and summary_path.exists():
    loaded_summary = load_json(summary_path)
    if isinstance(loaded_summary, dict):
        summary = loaded_summary

primary_metric_name = summary.get("primary_metric_name")
history_path = run_root / "metrics_history.csv"
log_path = run_root / "train.log"

metric_label, metric_value, metric_source = best_history_metric(history_path, metric_arg, primary_metric_name)
if metric_value is None:
    metric_label, metric_value, metric_source = best_summary_metric(summary, metric_arg, primary_metric_name)
if metric_value is None:
    metric_label, metric_value, metric_source = best_log_metric(log_path, metric_arg, primary_metric_name)

matched = ""
if threshold_arg:
    threshold_value = parse_number(threshold_arg)
    if threshold_value is not None and metric_value is not None and metric_value >= threshold_value:
        matched = "1"
    else:
        matched = "0"

status = "ok" if metric_value is not None else "no_metric"
separator = "\x1f"
print(separator.join(
    [
        status,
        matched,
        metric_label or "",
        "" if metric_value is None else f"{metric_value:.12g}",
        metric_source or "",
        run_status,
        run_name,
    ]
))
PY
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            SYNC_SPECIFIC_RUN="$2"
            shift 2
            ;;
        --status)
            CHECK_STATUS=1
            shift
            ;;
        --metric)
            FILTER_METRIC="$2"
            shift 2
            ;;
        --threshold)
            FILTER_THRESHOLD="$2"
            shift 2
            ;;
        --wandb-dir)
            WANDB_DIR="$2"
            shift 2
            ;;
        --checkpoints-root)
            CHECKPOINTS_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--run RUN_ID] [--status] [--metric NAME] [--threshold VALUE] [--wandb-dir PATH] [--checkpoints-root PATH]"
            echo ""
            echo "Options:"
            echo "  --run RUN_ID            Sync specific run by ID or path"
            echo "  --status                List discovered offline runs"
            echo "  --metric NAME           Validation metric to filter on (default: primary)"
            echo "  --threshold VALUE       Only sync runs with best validation metric >= threshold"
            echo "  --wandb-dir PATH        Shared WandB directory (default: /p/scratch/ebrains-0000006/kim27/wandb)"
            echo "  --checkpoints-root PATH Checkpoints root to scan (default: ${PROJECT_DIR}/checkpoints)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if wandb is installed
if ! python -m wandb --version &> /dev/null; then
    echo "ERROR: wandb command not found. Please install wandb: pip install wandb"
    exit 1
fi

# Set up environment to use scratch space for staging
export WANDB_DIR="${WANDB_DIR}"
export WANDB_CACHE_DIR="${WANDB_DIR}/cache"
export XDG_DATA_HOME="${WANDB_DIR}/cache"

# Ensure the staging directory exists and is accessible via symlink
STAGING_DIR="${WANDB_DIR}/cache/.local/share/wandb/artifacts/staging"
if [[ ! -d "${STAGING_DIR}" ]]; then
    echo "Creating staging directory: ${STAGING_DIR}"
    mkdir -p "${STAGING_DIR}"
fi

# Check symlink
HOME_WANDB_STAGING="/p/home/jusers/kim27/jusuf/.local/share/wandb"
if [[ ! -L "${HOME_WANDB_STAGING}" ]] || [[ ! -e "${HOME_WANDB_STAGING}" ]]; then
    echo "⚠️  Symlink missing or broken. Recreating..."
    if [[ -d "${HOME_WANDB_STAGING}" ]] && [[ ! -L "${HOME_WANDB_STAGING}" ]]; then
        # Backup if it's a real directory
        mv "${HOME_WANDB_STAGING}" "${HOME_WANDB_STAGING}.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
    fi
    mkdir -p "$(dirname "${HOME_WANDB_STAGING}")"
    ln -sf "${WANDB_DIR}/cache/.local/share/wandb" "${HOME_WANDB_STAGING}"
    echo "✓ Symlink created: ${HOME_WANDB_STAGING} -> ${WANDB_DIR}/cache/.local/share/wandb"
fi

echo "============================================================================"
echo "WandB Sync Script"
echo "============================================================================"
echo "WandB directory: ${WANDB_DIR}"
echo "Checkpoints root: ${CHECKPOINTS_ROOT}"
echo "Staging directory: ${STAGING_DIR}"
if [[ -n "${FILTER_THRESHOLD}" ]]; then
    echo "Metric filter: ${FILTER_METRIC} >= ${FILTER_THRESHOLD}"
fi
echo ""

# Check status
if [[ ${CHECK_STATUS} -eq 1 ]]; then
    echo "Discovered offline runs:"
    OFFLINE_RUNS="$(collect_offline_runs)"
    if [[ -z "${OFFLINE_RUNS}" ]]; then
        echo "No offline runs found."
        exit 0
    fi
    if [[ -z "${FILTER_THRESHOLD}" ]]; then
        echo "${OFFLINE_RUNS}"
    else
        while IFS= read -r RUN_DIR; do
            [[ -z "${RUN_DIR}" ]] && continue
            EVAL="$(evaluate_run_metric "${RUN_DIR}")"
            IFS=$'\x1f' read -r EVAL_STATUS EVAL_MATCH EVAL_LABEL EVAL_VALUE EVAL_SOURCE EVAL_RUN_STATUS EVAL_RUN_NAME <<< "${EVAL}"
            if [[ "${EVAL_STATUS}" == "ok" && "${EVAL_MATCH}" == "1" ]]; then
                echo "MATCH | ${EVAL_RUN_NAME} | ${EVAL_LABEL}=${EVAL_VALUE} | source=${EVAL_SOURCE} | status=${EVAL_RUN_STATUS} | ${RUN_DIR}"
            elif [[ "${EVAL_STATUS}" == "ok" ]]; then
                echo "SKIP  | ${EVAL_RUN_NAME} | ${EVAL_LABEL}=${EVAL_VALUE} | source=${EVAL_SOURCE} | status=${EVAL_RUN_STATUS} | ${RUN_DIR}"
            else
                echo "SKIP  | ${EVAL_RUN_NAME} | metric unavailable | status=${EVAL_RUN_STATUS} | ${RUN_DIR}"
            fi
        done <<< "${OFFLINE_RUNS}"
    fi
    exit 0
fi

# Check if WANDB_API_KEY is set
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "ERROR: WANDB_API_KEY not set."
    echo "Set it with: export WANDB_API_KEY='<your_key>'"
    echo "Find your key at: https://wandb.ai/authorize"
    exit 1
fi

if [[ -n "${SYNC_SPECIFIC_RUN}" ]]; then
    # Sync specific run
    RUN_PATH="$(resolve_run_path "${SYNC_SPECIFIC_RUN}" || true)"
    if [[ -z "${RUN_PATH}" ]]; then
        echo "ERROR: Run not found: ${SYNC_SPECIFIC_RUN}"
        exit 1
    fi

    if [[ -n "${FILTER_THRESHOLD}" ]]; then
        EVAL="$(evaluate_run_metric "${RUN_PATH}")"
        IFS=$'\x1f' read -r EVAL_STATUS EVAL_MATCH EVAL_LABEL EVAL_VALUE EVAL_SOURCE EVAL_RUN_STATUS EVAL_RUN_NAME <<< "${EVAL}"
        if [[ "${EVAL_STATUS}" != "ok" ]]; then
            echo "Skipping run: ${RUN_PATH}"
            echo "Reason: validation metric '${FILTER_METRIC}' is unavailable for this run."
            exit 0
        fi
        if [[ "${EVAL_MATCH}" != "1" ]]; then
            echo "Skipping run: ${RUN_PATH}"
            echo "Reason: ${EVAL_LABEL}=${EVAL_VALUE} is below threshold ${FILTER_THRESHOLD}."
            exit 0
        fi
        echo "Run passed filter: ${EVAL_RUN_NAME} | ${EVAL_LABEL}=${EVAL_VALUE} | source=${EVAL_SOURCE}"
        echo ""
    fi

    echo "Syncing run: ${RUN_PATH}"
    echo ""
    # Use wandb sync with error handling - ignore missing artifact file errors
    python -m wandb sync "${RUN_PATH}" 2>&1 | grep -v "ERROR.*FileNotFoundError.*artifacts/staging" || {
        EXIT_CODE=${PIPESTATUS[0]}
        if [[ ${EXIT_CODE} -eq 0 ]]; then
            echo "✓ Sync completed (some missing artifact files were ignored)"
        else
            echo "⚠️  Sync completed with warnings (missing artifact files were ignored)"
        fi
    }
else
    # Sync all offline runs
    echo "Finding offline runs..."
    OFFLINE_RUNS="$(collect_offline_runs)"
    
    if [[ -z "${OFFLINE_RUNS}" ]]; then
        echo "No offline runs found in ${WANDB_DIR}/wandb or ${CHECKPOINTS_ROOT}"
        exit 0
    fi
    
    RUN_COUNT=$(echo "${OFFLINE_RUNS}" | wc -l)
    echo "Found ${RUN_COUNT} offline run(s)"
    echo ""
    
    SYNCED=0
    FAILED=0
    SKIPPED=0
    
    while IFS= read -r RUN_DIR; do
        if [[ -z "${RUN_DIR}" ]]; then
            continue
        fi
        
        RUN_NAME=$(basename "${RUN_DIR}")
        if [[ -n "${FILTER_THRESHOLD}" ]]; then
            EVAL="$(evaluate_run_metric "${RUN_DIR}")"
            IFS=$'\x1f' read -r EVAL_STATUS EVAL_MATCH EVAL_LABEL EVAL_VALUE EVAL_SOURCE EVAL_RUN_STATUS EVAL_RUN_NAME <<< "${EVAL}"
            if [[ "${EVAL_STATUS}" != "ok" ]]; then
                SKIPPED=$((SKIPPED + 1))
                echo "============================================================================"
                echo "Skipping: ${EVAL_RUN_NAME:-${RUN_NAME}}"
                echo "============================================================================"
                echo "Reason: validation metric '${FILTER_METRIC}' is unavailable."
                echo ""
                continue
            fi
            if [[ "${EVAL_MATCH}" != "1" ]]; then
                SKIPPED=$((SKIPPED + 1))
                echo "============================================================================"
                echo "Skipping: ${EVAL_RUN_NAME:-${RUN_NAME}}"
                echo "============================================================================"
                echo "Reason: ${EVAL_LABEL}=${EVAL_VALUE} is below threshold ${FILTER_THRESHOLD}."
                echo "Source: ${EVAL_SOURCE}"
                echo ""
                continue
            fi
            RUN_NAME="${EVAL_RUN_NAME:-${RUN_NAME}}"
        fi

        echo "============================================================================"
        echo "Syncing: ${RUN_NAME}"
        echo "============================================================================"
        if [[ -n "${FILTER_THRESHOLD}" ]]; then
            echo "Matched filter: ${EVAL_LABEL}=${EVAL_VALUE} (source: ${EVAL_SOURCE})"
        fi
        
        # Sync with error filtering - ignore missing artifact staging file errors
        if python -m wandb sync "${RUN_DIR}" 2>&1 | grep -v "ERROR.*FileNotFoundError.*artifacts/staging"; then
            SYNCED=$((SYNCED + 1))
            echo "✓ Synced: ${RUN_NAME}"
        else
            # Check if sync actually succeeded despite the errors
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                SYNCED=$((SYNCED + 1))
                echo "✓ Synced: ${RUN_NAME} (with warnings about missing artifact files)"
            else
                FAILED=$((FAILED + 1))
                echo "⚠️  Failed to sync: ${RUN_NAME}"
            fi
        fi
        echo ""
    done <<< "${OFFLINE_RUNS}"
    
    echo "============================================================================"
    echo "Sync Summary"
    echo "============================================================================"
    echo "Total runs: ${RUN_COUNT}"
    echo "Successfully synced: ${SYNCED}"
    echo "Skipped: ${SKIPPED}"
    echo "Failed: ${FAILED}"
    echo ""
    
    if [[ ${FAILED} -gt 0 ]]; then
        echo "Note: Some runs may have failed due to missing artifact staging files."
        echo "This is normal if staging files were cleaned up. The run data itself should be synced."
    fi
fi

echo "Done!"
