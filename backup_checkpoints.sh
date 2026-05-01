#!/bin/bash

# ============================================================================
# Checkpoint Management & Backup Script
# ============================================================================
# Purpose: Prevent accidental checkpoint deletion and manage backups
# Usage: 
#   bash backup_checkpoints.sh          # Backup all current checkpoints
#   bash backup_checkpoints.sh --list   # Show backup contents
#   bash backup_checkpoints.sh --verify # Verify integrity
# ============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs"
BACKUP_DIR="${OUTPUTS_DIR}/checkpoints_backup"
MODELS=("segformer_upernet" "resnet101" "swin_b" "convnext_b")

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ──────────────────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

backup_model_checkpoints() {
    local model=$1
    local model_dir="${OUTPUTS_DIR}/${model}"
    
    if [ ! -d "$model_dir" ]; then
        log_warn "No output directory for $model"
        return 0
    fi
    
    # Count checkpoint files
    local checkpoint_count=$(find "$model_dir" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | wc -l)
    
    if [ $checkpoint_count -eq 0 ]; then
        log_warn "No checkpoints found for $model"
        return 0
    fi
    
    # Create timestamped backup subdirectory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_subdir="${BACKUP_DIR}/${model}_${timestamp}"
    mkdir -p "$backup_subdir"
    
    # Copy all checkpoint files
    local copied=0
    for checkpoint in "$model_dir"/*.pth; do
        if [ -f "$checkpoint" ]; then
            cp -v "$checkpoint" "$backup_subdir/" 2>&1 | sed 's/^/  /'
            ((copied++))
        fi
    done
    
    # Copy results.json if exists
    if [ -f "$model_dir/results.json" ]; then
        cp -v "$model_dir/results.json" "$backup_subdir/"
    fi
    
    log_info "Backed up $copied checkpoints for $model to $backup_subdir"
}

list_backups() {
    echo ""
    echo "Checkpoint Backups:"
    echo "────────────────────────────────────────────────────────────────"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_warn "No backup directory found"
        return 1
    fi
    
    for backup_dir in "$BACKUP_DIR"/*; do
        if [ -d "$backup_dir" ]; then
            local dir_name=$(basename "$backup_dir")
            local file_count=$(find "$backup_dir" -name "*.pth" -type f 2>/dev/null | wc -l)
            local total_size=$(du -sh "$backup_dir" | cut -f1)
            printf "%-40s Files: %d  Size: %s\n" "$dir_name" "$file_count" "$total_size"
        fi
    done
}

verify_checksums() {
    echo ""
    echo "Verifying Checkpoint Integrity..."
    echo "────────────────────────────────────────────────────────────────"
    
    local errors=0
    
    for model in "${MODELS[@]}"; do
        local model_dir="${OUTPUTS_DIR}/${model}"
        if [ -d "$model_dir" ]; then
            for checkpoint in "$model_dir"/*.pth; do
                if [ -f "$checkpoint" ]; then
                    # Try to load checkpoint with Python
                    if conda run -n encoders python -c "import torch; torch.load('$checkpoint', map_location='cpu', weights_only=False)" 2>/dev/null; then
                        log_info "$(basename $checkpoint) - OK"
                    else
                        log_error "$(basename $checkpoint) - CORRUPTED"
                        ((errors++))
                    fi
                fi
            done
        fi
    done
    
    if [ $errors -eq 0 ]; then
        log_info "All checkpoints verified OK"
    else
        log_error "Found $errors corrupted checkpoint(s)"
        return 1
    fi
}

protect_backups() {
    echo ""
    echo "Protecting Backup Checkpoints..."
    echo "────────────────────────────────────────────────────────────────"
    
    if [ -d "$BACKUP_DIR" ]; then
        # Make directories read-only (prevent accidental deletion)
        chmod -R a-w "$BACKUP_DIR"
        log_info "Backup directory set to read-only"
        
        # Show current permissions
        ls -lhd "$BACKUP_DIR"
    fi
}

unprotect_backups() {
    echo ""
    echo "Unprotecting Backup Checkpoints (for manual management)..."
    echo "────────────────────────────────────────────────────────────────"
    
    if [ -d "$BACKUP_DIR" ]; then
        chmod -R u+w "$BACKUP_DIR"
        log_info "Backup directory now writable"
    fi
}

show_current_checkpoints() {
    echo ""
    echo "Current Checkpoints (in training directories):"
    echo "────────────────────────────────────────────────────────────────"
    
    for model in "${MODELS[@]}"; do
        local model_dir="${OUTPUTS_DIR}/${model}"
        if [ -d "$model_dir" ]; then
            local file_count=$(find "$model_dir" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | wc -l)
            if [ $file_count -gt 0 ]; then
                local total_size=$(du -sh "$model_dir" | cut -f1)
                printf "%-20s: %d checkpoints  (%s)\n" "$model" "$file_count" "$total_size"
                find "$model_dir" -maxdepth 1 -name "*.pth" -type f 2>/dev/null | while read f; do
                    printf "  %-50s %s\n" "$(basename $f)" "$(du -h $f | cut -f1)"
                done
            fi
        fi
    done
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

main() {
    local command="${1:-backup}"
    
    echo "============================================================================"
    echo "Checkpoint Management Tool"
    echo "============================================================================"
    
    case $command in
        backup)
            echo ""
            echo "Backing up all model checkpoints..."
            for model in "${MODELS[@]}"; do
                backup_model_checkpoints "$model"
            done
            protect_backups
            show_current_checkpoints
            list_backups
            ;;
        list)
            show_current_checkpoints
            list_backups
            ;;
        verify)
            verify_checksums
            ;;
        protect)
            protect_backups
            ;;
        unprotect)
            unprotect_backups
            ;;
        *)
            echo "Usage: $0 {backup|list|verify|protect|unprotect}"
            echo ""
            echo "Commands:"
            echo "  backup     - Backup all current checkpoints to backup directory"
            echo "  list       - List all backups and current checkpoints"
            echo "  verify     - Verify checkpoint integrity"
            echo "  protect    - Make backups read-only (default)"
            echo "  unprotect  - Make backups writable"
            exit 1
            ;;
    esac
    
    echo ""
    echo "============================================================================"
}

main "$@"
