#!/bin/bash

# ============================================================================
# tmux Session Launcher for Sequential Model Training
# ============================================================================

SESSION_NAME="model-training"
SCRIPT_PATH="/home/tamoghno/rohit-encoders/scripts/train_all_models.sh"

echo "🚀 Starting sequential model training in tmux session: ${SESSION_NAME}"
echo ""

# Kill existing session if it exists
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "Killing existing session: ${SESSION_NAME}"
    tmux kill-session -t ${SESSION_NAME}
fi

# Create new tmux session and run the training script
echo "Creating new tmux session..."
tmux new-session -d -s ${SESSION_NAME} -x 250 -y 50

# Send commands to tmux
tmux send-keys -t ${SESSION_NAME} "bash ${SCRIPT_PATH}" Enter

echo "✓ Training session started in background"
echo ""
echo "To monitor training:"
echo "  tmux attach-session -t ${SESSION_NAME}"
echo ""
echo "To detach (keeping session running):"
echo "  Press Ctrl+B then D"
echo ""
echo "To view session without attaching:"
echo "  tmux capture-pane -t ${SESSION_NAME} -p"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t ${SESSION_NAME}"
