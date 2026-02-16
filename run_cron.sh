#!/bin/bash
# Signal Generator Cron Job for Railway
# Clones repo, runs signal generator, commits & pushes results

set -e

echo "=== Signal Generator Cron Job ==="
echo "Time: $(date)"

# Setup git
git config --global user.email "${GIT_EMAIL:-action@github.com}"
git config --global user.name "${GIT_USER:-Railway Cron}"

# Clone the repo into a temp directory
REPO_DIR=$(mktemp -d)
git clone "https://${GIT_TOKEN}@github.com/edwrdacrz-89/mean_reversion_strategy.git" "$REPO_DIR"
cd "$REPO_DIR"

# Run signal generator
echo "Running signal generator..."
python cloud_signal_generator.py

# Commit and push results
echo "Pushing results to GitHub..."
git add signals.json track_record.json
echo "$(date -u +'%Y-%m-%d %H:%M:%S UTC')" > last_run.txt
git add last_run.txt

if git diff --staged --quiet; then
    echo "No changes to commit"
else
    git commit -m "Daily signals $(date +'%Y-%m-%d') [Railway]"
    git push
    echo "Pushed successfully"
fi

# Cleanup
rm -rf "$REPO_DIR"
echo "=== Done ==="
