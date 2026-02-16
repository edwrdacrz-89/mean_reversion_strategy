#!/bin/bash
# Signal Generator Cron Job for Railway
# Cron: 45 16,17,19,20 * * 1-5 (UTC)
# - 16:45/17:45 UTC = 12:45 PM ET (for early close days)
# - 19:45/20:45 UTC = 3:45 PM ET (for normal days)

set -e

echo "=== Signal Generator Cron Job ==="
echo "Time (UTC): $(date -u +'%H:%M')"

# Determine current ET hour using Python (reliable cross-platform)
ET_HOUR=$(python3 -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).hour)")
ET_DATE=$(python3 -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d'))")
echo "Eastern Time hour: $ET_HOUR, Date: $ET_DATE"

# Check if this is an early close day using the Python function
IS_EARLY=$(python3 -c "
from datetime import date
from cloud_signal_generator import is_early_close_day, is_market_holiday
today = date.today()
if today.weekday() >= 5:
    print('WEEKEND')
elif is_market_holiday():
    print('HOLIDAY')
elif is_early_close_day():
    print('EARLY')
else:
    print('NORMAL')
" 2>/dev/null || echo "NORMAL")

echo "Day type: $IS_EARLY"

# Skip weekends and holidays
if [ "$IS_EARLY" = "WEEKEND" ] || [ "$IS_EARLY" = "HOLIDAY" ]; then
    echo "Not a trading day - skipping"
    exit 0
fi

# Route to correct time window
if [ "$IS_EARLY" = "EARLY" ]; then
    # Early close day: only run at 12:xx PM ET
    if [ "$ET_HOUR" != "12" ]; then
        echo "Early close day but ET hour is $ET_HOUR (need 12) - skipping"
        exit 0
    fi
    echo "Early close day - running at 12:45 PM ET"
else
    # Normal day: only run at 3:xx PM ET
    if [ "$ET_HOUR" != "15" ]; then
        echo "Normal day but ET hour is $ET_HOUR (need 15) - skipping"
        exit 0
    fi
    echo "Normal day - running at 3:45 PM ET"
fi

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
