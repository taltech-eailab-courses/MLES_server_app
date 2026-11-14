#!/bin/sh
# Fetch the robot-planner model (if configured), then run the container's command.
set -e
python3 /app/scripts/fetch_model.py
exec "$@"
