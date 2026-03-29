#!/bin/bash
# Launch the Conjecture Harness
cd "$(dirname "$0")"

# Check for dependencies
python3 -c "import flask" 2>/dev/null || { echo "Installing dependencies..."; pip3 install -r requirements.txt; }

# Seed if no data yet
if [ ! -f data/entries.json ]; then
    echo "Seeding baseline entries..."
    python3 seed.py
fi

echo ""
echo "  Conjecture Harness"
echo "  Open http://localhost:5111 in your browser"
echo ""

python3 -m flask --app app run --port 5111
