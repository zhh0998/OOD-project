#!/bin/bash
#
# Setup environment for RW1 Preliminary Experiments
#

set -e

echo "Setting up RW1 Preliminary Experiments environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p results/{h1,h2,h3,h4,h5}/figures
mkdir -p figures
mkdir -p data/{fewrel,docred,nyth}

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run all experiments, run:"
echo "  bash scripts/run_all_hypothesis_tests.sh"
