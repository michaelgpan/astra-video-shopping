#!/bin/bash

# Define ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print messages in color
function print_message() {
    echo -e "${1}${2}${NC}"
}

# Check if .venv directory exists
if [ ! -d ".venv" ]; then
    print_message $GREEN "Virtual environment not found. Creating one..."
    python3 -m venv .venv --system-site-packages || { print_message $RED "Failed to create virtual environment. Exiting."; exit 1; }
    #python3 -m venv .venv || { print_message $RED "Failed to create virtual environment. Exiting."; exit 1; }

    # Activate the virtual environment
    source .venv/bin/activate || { print_message $RED "Failed to activate virtual environment. Exiting."; exit 1; }

    # Run the installation
    pip install -r "requirements.txt" || { print_message $RED "Failed to install Python packages. Exiting."; exit 1; }

    # Prompt the user
    print_message $GREEN "\nInstallation complete."
else
    source .venv/bin/activate || { print_message $RED "Failed to activate virtual environment. Exiting."; exit 1; }
fi


print_message $GREEN "All setup for video shopping\n\nsource .venv/bin/activate\n python shopping_fullscreen.py"
