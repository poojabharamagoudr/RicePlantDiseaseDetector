<#
Setup a Python virtual environment and install project dependencies (PowerShell).

Usage (PowerShell):
  .\setup_env.ps1

This will:
- create a `.venv` virtual environment in the repo root
- activate it for the current script
- upgrade pip and install `requirements.txt` and optional `requirements-dev.txt`
- run a quick compile check and dependency sanity check
#>

Write-Host "Creating virtual environment .venv (if not exists)"
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "Activating virtual environment"
# Activation for the current PowerShell session
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip"
python -m pip install --upgrade pip

Write-Host "Installing required packages from requirements.txt"
python -m pip install -r requirements.txt

if (Test-Path "requirements-dev.txt") {
    Write-Host "Installing development requirements from requirements-dev.txt"
    python -m pip install -r requirements-dev.txt
}

Write-Host "Running quick compile check (python -m compileall .)"
python -m compileall .

Write-Host "Running dependency sanity check"
python .\backend\check_deps.py

Write-Host "Setup finished. To activate the venv in a new session run:`n .\\.venv\\Scripts\\Activate.ps1"
