#!/bin/bash

# OccolusAI Setup Script
# This script automates the setup process for the OccolusAI drug discovery platform

set -e

echo "🧬 Welcome to OccolusAI Setup!"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    print_warning "Detected Windows. Some commands may need adjustment."
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python
if command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python not found. Please install Python 3.11+"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION found"
else
    print_error "Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_success "npm $NPM_VERSION found"
else
    print_error "npm not found. Please install npm"
    exit 1
fi

# Check Git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    print_success "Git $GIT_VERSION found"
else
    print_error "Git not found. Please install Git"
    exit 1
fi

echo ""
print_status "Setting up backend..."

# Backend setup
cd server

# Create virtual environment
print_status "Creating Python virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
$PIP_CMD install --upgrade pip
$PIP_CMD install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file..."
    cat > .env << EOF
# Google Gemini API Key for AI-powered insights
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
HOST=localhost
PORT=8000
DEBUG=True

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Model Configuration
MODEL_PATH=drug_target_model.pth
DRUG_DB_PATH=drug_db.csv
EOF
    print_warning "Please edit server/.env and add your GEMINI_API_KEY"
fi

# Deactivate virtual environment
deactivate

cd ..

echo ""
print_status "Setting up frontend..."

# Frontend setup
cd client

# Install Node.js dependencies
print_status "Installing Node.js dependencies..."
npm install

# Create .env.local file if it doesn't exist
if [ ! -f .env.local ]; then
    print_status "Creating .env.local file..."
    cat > .env.local << EOF
# API Configuration
# Backend server URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Development Configuration
NODE_ENV=development
EOF
fi

cd ..

echo ""
print_success "Setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Edit server/.env and add your GEMINI_API_KEY"
echo "2. Start the backend: cd server && source venv/bin/activate && uvicorn main:app --reload"
echo "3. Start the frontend: cd client && npm run dev"
echo "4. Open http://localhost:3000 in your browser"
echo ""
echo "🔗 Useful Links:"
echo "================"
echo "• Frontend: http://localhost:3000"
echo "• Backend API: http://localhost:8000"
echo "• API Documentation: http://localhost:8000/docs"
echo "• Gemini API Key: https://makersuite.google.com/app/apikey"
echo ""
print_success "Happy drug discovering! 🧬"