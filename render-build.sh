#!/usr/bin/env bash
set -e  # exit on first error

echo "🔧 Installing ffmpeg..."
apt-get update -y
apt-get install -y ffmpeg

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build complete."
