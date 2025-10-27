#!/usr/bin/env bash
set -e  # exit on first error

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎬 Installing ffmpeg (via imageio-ffmpeg)..."
pip install imageio-ffmpeg

echo "✅ Build complete."
