#!/data/data/com.termux/files/usr/bin/bash

echo "============================================="
echo "  BGGG Vocoder V5.0 - Full Setup Script"
echo "  (BGGremlin Group recommended structure)"
echo "============================================="

sleep 1

#################################
# 1. Define all folders
#################################
FOLDERS=(
  "assets"
  "themes"
  "voice_models"
  "recordings"
  "presets"
  "config"
  "docs"
)

echo ""
echo "[INFO] Creating folder structure..."
for dir in "${FOLDERS[@]}"; do
  mkdir -p "$dir"
  echo "  [OK] $dir/"
done

sleep 1

#################################
# 2. Add .gitkeep to all folders
#################################
echo ""
echo "[INFO] Adding .gitkeep files..."
for dir in "${FOLDERS[@]}"; do
  touch "$dir/.gitkeep"
  echo "  [OK] $dir/.gitkeep created"
done

sleep 1

#################################
# 3. Write .gitignore with smart rules
#################################
echo ""
echo "[INFO] Writing .gitignore..."

cat > .gitignore <<EOL
# Python
__pycache__/
*.pyc

# Virtual environments
venv/
env/

# User audio
recordings/
voice_models/trained_voice_model.pt
assets/shape_predictor_68_face_landmarks.dat*

# Keep folders but ignore contents except .gitkeep
assets/*
!assets/.gitkeep

themes/*
!themes/.gitkeep

voice_models/*
!voice_models/.gitkeep

recordings/*
!recordings/.gitkeep

presets/*
!presets/.gitkeep

config/*
!config/.gitkeep

docs/*
!docs/.gitkeep

# System
.DS_Store
Thumbs.db
EOL

echo "  [OK] .gitignore created"

sleep 1

#################################
# 4. Write a recommended manifest.txt
#################################
echo ""
echo "[INFO] Writing manifest.txt..."

cat > manifest.txt <<EOL
# BGGG Vocoder V5.0 - Project Manifest

assets/
  - logo_screen.png
  - credits_screen.png
  - exit_screen.png
  - static_visualizer.png
  - shape_predictor_68_face_landmarks.dat
  - shape_predictor_68_face_landmarks.dat.fallback
  - default_model.obj
  - waifu_mascot.obj

themes/
  - web_ui.html
  - custom_theme.css

voice_models/
  - trained_voice_model.pt

recordings/
  - (your recorded .wav files here)

presets/
  - Default
  - Spooky Russian
  - FBI
  - Witness Protection

config/
  - config.json

docs/
  - user_manual.html

Root-level:
  - requirements.txt
  - LICENSE
  - README.md
  - .gitignore
  - manifest.txt
  - setup.sh / full_setup.sh
  - main.py
EOL

echo "  [OK] manifest.txt created"

sleep 1

#################################
# 5. Create default config/config.json if missing
#################################
echo ""
echo "[INFO] Checking config/config.json..."

if [ ! -f "config/config.json" ]; then
  echo "[INFO] Generating default config/config.json..."

  cat > config/config.json <<EOL
{
  "last_used_preset": "default_preset.json",
  "audio_device_input": "default",
  "audio_device_output": "default",
  "sampling_rate": 44100,
  "buffer_size": 1024,
  "gui_theme": "ParrotOS",
  "web_ui_port": 5000,
  "voice_cloning_enabled": true,
  "logging_level": "INFO"
}
EOL
  echo "  [OK] Default config/config.json created."
else
  echo "  [SKIP] config/config.json already exists."
fi

sleep 1

#################################
# 6. Optionally generate placeholder trained_voice_model.pt
#################################
echo ""
echo "[INFO] Checking for voice_models/trained_voice_model.pt..."

if [ ! -f "voice_models/trained_voice_model.pt" ]; then
  echo "[INFO] Attempting to generate placeholder .pt using Python..."

  echo "
import torch
placeholder = {
    'model_state_dict': {},
    'speaker_embedding': None,
    'config': {'note': 'Placeholder for BGGG Vocoder V5.0'}
}
torch.save(placeholder, 'voice_models/trained_voice_model.pt')
print('✅ Placeholder trained_voice_model.pt created.')
  " > temp_make_model.py

  if command -v python > /dev/null 2>&1 || command -v python3 > /dev/null 2>&1; then
    python3 temp_make_model.py || python temp_make_model.py
    rm temp_make_model.py
    echo "  [OK] voice_models/trained_voice_model.pt created."
  else
    echo "  [WARNING] Python not found. Skipping .pt generation."
  fi
else
  echo "  [SKIP] voice_models/trained_voice_model.pt already exists."
fi

sleep 1

#################################
# 7. Final message
#################################
echo ""
echo "============================================="
echo "✅ BGGG Vocoder Setup Complete!"
echo "============================================="
echo ""
echo "[INFO] Next steps:"
echo "  git add ."
echo "  git commit -m \"Setup project structure with .gitkeep and config\""
echo "  git push"
echo ""
echo "[INFO] Repo is now fully structured and ready for production."
