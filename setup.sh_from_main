#!/data/data/com.termux/files/usr/bin/bash

echo "============================================="
echo "  BGGG Vocoder V5.0 - Ultimate Setup Script"
echo "  (BGGremlin Group production-ready automation)"
echo "============================================="
sleep 1

#################################
# 1. Define folders
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
echo "[INFO] Creating folders..."
for dir in "${FOLDERS[@]}"; do
  mkdir -p "$dir"
  echo "  [OK] $dir/"
done

sleep 1

#################################
# 2. Add .gitkeep files
#################################
echo ""
echo "[INFO] Adding .gitkeep files..."
for dir in "${FOLDERS[@]}"; do
  touch "$dir/.gitkeep"
  echo "  [OK] $dir/.gitkeep created"
done

sleep 1

#################################
# 3. Write .gitignore
#################################
echo ""
echo "[INFO] Writing .gitignore..."

cat > .gitignore <<EOL
__pycache__/
*.pyc
venv/
env/
recordings/
voice_models/trained_voice_model.pt
assets/shape_predictor_68_face_landmarks.dat*

# Keep folders with .gitkeep
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

.DS_Store
Thumbs.db
EOL

echo "  [OK] .gitignore written"

sleep 1

#################################
# 4. Write manifest.txt
#################################
echo ""
echo "[INFO] Writing manifest.txt..."

cat > manifest.txt <<EOL
# BGGG Vocoder V5.0 - Manifest

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
  - ultimate_setup.sh
  - main.py
EOL

echo "  [OK] manifest.txt written"

sleep 1

#################################
# 5. Create default config/config.json
#################################
echo ""
if [ ! -f "config/config.json" ]; then
  echo "[INFO] Creating default config/config.json..."
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
  echo "  [OK] config/config.json created."
else
  echo "  [SKIP] config/config.json already exists."
fi

sleep 1

#################################
# 6. Optional dummy trained_voice_model.pt
#################################
echo ""
echo "[INFO] Checking for voice_models/trained_voice_model.pt..."

if [ -f "voice_models/trained_voice_model.pt" ]; then
  echo "  [SKIP] Placeholder model already exists."
else
  echo "  [INFO] Attempting to generate placeholder with Python..."
  echo "
try:
    import torch
    placeholder = {
        'model_state_dict': {},
        'speaker_embedding': None,
        'config': {'note': 'Placeholder for BGGG Vocoder V5.0'}
    }
    torch.save(placeholder, 'voice_models/trained_voice_model.pt')
    print('✅ Placeholder trained_voice_model.pt created.')
except ImportError:
    print('⚠️ PyTorch not installed. Skipping .pt generation.')
  " > temp_make_model.py

  (python3 temp_make_model.py || python temp_make_model.py)
  rm temp_make_model.py
fi

sleep 1

#################################
# 7. Git Init if needed
#################################
echo ""
if [ -d ".git" ]; then
  echo "[OK] Git repo already initialized."
else
  echo "[INFO] Initializing new git repo..."
  git init
  echo "  [OK] Git repo initialized."
fi

sleep 1

#################################
# 8. Git user config
#################################
echo ""
NAME=$(git config user.name)
EMAIL=$(git config user.email)

if [ -z "$NAME" ] || [ -z "$EMAIL" ]; then
  echo "[INFO] Setting global git user.name and user.email..."
  git config --global user.name "BGGremlin Group"
  git config --global user.email "bggremlin@example.com"
  echo "  [OK] Git identity set."
else
  echo "[OK] Git identity already set: $NAME <$EMAIL>"
fi

sleep 1

#################################
# 9. Initial Commit
#################################
echo ""
echo "[INFO] Staging all files..."
git add .

echo "[INFO] Making initial commit..."
git commit -m "Initial commit with full automated setup" || echo "⚠️ Nothing new to commit."

sleep 1

#################################
# 10. Optionally set remote
#################################
echo ""
if git remote | grep origin > /dev/null; then
  echo "[OK] Remote origin already set."
else
  echo "---------------------------------------------"
  echo "  🔥 REMOTE NOT SET!"
  echo "  👉 To add your GitHub repo, run:"
  echo "     git remote add origin YOUR_REPO_URL"
  echo "     git push -u origin main"
  echo "---------------------------------------------"
fi

echo ""
echo "============================================="
echo "✅ BGGG Vocoder Setup Complete!"
echo "============================================="
