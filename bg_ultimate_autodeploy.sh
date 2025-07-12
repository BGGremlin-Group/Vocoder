#!/data/data/com.termux/files/usr/bin/bash

echo "============================================="
echo "  BGGG Vocoder - ULTIMATE AUTO DEPLOY SCRIPT"
echo "  (Fully automated. No prompts. Production ready.)"
echo "============================================="

sleep 1

# =============================
# CONFIG - HARD WIRED
# =============================
GH_USER="BGGremlin-Group"
REPO_NAME="BGGG_Vocoder"
REMOTE_URL="https://github.com/$GH_USER/$REPO_NAME.git"

echo "[INFO] Using GitHub Repo: $GH_USER/$REPO_NAME"

############################################
# 1. Create All Folders + .gitkeep
############################################
FOLDERS=(assets themes voice_models recordings presets config docs)

echo ""
echo "[INFO] Ensuring all project folders with .gitkeep..."

for dir in "${FOLDERS[@]}"; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
  echo "  [OK] $dir/.gitkeep"
done

sleep 1

############################################
# 2. Add .gitignore
############################################
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

echo "[OK] .gitignore created"

sleep 1

############################################
# 3. Add manifest.txt
############################################
echo ""
echo "[INFO] Writing manifest.txt..."
cat > manifest.txt <<EOL
# BGGG Vocoder V5.0 - Project Manifest

assets/
themes/
voice_models/
recordings/
presets/
config/
docs/
requirements.txt
LICENSE
README.md
.gitignore
manifest.txt
main.py
EOL

echo "[OK] manifest.txt created"

sleep 1

############################################
# 4. Add config/config.json
############################################
echo ""
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

echo "[OK] config/config.json created"

sleep 1

############################################
# 5. Initialize Git
############################################
echo ""
if [ ! -d ".git" ]; then
  echo "[INFO] Initializing new git repo..."
  git init
  echo "[OK] Git repo initialized."
else
  echo "[OK] Git repo already exists."
fi

sleep 1

############################################
# 6. Ensure main branch
############################################
echo ""
echo "[INFO] Ensuring branch is 'main'..."
git branch -M main
echo "[OK] Now on 'main' branch."

sleep 1

############################################
# 7. Set User Info
############################################
echo ""
echo "[INFO] Ensuring git user info..."
git config --global user.name "BGGremlin Group"
git config --global user.email "bggremlin@example.com"
echo "[OK] User info set."

sleep 1

############################################
# 8. Stage and Commit
############################################
echo ""
echo "[INFO] Staging all files..."
git add .

echo "[INFO] Committing..."
git commit -m "Initial commit with full automated setup" || echo "[OK] Nothing new to commit."

sleep 1

############################################
# 9. Create Remote Repo with GH CLI
############################################
echo ""
echo "[INFO] Creating repo on GitHub (PRIVATE)..."

gh repo view "$GH_USER/$REPO_NAME" &>/dev/null
if [ $? -ne 0 ]; then
  echo "[INFO] Repo doesn't exist. Creating..."
  gh repo create "$GH_USER/$REPO_NAME" --private --source=. --remote=origin --push --branch=main
  echo "[OK] Repo created and pushed."
else
  echo "[OK] Repo already exists. Setting remote..."
  git remote remove origin 2>/dev/null
  git remote add origin "$REMOTE_URL"
fi

sleep 1

############################################
# 10. Force Push Main
############################################
echo ""
echo "[INFO] Pushing to GitHub..."
git push -u origin main --force

echo ""
echo "============================================="
echo "✅ BGGG Vocoder - FULL DEPLOY COMPLETE!"
echo "✅ Remote: $REMOTE_URL"
echo "✅ Branch: main"
echo "✅ Repo is PRIVATE"
echo "============================================="
