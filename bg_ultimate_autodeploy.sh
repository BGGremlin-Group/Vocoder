#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "================================================="
echo " BGGG Vocoder - TOTAL AUTODEPLOY SCRIPT"
echo " (No prompts. No bullshit. All-in-one.)"
echo "================================================="

sleep 1

##############################
# 0. CONFIG (HARD CODED)
##############################
GH_USER="BGGremlin-Group"
REPO_NAME="BGGG_Vocoder"
REMOTE_URL="https://github.com/$GH_USER/$REPO_NAME.git"

echo "[INFO] Target GitHub Repo: $GH_USER/$REPO_NAME"

sleep 1

##############################
# 1. Ensure folders + .gitkeep
##############################
FOLDERS=(assets themes voice_models recordings presets config docs)

echo ""
echo "[INFO] Creating folders and .gitkeep..."
for dir in "${FOLDERS[@]}"; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
  echo "  [OK] $dir/.gitkeep"
done

sleep 1

##############################
# 2. Add .gitignore
##############################
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

##############################
# 3. Add manifest.txt
##############################
echo ""
echo "[INFO] Writing manifest.txt..."
cat > manifest.txt <<EOL
# BGGG Vocoder V5.0 - Manifest

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

##############################
# 4. Add config/config.json
##############################
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

##############################
# 5. Initialize git if needed
##############################
echo ""
if [ ! -d ".git" ]; then
  echo "[INFO] Initializing git repo..."
  git init
  echo "[OK] Git initialized."
fi

sleep 1

##############################
# 6. Ensure branch is main
##############################
echo ""
echo "[INFO] Ensuring branch is 'main'..."
git branch -M main
echo "[OK] On branch 'main'"

sleep 1

##############################
# 7. Set user.name and user.email
##############################
echo ""
echo "[INFO] Configuring global Git identity..."
git config --global user.name "BGGremlin Group"
git config --global user.email "bggremlin@example.com"
echo "[OK] Git user.name and user.email set"

sleep 1

##############################
# 8. Stage and commit
##############################
echo ""
echo "[INFO] Staging all files..."
git add .

echo "[INFO] Committing..."
git commit -m "Automated initial deploy" || echo "[OK] Nothing new to commit."

sleep 1

##############################
# 9. Create repo if missing
##############################
echo ""
echo "[INFO] Checking if GitHub repo exists..."
if gh repo view "$GH_USER/$REPO_NAME" &>/dev/null; then
  echo "[OK] Repo exists on GitHub."
else
  echo "[INFO] Repo does not exist. Creating..."
  gh repo create "$GH_USER/$REPO_NAME" --private --source=. --remote=origin --push
  echo "[OK] Repo created on GitHub."
fi

sleep 1

##############################
# 10. Ensure correct remote
##############################
echo ""
echo "[INFO] Setting remote origin to $REMOTE_URL..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
echo "[OK] Remote origin set"

sleep 1

##############################
# 11. Force Push
##############################
echo ""
echo "[INFO] Pushing to GitHub..."
git push -u origin main --force
echo "[OK] Pushed to GitHub"

echo ""
echo "================================================="
echo "✅ BGGG Vocoder - FULL AUTO DEPLOY COMPLETE!"
echo "✅ Repo: $REMOTE_URL"
echo "✅ Branch: main"
echo "✅ Repo is PRIVATE on GitHub"
echo "================================================="
