#!/data/data/com.termux/files/usr/bin/bash

echo "============================================="
echo "  BGGG Vocoder V5.0 - Non-Interactive Setup"
echo "============================================="

sleep 1

########################################
# 0. Set the repo name you want on GitHub
########################################
REPO_NAME="BGGG_Vocoder"
echo "[INFO] Using repo name: $REPO_NAME"

sleep 1

########################################
# 1. Get logged-in GitHub username
########################################
echo "[INFO] Detecting logged-in GitHub username..."
GH_USERNAME=$(gh api user --jq .login)
echo "[OK] GitHub username detected: $GH_USERNAME"

sleep 1

########################################
# 2. Define all required folders
########################################
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
echo "[INFO] Creating folders and adding .gitkeep..."
for dir in "${FOLDERS[@]}"; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
  echo "  [OK] $dir/.gitkeep"
done

sleep 1

########################################
# 3. Write .gitignore
########################################
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
echo "[OK] .gitignore created"

sleep 1

########################################
# 4. Write manifest.txt
########################################
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
  - ultimate_push_setup.sh
  - main.py
EOL
echo "[OK] manifest.txt created"

sleep 1

########################################
# 5. Write default config/config.json if missing
########################################
echo ""
if [ ! -f "config/config.json" ]; then
  echo "[INFO] Writing default config/config.json..."
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
else
  echo "[SKIP] config/config.json already exists."
fi

sleep 1

########################################
# 6. Initialize Git repo if needed
########################################
echo ""
if [ ! -d ".git" ]; then
  echo "[INFO] Initializing Git repository..."
  git init
  echo "[OK] Git repo initialized."
else
  echo "[OK] Git repo already exists."
fi

sleep 1

########################################
# 7. Ensure branch is 'main'
########################################
echo ""
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
  echo "[INFO] Renaming branch to 'main'..."
  git branch -M main
  echo "[OK] Now on 'main'"
else
  echo "[OK] Already on 'main' branch."
fi

sleep 1

########################################
# 8. Configure global user if missing
########################################
echo ""
NAME=$(git config user.name)
EMAIL=$(git config user.email)
if [ -z "$NAME" ] || [ -z "$EMAIL" ]; then
  echo "[INFO] Setting global Git user.name and user.email..."
  git config --global user.name "BGGremlin Group"
  git config --global user.email "bggremlin@example.com"
  echo "[OK] Git identity set."
else
  echo "[OK] Git identity already set: $NAME <$EMAIL>"
fi

sleep 1

########################################
# 9. Link git to gh credentials for no prompts ever
########################################
echo ""
echo "[INFO] Linking git to gh auth for seamless pushing..."
git config --global credential.helper '!gh auth git-credential'
echo "[OK] Git will now use gh login automatically."

sleep 1

########################################
# 10. Stage and commit
########################################
echo ""
echo "[INFO] Staging all files..."
git add .

echo "[INFO] Making initial commit..."
git commit -m "Initial commit with automated setup" || echo "[OK] Nothing new to commit."

sleep 1

########################################
# 11. Create GitHub repo with gh (private)
########################################
echo ""
if gh repo view "$GH_USERNAME/$REPO_NAME" &> /dev/null; then
  echo "[OK] Repo already exists on GitHub."
else
  echo "[INFO] Creating PRIVATE repo on GitHub with gh..."
  gh repo create "$GH_USERNAME/$REPO_NAME" --private --source=. --push --remote=origin --branch=main
  echo "[OK] Repo created and pushed."
fi

sleep 1

########################################
# 12. Final Push
########################################
echo ""
echo "[INFO] Pushing to GitHub..."
git push -u origin main
echo "[OK] Pushed to GitHub."

sleep 1

########################################
# 13. Complete
########################################
echo ""
echo "============================================="
echo "✅ BGGG Vocoder is Fully Set Up and Pushed!"
echo "✅ Remote: https://github.com/$GH_USERNAME/$REPO_NAME"
echo "✅ Branch: main"
echo "✅ Repo is PRIVATE on GitHub"
echo "============================================="
