#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "================================================="
echo " BGGG Vocoder - FIX ORIGINAL REPO IN-PLACE"
echo " (Combines all branches into single main)"
echo "================================================="

sleep 1

################################
# CONFIG - Hardcoded
################################
REMOTE_URL="https://github.com/BGGremlin-Group/Vocoder.git"
TEMP_DIR="$HOME/temp_vocoder_fix"

echo "[INFO] Target Repo: $REMOTE_URL"
echo "[INFO] Temp workspace: $TEMP_DIR"

sleep 1

################################
# Clean temp dir
################################
echo ""
if [ -d "$TEMP_DIR" ]; then
  echo "[INFO] Removing old temp dir..."
  rm -rf "$TEMP_DIR"
fi
mkdir -p "$TEMP_DIR"

sleep 1

################################
# Define branches to pull
################################
BRANCHES=(main assets themes presets config docs voice_models)

################################
# Clone each branch one by one
################################
for BR in "${BRANCHES[@]}"; do
  echo ""
  echo "============================="
  echo "[INFO] Cloning branch: $BR"
  echo "============================="
  git clone --depth=1 --branch="$BR" "$REMOTE_URL" "$TEMP_DIR/$BR" || echo "  [WARN] Branch $BR may not exist."
done

sleep 1

################################
# Make sure we're on main
################################
echo ""
echo "[INFO] Ensuring local repo is on 'main' branch..."
git checkout main 2>/dev/null || git checkout -b main
echo "[OK] On branch 'main'."

sleep 1

################################
# Clean working directory
################################
echo ""
echo "[INFO] Removing existing folder structure..."
rm -rf assets themes presets config docs voice_models
rm -f *.py requirements.txt LICENSE README.md main.py || true
echo "[OK] Old files/folders removed."

sleep 1

################################
# Copy root-level files from SOURCE main
################################
echo ""
echo "[INFO] Copying root-level files from SOURCE main branch..."
MAIN_TEMP="$TEMP_DIR/main"
if [ -d "$MAIN_TEMP" ]; then
  cp -vu $MAIN_TEMP/*.py . || true
  cp -vu $MAIN_TEMP/requirements.txt . || true
  cp -vu $MAIN_TEMP/LICENSE . || true
  cp -vu $MAIN_TEMP/README.md . || true
  cp -vu $MAIN_TEMP/main.py . || true
else
  echo "[WARN] No main branch content found!"
fi

sleep 1

################################
# Merge contents from folder branches
################################
echo ""
echo "[INFO] Merging other branches into proper folders..."
for SUB in assets themes presets config docs voice_models; do
  SRC="$TEMP_DIR/$SUB"
  DEST="./$SUB"
  echo ""
  echo ">>> Processing $SUB"
  if [ -d "$SRC" ]; then
    mkdir -p "$DEST"
    cp -ruv "$SRC/"* "$DEST/" || echo "  [SKIP] Nothing to copy in $SUB branch."
  else
    echo "  [SKIP] Branch $SUB did not exist or was empty."
  fi
done

sleep 1

################################
# Ensure .gitkeep in empty folders
################################
echo ""
echo "[INFO] Adding .gitkeep to ensure empty folders tracked..."
ALL_FOLDERS=(assets themes presets config docs voice_models)
for dir in "${ALL_FOLDERS[@]}"; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
  echo "  [OK] $dir/.gitkeep"
done

sleep 1

################################
# Clean temp workspace
################################
echo ""
echo "[INFO] Cleaning up temp folders..."
rm -rf "$TEMP_DIR"
echo "[OK] Temp workspace removed."

sleep 1

################################
# Stage, Commit, Force Push
################################
echo ""
echo "[INFO] Staging all changes..."
git add .

echo "[INFO] Committing..."
git commit -m "Restructured repo: merged all branches into single main" || echo "[OK] Nothing new to commit."

echo "[INFO] Pushing forcefully to overwrite main..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
git push -u origin main --force
echo "[OK] Push complete. Repo is fixed."

echo ""
echo "================================================="
echo "✅ Original Vocoder Repo Fixed!"
echo "✅ All branches merged into single main."
echo "✅ Repo: $REMOTE_URL"
echo "================================================="
