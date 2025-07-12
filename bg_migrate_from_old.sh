#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "================================================="
echo " BGGG Vocoder - MIGRATION SCRIPT"
echo " (Pulls from old repo, reorganizes, pushes)"
echo "================================================="

sleep 1

##############################
# 0. CONFIG (HARDCODED)
##############################
OLD_REPO_URL="https://github.com/BGGremlin-Group/BGGG_Vocoder.git"
TEMP_CLONE_DIR="$HOME/temp_vocoder"

echo "[INFO] Source old repo: $OLD_REPO_URL"
echo "[INFO] Temp clone folder: $TEMP_CLONE_DIR"

sleep 1

##############################
# 1. Clone old repo to temp
##############################
echo ""
if [ -d "$TEMP_CLONE_DIR" ]; then
  echo "[INFO] Removing existing temp clone..."
  rm -rf "$TEMP_CLONE_DIR"
fi

echo "[INFO] Cloning old repo..."
git clone "$OLD_REPO_URL" "$TEMP_CLONE_DIR"
echo "[OK] Clone complete."

sleep 1

##############################
# 2. Copy root-level files
##############################
echo ""
echo "[INFO] Copying root-level files..."
cp -vu $TEMP_CLONE_DIR/*.py . || true
cp -vu $TEMP_CLONE_DIR/requirements.txt . || true
cp -vu $TEMP_CLONE_DIR/LICENSE . || true
cp -vu $TEMP_CLONE_DIR/README.md . || true
cp -vu $TEMP_CLONE_DIR/main.py . || true

sleep 1

##############################
# 3. Copy folders into structured subfolders
##############################
echo ""
echo "[INFO] Syncing folders..."

function sync_folder() {
  FROM="$TEMP_CLONE_DIR/$1"
  TO="$1"
  if [ -d "$FROM" ]; then
    echo "  [OK] Copying $FROM --> $TO"
    cp -ruv "$FROM/"* "$TO/" || echo "    [SKIP] Nothing to copy in $FROM"
  else
    echo "  [SKIP] $FROM does not exist."
  fi
}

sync_folder assets
sync_folder themes
sync_folder voice_models
sync_folder recordings
sync_folder presets
sync_folder config
sync_folder docs

sleep 1

##############################
# 4. Cleanup temp clone
##############################
echo ""
echo "[INFO] Removing temp clone..."
rm -rf "$TEMP_CLONE_DIR"
echo "[OK] Temp clone removed."

sleep 1

##############################
# 5. Stage, Commit, Push
##############################
echo ""
echo "[INFO] Staging all changes..."
git add .

echo "[INFO] Committing..."
git commit -m "Migrated content from old repo into new structured layout" || echo "[OK] Nothing new to commit."

echo "[INFO] Pushing to GitHub..."
git push -u origin main --force
echo "[OK] Pushed to GitHub."

echo ""
echo "================================================="
echo "✅ Migration Complete! Repo is up-to-date."
echo "================================================="
