#!/data/data/com.termux/files/usr/bin/bash

echo "============================================="
echo "  BGGG Vocoder - Migration Script"
echo "  (Copy files from ~/temp_vocoder to structured folders)"
echo "============================================="
sleep 1

SRC=~/temp_vocoder

if [ ! -d "$SRC" ]; then
  echo "[ERROR] Source folder $SRC does not exist. Did you forget to clone your old repo?"
  echo "Run: git clone https://github.com/YOUR_USERNAME/vocoder ~/temp_vocoder"
  exit 1
fi

echo "[INFO] Copying root-level files..."
cp -v $SRC/*.py . 2>/dev/null
cp -v $SRC/requirements.txt . 2>/dev/null
cp -v $SRC/LICENSE . 2>/dev/null
cp -v $SRC/README.md . 2>/dev/null

echo ""
echo "[INFO] Syncing folders..."

# Function to sync a folder safely
sync_folder() {
  FROM="$SRC/$1"
  TO="$1"

  if [ -d "$FROM" ]; then
    echo "  [OK] Copying $FROM -> $TO"
    cp -rv "$FROM/"* "$TO/" 2>/dev/null || echo "    [SKIP] Nothing to copy in $FROM"
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

########################################
# Git Add / Commit / Push
########################################
echo ""
echo "[INFO] Staging all changes..."
git add .

echo "[INFO] Committing..."
git commit -m "Migrated content from old repo into new structured layout" || echo "[OK] Nothing new to commit."

echo "[INFO] Pushing to GitHub..."
git push

echo ""
echo "============================================="
echo "✅ Migration Complete! Pushed to GitHub."
echo "============================================="
