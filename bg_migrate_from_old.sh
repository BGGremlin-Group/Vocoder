#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "================================================="
echo " BGGG Vocoder - MIGRATE FROM VOCODER TO BGGG_VOCODER"
echo " (Pulls all branches from Vocoder, pushes to BGGG_Vocoder)"
echo "================================================="

sleep 1

################################
# CONFIG
################################
SOURCE_REPO="https://github.com/BGGremlin-Group/Vocoder.git"
DEST_REPO="https://github.com/BGGremlin-Group/BGGG_Vocoder.git"
TEMP_DIR="$HOME/temp_vocoder_migration"

echo "[INFO] Source Repo: $SOURCE_REPO"
echo "[INFO] Destination Repo: $DEST_REPO"
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
# Clone each branch of SOURCE
################################
for BR in "${BRANCHES[@]}"; do
  echo ""
  echo "============================="
  echo "[INFO] Cloning branch: $BR FROM SOURCE REPO"
  echo "============================="
  git clone --depth=1 --branch="$BR" "$SOURCE_REPO" "$TEMP_DIR/$BR" || echo "  [WARN] Branch $BR may not exist."
done

sleep 1

################################
# Copy root-level files from SOURCE main
################################
echo ""
echo "[INFO] Copying root-level files from SOURCE main..."
MAIN_TEMP="$TEMP_DIR/main"
if [ -d "$MAIN_TEMP" ]; then
  cp -vu $MAIN_TEMP/*.py . || true
  cp -vu $MAIN_TEMP/requirements.txt . || true
  cp -vu $MAIN_TEMP/LICENSE . || true
  cp -vu $MAIN_TEMP/README.md . || true
  cp -vu $MAIN_TEMP/main.py . || true
else
  echo "[WARN] No main branch content found in SOURCE!"
fi

sleep 1

################################
# Copy folder branches into DEST structure
################################
echo ""
echo "[INFO] Copying branch folders into DESTINATION structure..."
for SUB in assets themes presets config docs voice_models; do
  SRC="$TEMP_DIR/$SUB"
  DEST="./$SUB"
  echo ""
  echo ">>> Processing $SUB"
  if [ -d "$SRC" ]; then
    mkdir -p "$DEST"
    cp -ruv "$SRC/"* "$DEST/" || echo "  [SKIP] Nothing to copy in $SUB branch."
  else
    echo "  [SKIP] Branch $SUB did not exist or was empty in SOURCE."
  fi
done

sleep 1

################################
# Ensure .gitkeep in all folders
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
# Clean up temp workspace
################################
echo ""
echo "[INFO] Cleaning up temp folders..."
rm -rf "$TEMP_DIR"
echo "[OK] Temp workspace removed."

sleep 1

################################
# Stage, Commit, Push to DESTINATION
################################
echo ""
echo "[INFO] Staging all changes..."
git add .

echo "[INFO] Committing..."
git commit -m "Migrated content from original Vocoder repo" || echo "[OK] Nothing new to commit."

echo "[INFO] Pushing to BGGG_Vocoder (DESTINATION)..."
git remote remove origin 2>/dev/null || true
git remote add origin "$DEST_REPO"
git push -u origin main --force
echo "[OK] Push complete to BGGG_Vocoder."

echo ""
echo "================================================="
echo "✅ Migration Complete!"
echo "✅ SOURCE: $SOURCE_REPO"
echo "✅ DESTINATION: $DEST_REPO"
echo "✅ All branches merged into single main."
echo "================================================="
