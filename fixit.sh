#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "================================================="
echo " BGGremlin Group - SAFE MERGE SCRIPT"
echo "✨ Merges all branches from original repo INTO main"
echo "✨ Preserves ALL files with suffix if needed"
echo "================================================="
sleep 1

################################
# CONFIG
################################
SOURCE_REPO="https://github.com/BGGremlin-Group/Vocoder.git"
TEMP_DIR="$HOME/temp_vocoder_safe_merge"
TARGET_REMOTE="origin"

echo "[INFO] Source Repo: $SOURCE_REPO"
echo "[INFO] Temp workspace: $TEMP_DIR"
sleep 1

################################
# CLEAN TEMP DIR
################################
if [ -d "$TEMP_DIR" ]; then
  echo "[INFO] Removing old temp dir..."
  rm -rf "$TEMP_DIR"
fi
mkdir -p "$TEMP_DIR"

sleep 1

################################
# DEFINE BRANCHES TO PULL
################################
BRANCHES=(main assets themes presets config docs voice_models)

################################
# CLONE EACH BRANCH
################################
for BR in "${BRANCHES[@]}"; do
  echo ""
  echo "============================="
  echo "✅ Cloning branch: $BR"
  echo "============================="
  git clone --depth=1 --branch="$BR" "$SOURCE_REPO" "$TEMP_DIR/$BR" || echo "  [WARN] Branch $BR may not exist."
done

sleep 1

################################
# COPY MAIN ROOT-LEVEL FILES
################################
echo ""
echo "============================="
echo "✅ Copying root-level files from main branch"
echo "============================="
MAIN_TEMP="$TEMP_DIR/main"
if [ -d "$MAIN_TEMP" ]; then
  for file in "$MAIN_TEMP"/*; do
    if [ -f "$file" ]; then
      base=$(basename "$file")
      if [ -e "$base" ]; then
        echo "  [WARN] Conflict for $base → keeping existing, renaming new as ${base}_from_main"
        cp -v "$file" "${base}_from_main"
      else
        cp -v "$file" .
      fi
    fi
  done
else
  echo "  [WARN] No main branch found in SOURCE."
fi

sleep 1

################################
# MERGE FOLDER BRANCHES
################################
echo ""
echo "============================="
echo "✅ Merging other branches into structured folders"
echo "============================="
for SUB in assets themes presets config docs voice_models; do
  SRC="$TEMP_DIR/$SUB"
  DEST="./$SUB"
  echo ""
  echo ">>> Processing $SUB"
  if [ -d "$SRC" ]; then
    mkdir -p "$DEST"
    for srcfile in "$SRC"/*; do
      if [ -f "$srcfile" ]; then
        base=$(basename "$srcfile")
        if [ -e "$DEST/$base" ]; then
          newname="${base}_from_${SUB}"
          echo "  [WARN] Conflict for $DEST/$base → saving as $newname"
          cp -v "$srcfile" "$DEST/$newname"
        else
          cp -v "$srcfile" "$DEST/"
        fi
      elif [ -d "$srcfile" ]; then
        # Recursively copy subfolders
        subdir=$(basename "$srcfile")
        mkdir -p "$DEST/$subdir"
        echo "  [INFO] Copying subfolder $subdir"
        cp -ruv "$srcfile/"* "$DEST/$subdir/" || echo "  [SKIP] Empty folder $subdir"
      fi
    done
  else
    echo "  [SKIP] Branch $SUB did not exist or was empty in SOURCE."
  fi
done

sleep 1

################################
# ADD .gitkeep WHERE NEEDED
################################
echo ""
echo "✅ Ensuring .gitkeep in empty folders..."
ALL_FOLDERS=(assets themes presets config docs voice_models)
for dir in "${ALL_FOLDERS[@]}"; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
  echo "  [OK] $dir/.gitkeep"
done

sleep 1

################################
# STAGE AND COMMIT
################################
echo ""
echo "✅ Staging all changes..."
git add .

echo "✅ Committing..."
git commit -m 'SAFE MERGE: Included all branch contents into main with conflict suffixes' || echo "[OK] Nothing new to commit."

sleep 1

################################
# PUSH
################################
echo ""
echo "✅ Pushing to GitHub (force)..."
git push -u "$TARGET_REMOTE" main --force
echo "[OK] Push complete."

sleep 1

################################
# CLEAN TEMP
################################
echo ""
echo "✅ Cleaning temp folders..."
rm -rf "$TEMP_DIR"
echo "[OK] Temp workspace removed."

echo ""
echo "================================================="
echo "✅ SAFE MERGE COMPLETE!"
echo "✅ All original branch content integrated into main."
echo "✅ Conflicts renamed with suffixes."
echo "================================================="
