#!/data/data/com.termux/files/usr/bin/bash
set -e

##########################################################
# BGGremlin Group - SAFE REBUILD EXPORT SCRIPT
# Purpose: Reorganize local "dirty" Termux folder
#          into a clean repo-ready structure
#          WITHOUT DELETING ANYTHING
# Author: BGGremlin Group
# License: MIT
##########################################################

echo
echo "================================================="
echo "  BGGremlin Group - SAFE REBUILD EXPORT SCRIPT"
echo "  Creates a new folder with correct repo layout"
echo "  PRESERVES original files untouched"
echo "================================================="
echo

################################
# CONFIG
################################
SOURCE_DIR="$PWD"
TARGET_DIR="$HOME/bggg_vocoder_rebuild"

echo "[INFO] Source Dir: $SOURCE_DIR"
echo "[INFO] Target Dir: $TARGET_DIR"
echo

################################
# Confirm safe location
################################
if [ "$SOURCE_DIR" == "$TARGET_DIR" ]; then
  echo "[ERROR] Source and Target folders must be different."
  exit 1
fi

################################
# Create target directory
################################
echo "[INFO] Creating target folder..."
mkdir -p "$TARGET_DIR"

################################
# FOLDER STRUCTURE
################################
FOLDERS=("assets" "themes" "presets" "voice_models" "docs" "config" "recordings")

for folder in "${FOLDERS[@]}"; do
  echo "[INFO] Ensuring folder: $TARGET_DIR/$folder"
  mkdir -p "$TARGET_DIR/$folder"
done

################################
# COPY ROOT-LEVEL FILES
################################
echo
echo "============================="
echo "✅ Copying root-level files"
echo "============================="
echo

for f in *; do
  if [ -f "$f" ]; then
    echo "[COPY] $f --> $TARGET_DIR/$f"
    cp -n "$f" "$TARGET_DIR/$f" 2>/dev/null || cp "$f" "$TARGET_DIR/${f}_from_conflict"
  fi
done

################################
# COPY SUBFOLDER CONTENTS
################################
echo
echo "============================="
echo "✅ Copying folders into structure"
echo "============================="
echo

for folder in "${FOLDERS[@]}"; do
  if [ -d "$SOURCE_DIR/$folder" ]; then
    echo
    echo ">>> Processing $folder"
    echo
    find "$SOURCE_DIR/$folder" -type f | while read -r filepath; do
      filename=$(basename "$filepath")
      targetpath="$TARGET_DIR/$folder/$filename"

      if [ -e "$targetpath" ]; then
        # Conflict
        suffix_from=$(echo "$filepath" | sed "s|$SOURCE_DIR/||" | tr / _)
        newname="${filename}_from_${suffix_from}"
        echo "  [WARN] Conflict for $filename → saving as $newname"
        cp "$filepath" "$TARGET_DIR/$folder/$newname"
      else
        echo "  [COPY] $filename"
        cp "$filepath" "$TARGET_DIR/$folder/"
      fi
    done
  else
    echo "[INFO] Source folder $folder does not exist. Skipping."
  fi
done

################################
# ADD .gitkeep to empty folders
################################
echo
echo "✅ Ensuring .gitkeep in empty folders..."
for folder in "${FOLDERS[@]}"; do
  if [ -z "$(ls -A "$TARGET_DIR/$folder" 2>/dev/null)" ]; then
    echo "  [ADD] $folder/.gitkeep"
    touch "$TARGET_DIR/$folder/.gitkeep"
  fi
done

################################
# DONE
################################
echo
echo "================================================="
echo "✅ SAFE EXPORT COMPLETE!"
echo "✅ All files and folders copied to:"
echo "   $TARGET_DIR"
echo
echo "✅ Nothing deleted. Original source is untouched."
echo "✅ Review and git init / commit manually."
echo "================================================="
echo

exit 0
