#!/data/data/com.termux/files/usr/bin/bash
set -e

##############################################################
# BGGremlin Group - ULTIMATE FORCE SYNC SCRIPT
# Author: BGGremlin Group
# Purpose:
#   - Stages EVERY local file, preserving all _from_* versions.
#   - Commits them to your local repo.
#   - Pushes to BOTH target GitHub repos on main.
#   - Never deletes local files.
##############################################################

echo
echo "==================================================="
echo "  BGGremlin Group - ULTIMATE FORCE SYNC SCRIPT"
echo "  ⚡️ Makes BOTH GitHub repos match this local folder"
echo "  ⚡️ Includes ALL _from_* conflict files"
echo "  ⚡️ NO local deletions, just commit and push"
echo "==================================================="

################################
# Settings
################################
REPO1_URL="https://github.com/BGGremlin-Group/Vocoder.git"
REPO2_URL="https://github.com/BGGremlin-Group/BGGG_Vocoder.git"

echo
echo "[INFO] Will push to:"
echo " - Repo 1: $REPO1_URL"
echo " - Repo 2: $REPO2_URL"

sleep 1

################################
# Check we're in a git repo
################################
if [ ! -d ".git" ]; then
  echo
  echo "[ERROR] This folder is not a git repository."
  echo "[HINT] Run 'git init' first if needed."
  exit 1
fi

################################
# Ensure on main
################################
echo
echo "[INFO] Ensuring we're on branch 'main'..."
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
  echo "[WARN] Not on main. Switching or creating..."
  git checkout main || git checkout -b main
fi

################################
# Add Both Remotes
################################
echo
echo "[INFO] Setting remotes..."
git remote remove repo1 2>/dev/null || true
git remote remove repo2 2>/dev/null || true
git remote add repo1 "$REPO1_URL"
git remote add repo2 "$REPO2_URL"

git remote -v

################################
# Stage EVERYTHING
################################
echo
echo "[INFO] Staging all files (including conflicts)..."
git add -A

################################
# Show Status
################################
echo
git status

################################
# Commit
################################
echo
echo "[INFO] Committing..."
git commit -m 'BGGremlin Group: FULL local restore with all conflict variants' || echo "[INFO] Nothing new to commit."

################################
# Push to BOTH repos
################################
echo
echo "[INFO] Force pushing to REPO1..."
git push repo1 main --force

echo
echo "[INFO] Force pushing to REPO2..."
git push repo2 main --force

################################
# Done
################################
echo
echo "==================================================="
echo "✅ PUSH COMPLETE"
echo "✅ Both GitHub repos now match local Termux folder."
echo "✅ Including _from_* conflict versions."
echo "✅ Nothing was deleted locally."
echo "==================================================="
