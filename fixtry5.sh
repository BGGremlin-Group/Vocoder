#!/data/data/com.termux/files/usr/bin/bash
set -e

##############################################################
# BGGremlin Group - ULTIMATE FORCE SYNC SCRIPT
# Author: BGGremlin Group
# 
# PURPOSE:
#  - Takes ENTIRE local folder as-is (including subfolders and _from_* conflict files)
#  - Commits all files
#  - Force pushes to BOTH GitHub repos
#  - NO local deletions, NO filtering
##############################################################

echo
echo "==================================================="
echo "  BGGremlin Group - ULTIMATE FORCE SYNC SCRIPT"
echo "  ⚡️ This will make BOTH GitHub repos match this local folder EXACTLY."
echo "  ⚡️ Includes ALL files, all subfolders, conflict versions."
echo "  ⚡️ NO local deletions. Just commit + force-push."
echo "==================================================="

################################
# Settings
################################
REPO1_URL="https://github.com/BGGremlin-Group/Vocoder.git"
REPO2_URL="https://github.com/BGGremlin-Group/BGGG_Vocoder.git"

echo
echo "[INFO] Repo 1: $REPO1_URL"
echo "[INFO] Repo 2: $REPO2_URL"
sleep 1

################################
# Check we're in a git repo
################################
if [ ! -d ".git" ]; then
  echo
  echo "[ERROR] This folder is NOT a git repo."
  echo "[HINT] You must run 'git init' first if needed."
  exit 1
fi

################################
# Ensure on main
################################
echo
echo "[INFO] Checking branch..."
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
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

echo "[INFO] Current remotes:"
git remote -v

################################
# Stage EVERYTHING
################################
echo
echo "[INFO] Staging EVERY file in this folder and subfolders..."
git add -A

################################
# Show git status
################################
echo
echo "---------------------------------------------------"
git status
echo "---------------------------------------------------"

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
echo "[INFO] Force pushing to Repo 1 (original)..."
git push repo1 main --force

echo
echo "[INFO] Force pushing to Repo 2 (new)..."
git push repo2 main --force

################################
# Done
################################
echo
echo "==================================================="
echo "✅ PUSH COMPLETE"
echo "✅ Both GitHub repos now match this local Termux folder EXACTLY."
echo "✅ Including _from_* conflict files and all folders."
echo "✅ Nothing deleted locally."
echo "==================================================="
