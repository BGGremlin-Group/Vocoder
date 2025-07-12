#!/data/data/com.termux/files/usr/bin/bash
set -e

echo
echo "======================================================"
echo "  BGGremlin Group - CLEAN PUSH SCRIPT"
echo "======================================================"
echo "✅ Adds only real files"
echo "✅ Excludes all _from_* conflict copies"
echo "✅ Pushes to both repos"
echo "======================================================"

sleep 1

############################################
# CONFIG
############################################
REPO1="origin"
REPO2="repo2"

echo
echo "[INFO] Checking remotes..."
git remote -v

echo
echo "[INFO] Clearing index cache to fully rescan..."
git rm -r --cached . || true

echo
echo "[INFO] Adding only real files (excluding *_from_*)..."
find . -type f ! -name '*_from_*' -print0 | xargs -0 git add --force

echo
echo "[INFO] Verifying staged files:"
git status

sleep 1

echo
echo "[INFO] Committing..."
git commit -m 'BGGremlin Group: Clean push with only needed files' || echo "[INFO] Nothing new to commit."

sleep 1

echo
echo "[INFO] Force pushing to Repo 1 ($REPO1)..."
git push $REPO1 main --force

echo
echo "[INFO] Force pushing to Repo 2 ($REPO2)..."
git push $REPO2 main --force

echo
echo "======================================================"
echo "✅ CLEAN PUSH COMPLETE"
echo "✅ Both repos now match local Termux, minus _from_* copies"
echo "======================================================"
