#!/data/data/com.termux/files/usr/bin/bash
set -e

echo
echo "======================================================"
echo "  BGGremlin Group - ULTIMATE ALL-FILES FORCE PUSH"
echo "======================================================"
echo "⚡️ WARNING: This will overwrite BOTH repos to match EXACTLY this local Termux folder."
echo "⚡️ Including ALL _from_* conflict files."
echo "⚡️ Including EVERYTHING in all subfolders."
echo "⚡️ Nothing deleted locally."
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
echo "[INFO] DISABLING ALL IGNORING..."
echo "(emptying .gitignore so nothing is skipped)"
echo "" > .gitignore

echo
echo "[INFO] Clearing Git index cache to fully rescan..."
git rm -r --cached . || true

echo
echo "[INFO] FORCING ADD of absolutely every file..."
git add --force -A

echo
echo "[INFO] Verifying staged files:"
git status

sleep 1

echo
echo "[INFO] Committing..."
git commit -m 'BGGremlin Group: ULTIMATE full restore push with all conflict variants' || echo "[INFO] Nothing new to commit."

sleep 1

echo
echo "[INFO] Force pushing to Repo 1 ($REPO1)..."
git push $REPO1 main --force

echo
echo "[INFO] Force pushing to Repo 2 ($REPO2)..."
git push $REPO2 main --force

echo
echo "======================================================"
echo "✅ PUSH COMPLETE"
echo "✅ Both repos now EXACTLY match this Termux folder."
echo "✅ Including all subfolders and _from_* conflict copies."
echo "✅ Check GitHub web UI to confirm all contents."
echo "======================================================"
