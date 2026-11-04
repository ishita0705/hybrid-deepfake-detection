# Push this project to GitHub

The repo is already initialized and committed locally on `main`.
GitHub CLI is at `%LOCALAPPDATA%\gh-cli\bin\gh.exe` (not logged in yet).

## Option A — GitHub website + git (simplest)

1. Open https://github.com/new  
2. Create a repository (example name: `hybrid-deepfake-detection`).  
   - **Do not** add a README/license on GitHub (you already have commits locally).  
3. In PowerShell:

```powershell
cd "C:\Users\itl\Desktop\DL PROJECT"
git remote add origin https://github.com/YOUR_USERNAME/hybrid-deepfake-detection.git
git branch -M main
git push -u origin main
```

Sign in when Windows / Git Credential Manager asks.

## Option B — GitHub CLI

```powershell
cd "C:\Users\itl\Desktop\DL PROJECT"
& "$env:LOCALAPPDATA\gh-cli\bin\gh.exe" auth login
# choose GitHub.com → HTTPS → Login with a web browser

& "$env:LOCALAPPDATA\gh-cli\bin\gh.exe" repo create hybrid-deepfake-detection --public --source=. --remote=origin --push
```

## What is NOT uploaded (by design)

Videos, checkpoints, HF cache, face caches, and `user_media` uploads stay local (see `.gitignore`).  
Docs under `docs/` include the report, viva Q&A, and a metrics snapshot.
