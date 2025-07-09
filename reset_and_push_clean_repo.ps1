# Step 1: Navigate to your project folder
cd "C:\Users\nitis\OneDrive\Desktop\Infocept\mcp_server_project"

# Step 2: Delete existing .git folder
if (Test-Path ".git") {
    Remove-Item -Recurse -Force ".git"
    Write-Host "Removed old .git folder."
} else {
    Write-Host "No .git folder found. Skipping."
}

# Step 3: Re-initialize Git
git init
Write-Host "Initialized new Git repo."

# Step 4: Create .gitignore (multi-line string block)
$gitignore = @'
.venv/
venv/
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
*.log
*.db
*.sqlite
Sample_dataset/
mcp_server/files/*
!mcp_server/files/.gitkeep
'@

Set-Content -Path ".gitignore" -Value $gitignore -Encoding UTF8
Write-Host "Created .gitignore file."

# Step 5: Add and commit clean files
git add .
git commit -m "Clean start: reset Git history and ignore large folders"
Write-Host "Committed clean project state."

# Step 6: Connect to GitHub
git remote add origin https://github.com/NitishRaman/mcp_server.git
Write-Host "Connected to GitHub remote."

# Step 7: Force push to overwrite GitHub repo
git branch -M main
git push --force origin main
Write-Host "Force pushed clean version to GitHub."
