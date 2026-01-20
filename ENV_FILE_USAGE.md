# Environment File Usage Guide

## Files Overview

| File | Purpose | Git Tracked | Usage |
|------|---------|-------------|-------|
| [.env.example](.env.example) | Template with all options | ✅ Yes | Copy to create your local `.env` |
| [.env](.env) | **Your local configuration** | ❌ No (gitignored) | Edit with your actual settings |
| `.env.local` | Alternative local config | ❌ No (gitignored) | Optional, higher priority |

## Quick Setup

### Option 1: Use the existing .env file (Recommended)

A `.env` file has already been created for you:

```bash
# Edit the .env file with your settings
nano .env
```

Or use your preferred editor:
```bash
code .env      # VS Code
vim .env       # Vim
gedit .env     # Gedit
```

**Current contents:**
```bash
# Vision Models API Server - Local Configuration
HF_ROOT_FOLDER=.
# HF_TOKEN=

# CUDA Configuration (optional)
# CUDA_VISIBLE_DEVICES=0
```

### Option 2: Copy from template

If you need to recreate the file:

```bash
cp .env.example .env
# Then edit .env with your settings
```

## Configuration Examples

### Example 1: Development (Default)

Use current directory for cache:

```bash
# .env file
HF_ROOT_FOLDER=.
```

Start server:
```bash
python run_api_server.py --reload
```

### Example 2: Custom Cache Directory

Set a specific cache location:

```bash
# .env file
HF_ROOT_FOLDER=/data/models/cache
```

### Example 3: With HuggingFace Token

For private/gated models:

```bash
# .env file
HF_ROOT_FOLDER=/data/models/cache
HF_TOKEN=hf_your_actual_token_here
```

### Example 4: CPU-Only Mode

Disable GPU:

```bash
# .env file
HF_ROOT_FOLDER=.
CUDA_VISIBLE_DEVICES=-1
```

### Example 5: Specific GPU

Use only GPU 0:

```bash
# .env file
HF_ROOT_FOLDER=.
CUDA_VISIBLE_DEVICES=0
```

### Example 6: Production Setup

```bash
# .env file
HF_ROOT_FOLDER=/var/lib/vision-models/cache
HF_TOKEN=hf_your_production_token
CUDA_VISIBLE_DEVICES=0,1
```

## Loading Environment Variables

### Automatic Loading (Recommended - requires python-dotenv)

Install python-dotenv:
```bash
pip install python-dotenv
```

Load in your startup script:
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env automatically

# Now start server
import uvicorn
from vision_model_repo.api.server import app
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Manual Loading (Shell)

Load into your shell session:
```bash
# Export all variables from .env
export $(cat .env | grep -v '^#' | xargs)

# Then start server
python run_api_server.py
```

Or source it:
```bash
set -a
source .env
set +a

python run_api_server.py
```

### Using direnv (Optional - Automatic)

Install direnv:
```bash
# Ubuntu/Debian
sudo apt install direnv

# macOS
brew install direnv
```

Add to your shell config (e.g., `~/.bashrc`):
```bash
eval "$(direnv hook bash)"
```

Allow the directory:
```bash
cd /home/vitimarioganiga/VisionModelsRepo
echo "dotenv" > .envrc
direnv allow
```

Now `.env` is automatically loaded when you `cd` into this directory!

## Verification

Check if environment variables are set:

```bash
# Check individual variables
echo $HF_ROOT_FOLDER
echo $HF_TOKEN

# Or check all
env | grep HF_
```

Start server and check logs:
```bash
python run_api_server.py

# Look for:
# INFO: Configuring HuggingFace environment: root=/your/path, token=***
```

## Security Best Practices

### ✅ DO:
- ✅ Keep `.env` files local (they are gitignored)
- ✅ Use different `.env` files for development/production
- ✅ Store production tokens in secure vaults (AWS Secrets, etc.)
- ✅ Use read-only HuggingFace tokens when possible
- ✅ Rotate tokens periodically

### ❌ DON'T:
- ❌ Commit `.env` files to git (already prevented by `.gitignore`)
- ❌ Share your `.env` file with others
- ❌ Include `.env` in Docker images
- ❌ Use production tokens in development
- ❌ Hardcode tokens in source code

## Git Status

Verify `.env` is ignored:

```bash
# Check if .env is ignored
git check-ignore .env

# Should output: .env

# Verify it's not tracked
git status

# .env should NOT appear in the output
```

## Troubleshooting

### `.env` file not loaded

**Problem**: Environment variables not set when starting server.

**Solutions**:

1. Install python-dotenv:
   ```bash
   pip install python-dotenv
   ```

2. Manually export before starting:
   ```bash
   export HF_ROOT_FOLDER=/your/path
   export HF_TOKEN=your_token
   python run_api_server.py
   ```

3. Use shell source:
   ```bash
   set -a; source .env; set +a
   python run_api_server.py
   ```

### Wrong cache directory

**Problem**: Models downloaded to wrong location.

**Solution**: Check `HF_ROOT_FOLDER` is set correctly:

```bash
# Verify
echo $HF_ROOT_FOLDER

# Set if needed
export HF_ROOT_FOLDER=/correct/path
```

### Token not working

**Problem**: Private models fail to download.

**Solution**:

1. Verify token is valid: https://huggingface.co/settings/tokens
2. Check token has read access
3. Ensure no extra spaces in `.env` file:
   ```bash
   HF_TOKEN=hf_xxx  # Correct
   HF_TOKEN = hf_xxx  # Wrong (spaces)
   ```

## Multiple Environment Files

You can use multiple `.env` files:

```bash
# Development
.env.development

# Production
.env.production

# Load specific file
python -c "from dotenv import load_dotenv; load_dotenv('.env.production')"
python run_api_server.py
```

## Current File Status

Your repository already has:

✅ `.env` - Local configuration file (gitignored)
✅ `.env.example` - Template file (tracked in git)
✅ `.gitignore` - Configured to ignore `.env` files

You can safely edit `.env` with your actual credentials - it will never be committed to git.

## Quick Reference Commands

```bash
# Create/edit .env
nano .env

# Load and verify
export $(cat .env | grep -v '^#' | xargs)
env | grep HF_

# Start server with config
python run_api_server.py

# Check cache location
ls -la $(python -c "import os; print(os.getenv('HF_ROOT_FOLDER', '.'))")/content/hf_cache/
```

## See Also

- [CONFIG.md](CONFIG.md) - Complete configuration guide
- [CONFIGURATION_SUMMARY.md](CONFIGURATION_SUMMARY.md) - Quick reference
- [API_USAGE.md](API_USAGE.md) - API usage examples
