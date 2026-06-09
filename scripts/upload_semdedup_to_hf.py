"""Upload the 397B-semdedup data files (per-news complete attributions) to HF.

Uploads the 6 jsonl files to the REPO ROOT. Does NOT touch the repo README.

Login first (interactive, run it yourself):
    hf auth login

Usage:
    python scripts/upload_semdedup_to_hf.py <repo_id>
    # or set REPO_ID env var
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, get_token

REPO_TYPE = 'dataset'
PRIVATE = True
DATA_DIR = Path(
    '/home/haofeiy2/social-world-model/data/'
    'social-world-model-v6-qwen3.5-397B-clean-semdedup'
)

FILES = [
    'test_kalshi_final.jsonl',
    'test_polymarket_final.jsonl',
    'train.jsonl',
    'train_clean.jsonl',
    'valid_clean.jsonl',
    'valid_subset150.jsonl',
]


def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('REPO_ID')
    if not repo_id:
        sys.exit('ERROR: pass repo id as arg or set REPO_ID env var')
    if not get_token():
        sys.exit('ERROR: not logged in. Run `hf auth login` first.')

    api = HfApi()
    print(f'Ensuring repo: {repo_id} (private={PRIVATE})')
    create_repo(repo_id, repo_type=REPO_TYPE, private=PRIVATE, exist_ok=True)

    for fname in FILES:
        local = DATA_DIR / fname
        if not local.exists():
            print(f'  SKIP (missing): {fname}')
            continue
        size_mb = local.stat().st_size / 1024 / 1024
        print(f'  uploading {fname} ({size_mb:.1f} MB) -> repo root ...')
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
        )
        print('    done')

    print(f'\nDone -> https://huggingface.co/datasets/{repo_id}')


if __name__ == '__main__':
    main()
