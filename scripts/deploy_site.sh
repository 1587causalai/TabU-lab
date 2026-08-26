#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC="$ROOT/site/public"
HOST="${TABU_SITE_HOST:-dgx2}"
STAGING="/home/cms/wehub-sites/research/tabu-lab"
LIVE="/var/www/research.wehub.us/tabu-lab"
ARCHIVE="/home/cms/wehub-sites/research/.backups/tabu-lab"
MARKER="tabu-lab-site-v20260826-01"
STAMP="$(date +%Y%m%d-%H%M%S)"

python3 "$ROOT/scripts/verify_site.py"

ssh "$HOST" "set -e; mkdir -p '$ARCHIVE'; if [ -d '$STAGING' ]; then cp -a '$STAGING' '$ARCHIVE/staging-before-$STAMP'; fi; if [ -d '$LIVE' ]; then cp -a '$LIVE' '$ARCHIVE/live-before-$STAMP'; fi; mkdir -p '$STAGING' '$LIVE'"
rsync -az --delete "$PUBLIC"/ "$HOST":"$STAGING"/
rsync -az --delete "$PUBLIC"/ "$HOST":"$LIVE"/

ssh "$HOST" "set -e; grep -Fq '$MARKER' '$STAGING/index.html'; grep -Fq '$MARKER' '$LIVE/index.html'; test -f '$STAGING/agent.json'; test -f '$LIVE/agent.json'; sha256sum '$STAGING/index.html' '$LIVE/index.html'"

echo "Deployed https://research.wehub.us/tabu-lab/"
echo "Backup root: $HOST:$ARCHIVE"
