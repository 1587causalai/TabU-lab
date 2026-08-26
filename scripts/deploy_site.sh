#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC="$ROOT/site/public"
HOST="${TABU_SITE_HOST:-dgx2}"
STAGING="/home/cms/wehub-sites/research/tabu-lab"
LIVE="/var/www/research.wehub.us/tabu-lab"
ARCHIVE="/home/cms/wehub-sites/research/.backups/tabu-lab"
MARKER="tabu-lab-site-v20260826-02"
STAMP="$(date +%Y%m%d-%H%M%S)"

python3 "$ROOT/scripts/verify_site.py"

ssh "$HOST" "set -e; mkdir -p '$ARCHIVE'; if [ -d '$STAGING' ]; then cp -a '$STAGING' '$ARCHIVE/staging-before-$STAMP'; fi; mkdir -p '$STAGING'"
rsync -az --delete "$PUBLIC"/ "$HOST":"$STAGING"/

if [[ -z "${SSH_PASS_CMS:-}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$HOME/.openclaw/.env"
  set -u
fi
: "${SSH_PASS_CMS:?SSH_PASS_CMS is required for the www-data-owned public root}"
trap 'unset SSH_PASS_CMS' EXIT

printf '%s\n' "$SSH_PASS_CMS" | ssh "$HOST" "sudo -S -p '' bash -c 'set -e; mkdir -p \"$ARCHIVE\"; if [ -d \"$LIVE\" ]; then cp -a \"$LIVE\" \"$ARCHIVE/live-before-$STAMP\"; fi; mkdir -p \"$LIVE\"; rsync -a --delete \"$STAGING/\" \"$LIVE/\"; chown -R www-data:www-data \"$LIVE\"; find \"$LIVE\" -type d -exec chmod 755 {} +; find \"$LIVE\" -type f -exec chmod 644 {} +'"
unset SSH_PASS_CMS

ssh "$HOST" "set -e; grep -Fq '$MARKER' '$STAGING/index.html'; grep -Fq '$MARKER' '$LIVE/index.html'; grep -Fq '$MARKER' '$STAGING/zh/index.html'; grep -Fq '$MARKER' '$LIVE/zh/index.html'; test -f '$STAGING/agent.json'; test -f '$LIVE/agent.json'; sha256sum '$STAGING/index.html' '$LIVE/index.html' '$STAGING/zh/index.html' '$LIVE/zh/index.html'"

echo "Deployed https://research.wehub.us/tabu-lab/"
echo "Backup root: $HOST:$ARCHIVE"
