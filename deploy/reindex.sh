#!/usr/bin/env bash
#
# Refresh the Chroma index from the wiki, with the bot stopped for the
# duration.
#
# The stop/start isn't paranoia: Chroma persists to a SQLite file plus its
# own HNSW segment files, and it doesn't support two processes touching one
# path at once. A running bot would also keep serving its already-loaded
# in-memory index, so it wouldn't see the new pages until a restart anyway.
#
# Run as root (the systemd unit does); the indexing itself drops to $BOT_USER
# so nothing under data/ ends up owned by root.
set -euo pipefail

BOT_USER=ubuntu
APP_DIR=/home/ubuntu/browiki-bot
SERVICE=browiki-bot.service

was_running=0
if systemctl is-active --quiet "$SERVICE"; then
    was_running=1
    echo "Stopping $SERVICE for reindex..."
    systemctl stop "$SERVICE"
fi

# Bring the bot back even if the crawl fails partway. build_index is
# resumable (data/ingest_state.json), so a failed run costs a retry, not a
# rebuild — but an offline bot costs the whole server.
restart_bot() {
    if [ "$was_running" -eq 1 ]; then
        echo "Restarting $SERVICE..."
        systemctl start "$SERVICE"
    fi
}
trap restart_bot EXIT

cd "$APP_DIR"
runuser -u "$BOT_USER" -- "$APP_DIR/.venv/bin/python" -m ingest.build_index "$@"
echo "Reindex finished."
