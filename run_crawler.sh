#!/bin/bash
# Start the crawler detached (survives the terminal). Safe to re-run: no-op if already running.
# A small supervisor loop restarts it if it crashes; a clean stop (SIGTERM / access denied) ends the loop.
cd "$(dirname "$0")"
if [ -f data/crawler.pid ] && kill -0 "$(cat data/crawler.pid)" 2>/dev/null; then
  echo "crawler already running (pid $(cat data/crawler.pid))"; exit 0
fi
if [ -f data/supervisor.pid ] && kill -0 "$(cat data/supervisor.pid)" 2>/dev/null; then
  echo "supervisor already running (pid $(cat data/supervisor.pid))"; exit 0
fi
nohup bash -c '
  echo $$ > data/supervisor.pid
  while true; do
    .venv/bin/python crawler.py "$@" >> data/crawler.stdout.log 2>&1
    code=$?
    [ $code -eq 0 ] && break          # clean stop
    echo "$(date) crawler exited $code, restarting in 30s" >> data/crawl.log
    sleep 30
  done
  rm -f data/supervisor.pid
' _ "$@" > /dev/null 2>&1 &
echo "crawler started"
