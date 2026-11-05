#!/bin/bash
# Stop crawler cleanly (it commits and exits; supervisor then exits too).
cd "$(dirname "$0")"
[ -f data/crawler.pid ] && kill "$(cat data/crawler.pid)" 2>/dev/null && echo "sent SIGTERM to crawler"
