#!/bin/bash
# DuckDNS IP update script — run as cron on the server
# Updates the DuckDNS domain to point to this server's current public IP
#
# Setup on server:
#   1. Set the token: echo "YOUR_DUCKDNS_TOKEN" > ~/.duckdns-token && chmod 600 ~/.duckdns-token
#   2. Add cron: crontab -e → */5 * * * * /home/ubuntu/stock-tracker/deploy/duckdns-update.sh >> /tmp/duckdns.log 2>&1
#   3. Also runs on boot via @reboot cron entry

DOMAIN="prediq"
TOKEN_FILE="$HOME/.duckdns-token"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "$(date): ERROR — token file missing: $TOKEN_FILE"
    exit 1
fi

TOKEN=$(cat "$TOKEN_FILE" | tr -d '[:space:]')
RESPONSE=$(curl -s "https://www.duckdns.org/update?domains=${DOMAIN}&token=${TOKEN}&ip=")

if [ "$RESPONSE" = "OK" ]; then
    echo "$(date): DuckDNS updated successfully"
else
    echo "$(date): DuckDNS update FAILED — response: $RESPONSE"
fi
