#!/usr/bin/env bash
# vLLM Watchdog — health check and auto-restart

PORT=${1:-8001}
CHECK_INTERVAL=30
FAIL_THRESHOLD=6       # 6 * 30s = 3 min before restart (model loads in ~30s)
RESTART_COOLDOWN=180   # 3 min between restarts
STARTUP_GRACE=120      # 2 min grace on first start

fail_count=0
last_restart=0

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Wait for initial startup
log "Watchdog started, grace period ${STARTUP_GRACE}s"
sleep $STARTUP_GRACE

while true; do
    if curl -sf --max-time 10 http://localhost:$PORT/health > /dev/null 2>&1; then
        if [[ $fail_count -gt 0 ]]; then
            log "Recovered after $fail_count failures"
        fi
        fail_count=0
    else
        fail_count=$((fail_count + 1))
        log "Health check failed ($fail_count/$FAIL_THRESHOLD)"
        
        if [[ $fail_count -ge $FAIL_THRESHOLD ]]; then
            now=$(date +%s)
            since_last=$((now - last_restart))
            
            if [[ $since_last -ge $RESTART_COOLDOWN ]]; then
                log "Restarting vLLM..."
                
                # Force kill GPU processes
                nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
                sleep 10
                
                systemctl restart vllm
                last_restart=$now
                fail_count=0
                log "Restart issued, grace period ${STARTUP_GRACE}s"
                sleep $STARTUP_GRACE
                continue
            else
                remaining=$((RESTART_COOLDOWN - since_last))
                log "Cooldown: ${remaining}s remaining"
            fi
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
