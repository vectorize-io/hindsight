# Hindsight systemd Services

This directory contains systemd service units for Hindsight.

## Services

- `hindsight-api.service` — FastAPI API server (port 8888)
- `hindsight-control-plane.service` — Next.js web UI (port 9999)

## Installation

1. Copy the env template to the config location:

   ```bash
   sudo cp /opt/hindsight/env /etc/hindsight/env
   ```

2. Edit `/etc/hindsight/env` and set your `HINDSIGHT_API_LLM_API_KEY`:

   ```bash
   sudo $EDITOR /etc/hindsight/env
   ```

3. Enable and start the services:

   ```bash
   sudo systemctl enable --now hindsight-api
   sudo systemctl enable --now hindsight-control-plane
   ```

## Verification

```bash
# Check service status
sudo systemctl status hindsight-api
sudo systemctl status hindsight-control-plane

# Verify API is responding
curl http://localhost:8888/health

# View logs
sudo journalctl -u hindsight-api --no-pager -f
sudo journalctl -u hindsight-control-plane --no-pager -f
```

## Ports

| Service | Port | URL |
|---------|------|-----|
| API | 8888 | http://localhost:8888 |
| Control Plane | 9999 | http://localhost:9999 |

## Notes

- The API service starts before the control plane (via `Wants=`/`After=`).
- Both services run under the `hindsight` system user.
- Config is read from `/etc/hindsight/env`.
- Data is stored in `/var/lib/hindsight`.
- Graceful shutdown timeout is 30 seconds.
