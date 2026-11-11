"""Serve Statline over HTTP: landing page, /health, MCP at /mcp, Yahoo MCP at /league/mcp."""

import logging
import os
from pathlib import Path

import uvicorn

from statline.server.auth_config import load_auth_config
from statline.server.data import ServerData
from statline.server.http import build_http_app

log = logging.getLogger("statline.serve")


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    draft = env_path("STATLINE_DRAFT_PROJECTIONS", "data/projections_2026_w0.parquet")
    adp = env_path("STATLINE_ADP", "data/raw/adp_2026.parquet")
    have_board = draft.exists() and adp.exists()
    if not have_board:
        log.warning("draft inputs missing (%s, %s); mock_draft will be unavailable", draft, adp)
    data = ServerData.load(
        env_path("STATLINE_PROJECTIONS", "data/projections.parquet"),
        env_path("STATLINE_LEAGUE", "config/league.yaml"),
        draft if have_board else None,
        adp if have_board else None,
    )
    # No banner or update check: those only run via FastMCP.run(), which we bypass.
    auth_cfg = load_auth_config()
    log.info("league auth %s", "enabled" if auth_cfg else "disabled")
    uvicorn.run(
        build_http_app(data, auth_cfg),
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        # Trust X-Forwarded-Proto so URLs built from the request say https; only Cloud Run's
        # front end can reach the container, so trusting any forwarder is safe.
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
