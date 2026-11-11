"""HTTP app: landing page, health check and the MCP endpoint."""

from html import escape

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from statline.server.app import build_server
from statline.server.data import ServerData

GITHUB_URL = "https://github.com/kych23/statline"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Statline</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;max-width:40rem;margin:2.5rem auto;padding:0 1rem;
color:#1a1a1a}}
code{{background:#f0f0f0;padding:.1rem .3rem;border-radius:3px}}
h1{{margin-bottom:.2rem}}
@media(prefers-color-scheme:dark){{body{{background:#161616;color:#e8e8e8}}
code{{background:#2a2a2a}}a{{color:#8ab4ff}}}}
</style>
</head>
<body>
<h1>Statline</h1>
<p>Fantasy football projections and mock drafts for Claude. Statline is an MCP server:
ask Claude about players, rankings, or a full snake mock draft, and get 10th/50th/90th
percentile points-per-game forecasts from a gradient-boosted model.</p>
<p>Projections are for the {season} season, as of week {as_of_week}; data as of {data_date}.</p>
<h2>Add it to Claude</h2>
<p>Settings &rarr; Connectors &rarr; Add custom connector, then use the URL
<code>{mcp_url}</code></p>
<h2>Tools</h2>
<ul>
<li><code>get_player</code>: look up and compare players by name.</li>
<li><code>rankings</code>: top players by value over replacement or median projection.</li>
<li><code>mock_draft</code>: simulate a snake draft from your seat and return your roster.</li>
</ul>
<h2>Data</h2>
<p>Play-by-play and weekly stats come from nflverse. Projections are model output, not
official numbers. Average draft position is used internally to drive opponents in mock
drafts and is never shown.</p>
<p><a href="{github}">{github}</a></p>
</body>
</html>
"""


def build_http_app(data: ServerData) -> Starlette:
    # Stateless: Cloud Run gives no session affinity across instances.
    # Host/Origin checks defend localhost servers against DNS rebinding; this is a public
    # read-only server on an unknown hostname, so they are explicitly off.
    app = build_server(data).http_app(
        path="/mcp", stateless_http=True, host_origin_protection=False
    )
    # The MCP app already carries the lifespan its session manager needs.

    async def landing(request: Request) -> HTMLResponse:
        origin = str(request.base_url).rstrip("/")
        return HTMLResponse(
            PAGE.format(
                season=data.season,
                as_of_week=data.as_of_week,
                data_date=escape(data.data_date),
                mcp_url=escape(f"{origin}/mcp"),
                github=GITHUB_URL,
            )
        )

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "season": data.season,
                "as_of_week": data.as_of_week,
                "data_date": data.data_date,
            }
        )

    app.add_route("/", landing, methods=["GET"])
    app.add_route("/healthz", healthz, methods=["GET"])
    return app
