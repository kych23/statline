"""HTTP app: landing page, health check and the MCP endpoint."""

import contextlib
from collections.abc import AsyncIterator
from html import escape
from typing import Any

import httpx2
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount

from statline.server.app import build_server, mock_gate
from statline.server.auth import build_auth
from statline.server.auth_config import AuthConfig
from statline.server.data import ServerData
from statline.server.league import ATTRIBUTION, register_league
from statline.server.usage import UsageMiddleware, usage_key
from statline.yahoo.client import TTLCache

YAHOO_CACHE_TTL_S = 900

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
{league_section}<h2>Tools</h2>
<ul>
<li><code>get_player</code>: look up and compare players by name.</li>
<li><code>rankings</code>: top players by value over replacement or median projection.</li>
<li><code>mock_draft</code>: simulate a snake draft from your seat and return your roster.</li>
</ul>
<h2>Data</h2>
<p>Play-by-play and weekly stats come from nflverse. Projections are model output, not
official numbers. Average draft position is used internally to drive opponents in mock
drafts and is never shown.</p>
<p><a href="{github}">{github}</a> &middot; <a href="/privacy">Privacy</a></p>
{attribution}</body>
</html>
"""

LEAGUE_SECTION = """<h2>Your Yahoo league</h2>
<p>To see your roster and free agents with projections, add a second connector with the URL
<code>{league_url}</code> and sign in with Yahoo. Access is read-only.</p>
"""

ATTRIBUTION_HTML = (
    '<p><a href="{link}"><img src="{logo}" alt="Yahoo Fantasy" height="28"></a><br>{text}</p>\n'
)

PRIVACY = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Statline privacy policy</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;max-width:40rem;margin:2.5rem auto;padding:0 1rem;
color:#1a1a1a}
@media(prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}a{color:#8ab4ff}}
</style>
</head>
<body>
<h1>Privacy policy</h1>
<p>The public Statline tools (/mcp) need no sign-in. Apart from the usage logs below, the
rest applies only if you connect your Yahoo account to the league endpoint.</p>
<h2>What is stored</h2>
<ul>
<li>Usage logs: each tool call logs the tool name, whether it succeeded, how long it took
and a keyed one-way hash of your IP address (public tools) or Yahoo account id (league
tools). Never your IP address, account id, questions or results. Google Cloud keeps these
logs for 30 days.</li>
<li>Your Yahoo OAuth tokens, encrypted, in Google Firestore. They let Statline read your
fantasy leagues on your behalf.</li>
<li>A 15-minute in-memory cache of Yahoo responses, keyed by a hashed user id. It is never
written to disk and is lost on restart.</li>
</ul>
<h2>What is not done</h2>
<p>Your data is not sold or shared. Statline has read-only Yahoo access ("Fantasy Sports -
Read") and cannot change your leagues or rosters.</p>
<h2>Deleting your data</h2>
<p>Ask Claude to run the <code>delete_my_data</code> tool: it removes your stored tokens and
cached responses. You can also remove Statline at
<a href="https://login.yahoo.com/account/security/app-access">Yahoo's app access page</a>
to revoke access at the source.</p>
<h2>Contact</h2>
<p>Open an issue at <a href="{github}">{github}</a>.</p>
</body>
</html>
"""


def build_http_app(
    data: ServerData,
    auth_cfg: AuthConfig | None = None,
    *,
    auth: Any = None,
    yahoo_transport: httpx2.AsyncBaseTransport | None = None,
) -> Starlette:
    """Public /mcp always; with auth settings (or an injected proxy), /league/mcp too."""
    # Stateless: Cloud Run gives no session affinity across instances.
    # Host/Origin checks defend localhost servers against DNS rebinding; this is a public
    # read-only server on an unknown hostname, so they are explicitly off.
    http_options: dict[str, Any] = {"stateless_http": True, "host_origin_protection": False}
    gate = mock_gate()  # one gate: a single simulation at a time across both servers
    # One key for both servers: the same caller hashes alike across them.
    key = usage_key(auth_cfg.storage_encryption_key if auth_cfg is not None else None)
    public = build_server(data, gate, usage=UsageMiddleware("public", key))
    app = public.http_app(path="/mcp", **http_options)

    proxy = auth if auth is not None else build_auth(auth_cfg) if auth_cfg is not None else None
    league_app = None
    if proxy is not None:
        league_server = build_server(
            data,
            gate,
            name="statline-league",
            auth=proxy,
            usage=UsageMiddleware("league", key),
        )
        register_league(
            league_server, data, TTLCache(YAHOO_CACHE_TTL_S), proxy=proxy, transport=yahoo_transport
        )
        league_app = league_server.http_app(path="/league/mcp", **http_options)

    mcp_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(scope: Starlette) -> AsyncIterator[Any]:
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(mcp_lifespan(scope))
            if league_app is not None:
                await stack.enter_async_context(league_app.router.lifespan_context(league_app))
            yield

    app.router.lifespan_context = lifespan

    async def landing(request: Request) -> HTMLResponse:
        origin = str(request.base_url).rstrip("/")
        league_section = (
            LEAGUE_SECTION.format(league_url=escape(f"{origin}/league/mcp"))
            if league_app is not None
            else ""
        )
        return HTMLResponse(
            PAGE.format(
                season=data.season,
                as_of_week=data.as_of_week,
                data_date=escape(data.data_date),
                mcp_url=escape(f"{origin}/mcp"),
                league_section=league_section,
                github=GITHUB_URL,
                attribution=(
                    ATTRIBUTION_HTML.format(**{k: escape(v) for k, v in ATTRIBUTION.items()})
                    if league_app is not None
                    else ""
                ),
            )
        )

    async def privacy(request: Request) -> HTMLResponse:
        return HTMLResponse(PRIVACY.replace("{github}", GITHUB_URL))

    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "season": data.season,
                "as_of_week": data.as_of_week,
                "data_date": data.data_date,
            }
        )

    app.add_route("/", landing, methods=["GET"])
    app.add_route("/privacy", privacy, methods=["GET"])
    # not /healthz: Cloud Run reserves paths ending in "z" for its own front end
    app.add_route("/health", health, methods=["GET"])
    if league_app is not None:
        # Last, so the pages and /mcp win; the league app brings /league/mcp and the OAuth
        # and well-known routes, and its auth middleware stays scoped to it.
        app.router.routes.append(Mount("/", app=league_app))
    return app
