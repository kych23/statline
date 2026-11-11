# Statline

[![CI](https://github.com/kych23/statline/actions/workflows/ci.yml/badge.svg)](https://github.com/kych23/statline/actions/workflows/ci.yml)

Fantasy football projections and league context for Claude. Statline is an
[MCP](https://modelcontextprotocol.io) server: add it to Claude as a connector and ask
about players, mock drafts, or your Yahoo league. Projections are quantile forecasts
(10th/50th/90th percentile points per game) from a gradient-boosted model trained on
nflverse data.

**Status:** all 5 milestones live: projections, draft logic, public MCP server, Yahoo league.

## Try it

Add Statline to Claude: **Settings → Connectors → Add custom connector**, name it Statline,
URL:

```
https://statline-471351099718.us-central1.run.app/mcp
```

No sign-in needed. Then ask things like *"top 5 TEs rest of season"*, *"compare Gibbs and
Bijan"* or *"mock draft from seat 4"*.

| Tool | What it returns |
|:--|:--|
| `get_player` | Projections (10th/50th/90th percentile), VOR and position rank for 1-8 players, with a comparison |
| `rankings` | Top players overall or by position, by VOR or median projection |
| `mock_draft` | A full 12-team snake draft from your seat, replayable by seed |

### Your Yahoo league

Add a second connector with **Sign in now**:

```
https://statline-471351099718.us-central1.run.app/league/mcp
```

It signs you in with Yahoo (read-only) and adds league tools on top of the three above:

| Tool | What it returns |
|:--|:--|
| `my_league` | Your leagues, scoring (flags non-PPR), standings, and your roster with projections, bye weeks, IR eligibility and IR slots used |
| `free_agents` | The best available players in your league, ranked by VOR, with injury status, bye week and IR eligibility |
| `delete_my_data` | Deletes your stored tokens and cached Yahoo data |

Statline is an OAuth proxy between Claude and Yahoo: Claude receives Statline's own token and
Yahoo tokens never leave the server. They are stored encrypted (Fernet) in Firestore, so
sign-ins survive restarts. Details in [/privacy](https://statline-471351099718.us-central1.run.app/privacy).
Fantasy data provided by Yahoo Fantasy.

## Projections

Rest-of-season PPR points per game for every QB, RB, WR and TE, at any point in the season:
a 10th percentile (floor), median and 90th percentile (ceiling). One LightGBM model per
quantile, trained on nflverse weekly data from 2016 on, with features computed strictly
from games played before the prediction (season-to-date and last-3-game usage, prior
season, age). Intervals are calibrated per position on held-out seasons (split conformal).

```sh
uv run python scripts/build.py --season 2026 --as-of-week 9   # -> data/projections.parquet
uv run python scripts/build.py --backtest                     # the table below
```

### Backtest

Walk-forward: each test season (2022-2025) is predicted by a model trained only on
earlier seasons, at every week t = 0 (preseason) to 14. Error is mean absolute error in
points per game; lower is better. The blend weights last season against this season by
games played, with its weight fitted on training seasons.

| Method | MAE t 0-4 | MAE t 5-9 | MAE t 10-14 | MAE all | p10-p90 coverage |
|:--|--:|--:|--:|--:|--:|
| Season to date | 3.28 | 2.84 | 2.88 | 3.01 | |
| Prior season | 3.00 | 3.13 | 3.40 | 3.17 | |
| Blend | 2.78 | 2.66 | 2.81 | 2.75 | |
| **Statline** | **2.51** | **2.50** | **2.70** | **2.56** | **0.80** |

On players already fantasy-relevant before the prediction (prior-season or season-to-date
PPG of 10 or more), Statline's MAE is 3.34 vs 3.60 for the blend. Coverage is the share of
outcomes that land between the floor and the ceiling; 0.80 is the target.

**Limitations**

- Backtest rows require 2+ remaining games, so misses below the floor will be more
  common live, where some players are about to get hurt or benched.
- No preseason projection for rookies (no prior-season data yet).
- Coverage is lower for QBs (0.76) and varies by season (0.74 in 2024).

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync              # create .venv and install dependencies
uv run pytest        # tests
uv run ruff check    # lint
uv run pyright       # type check
```


## Mock draft

Simulates a 12-team snake draft (settings in `config/league.yaml`) from your seat.

```sh
uv run python scripts/build.py --season 2026 --as-of-week 0 --output data/projections_2026_w0.parquet
uv run python scripts/mock.py --seat 4 --seed 7 --projections data/projections_2026_w0.parquet
```

```
 round  pick_no                name position  p50  vor
     1        4  Jaxon Smith-Njigba       WR 16.9  7.1
     2       21        Trey McBride       TE 15.3  5.9
     3       28       Davante Adams       WR 13.9  4.1
     4       45    Javonte Williams       RB 12.5  3.1
   ...
```

- **Your picks** maximize the gain to your starting lineup, measured as value over
  replacement (VOR): projected points minus the best player left unstarted league-wide at
  that slot. FLEX gets its own replacement level, since a TE there competes with RBs and
  WRs. A small credit for bench depth, weighted by position (RB/WR bench players cover more
  starting slots than QB/TE), breaks late-round ties.
- **Opponents** pick from ADP (average draft position, Fantasy Football Calculator) with
  noise: P(player) ∝ exp(−ADP / τ), where τ = 1 + 0.07 × pick number. The randomness grows
  through the draft, as it does in real drafts. τ was fitted so simulated pick spreads
  match real ones from thousands of drafts, 2016-2026 (e.g. 1.6 vs 1.6 picks for ADP
  1-12, 13.8 vs 14.2 for ADP 121+).
- Every team fills a legal starting lineup (position caps, plus a must-fill rule near the
  end), and the same seed replays the same draft.
- ADP is used only to drive opponents and is never displayed. The cached board is local
  and not committed.


## Usage metrics

Each tool call logs one JSON line (logger `statline.usage`, stdout) that Cloud Run ingests as
a structured entry: tool name, `public` or `league` server, success, duration in ms, and a
12-character keyed hash of the caller (IP address on the public server, Yahoo account id on
the league server). Arguments, results, IPs and account ids are never logged. The HMAC key is
derived from `STORAGE_ENCRYPTION_KEY`; without it (public-only) hashes change on restart.
Only league hashes count real users: claude.ai sends public requests from shared
infrastructure, so many users share one public hash. Cloud Logging keeps the lines 30 days.
Summarize them (needs `gcloud` auth):

```sh
uv run python scripts/usage.py --days 30 --project statline-mcp
```

## Deployment

Google Cloud Run, built from the `Dockerfile` by Cloud Build (no local Docker needed). The
image installs only runtime dependencies (`uv sync --no-default-groups`); the training stack
lives in the `train` dependency group. Cost stays at $0 on the free tier: `--min-instances 0`
(scales to zero), `--max-instances 1`, an hourly `mock_draft` limit (60 shared by public callers, since claude.ai relays them from shared IPs; 10 per signed-in Yahoo account), an image cleanup
policy that keeps only the newest image, and a $1 budget alert.

Secrets (Yahoo client ID/secret, JWT signing key, storage encryption key) live in Secret
Manager and are passed with `--set-secrets`; the service account has only
`secretmanager.secretAccessor` and `datastore.user`.

Deploys normally happen through the weekly refresh workflow below. To deploy by hand
(first deploy, or changing flags, env vars or secrets):

```sh
uv run python scripts/build.py --season 2026 --as-of-week latest
gcloud run deploy statline --source . --region us-central1 --allow-unauthenticated \
  --min-instances 0 --max-instances 1 --memory 1Gi --cpu 1 --concurrency 10 --timeout 300 \
  --set-secrets YAHOO_CLIENT_ID=YAHOO_CLIENT_ID:latest,YAHOO_CLIENT_SECRET=YAHOO_CLIENT_SECRET:latest,JWT_SIGNING_KEY=JWT_SIGNING_KEY:latest,STORAGE_ENCRYPTION_KEY=STORAGE_ENCRYPTION_KEY:latest \
  --set-env-vars STATLINE_BASE_URL=https://statline-471351099718.us-central1.run.app,GOOGLE_CLOUD_PROJECT=statline-mcp
```

`.gcloudignore` is an allowlist: it uploads the gitignored `data/` files the image needs and
nothing else.

## Weekly refresh

`.github/workflows/refresh.yml` rebuilds and redeploys the projections every Tuesday at
15:00 UTC (after Monday Night Football). It reads the completed regular-season weeks from
the nflverse schedule, skips if that week is already live (`/health` `as_of_week`), and
otherwise builds the projections, checks them, builds the image, deploys it to Cloud Run
and smoke-tests `/health`. A deploy only swaps the image: env vars, secrets and scaling
stay as configured.

```sh
gh workflow run refresh.yml                 # run now; skips if the week is already live
gh workflow run refresh.yml -f force=true   # redeploy even if unchanged, e.g. after a code change
```

- **Auth:** GitHub OIDC through Workload Identity Federation to a deploy-only service
  account. No keys are stored in the repo or in GitHub secrets.
- **Quality gate:** `scripts/build.py --check` fails the run before anything is deployed
  unless the file loads the way the server loads it, has at least 300 players, no null or
  crossed p10/p50/p90, and a single `as_of_week` and season matching the expected ones.
- **Rollback:** if `/health` does not report `ok` and the new week within about a minute
  after deploy, traffic is shifted back to the previous revision and the run fails.
- **Private data:** ADP and the week-0 projections (used by the mock draft board) are
  gitignored and live in a private Cloud Storage bucket; the workflow copies them in before
  building. Bump `SEASON` and the file names in the workflow each season.
