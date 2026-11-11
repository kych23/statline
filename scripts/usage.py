"""Summarize tool-call usage from Cloud Run logs.

uv run python scripts/usage.py --days 30 --project statline-mcp
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

SERVICE = "statline"
LIMIT = 100_000
GCLOUD_TIMEOUT_S = 300


def build_filter(since: datetime) -> str:
    return (
        'resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{SERVICE}" '
        'AND jsonPayload.event="tool_call" '
        f'AND timestamp>="{since.isoformat()}"'
    )


def fetch_entries(project: str, days: int) -> list[dict[str, Any]]:
    since = datetime.now(UTC) - timedelta(days=days)
    cmd = [
        "gcloud",
        "logging",
        "read",
        build_filter(since),
        "--format=json",
        f"--project={project}",
        f"--limit={LIMIT}",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=GCLOUD_TIMEOUT_S)
    return json.loads(out.stdout or "[]")


def parse_entries(entries: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per tool_call entry: ts, tool, server, user, ok, ms."""
    rows = []
    for entry in entries:
        payload = entry.get("jsonPayload") or {}
        if payload.get("event") != "tool_call":
            continue
        rows.append(
            {
                "ts": pd.Timestamp(entry["timestamp"]),
                "tool": payload.get("tool"),
                "server": payload.get("server"),
                "user": payload.get("user"),
                "ok": bool(payload.get("ok")),
                "ms": payload.get("ms"),
            }
        )
    return pd.DataFrame(rows, columns=["ts", "tool", "server", "user", "ok", "ms"])


def _identified_users(frame: pd.DataFrame) -> int:
    return int(frame.loc[frame["user"] != "anon", "user"].nunique())


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Pure aggregation; every value is plain Python or a DataFrame."""
    if frame.empty:
        return {"total_calls": 0}
    by_server = frame.groupby("server").agg(
        calls=("tool", "size"), users=("user", lambda s: s[s != "anon"].nunique())
    )
    by_tool = frame.groupby("tool").agg(
        calls=("tool", "size"),
        users=("user", lambda s: s[s != "anon"].nunique()),
        error_rate=("ok", lambda s: 1.0 - s.mean()),
        p50_ms=("ms", lambda s: s.quantile(0.5)),
        p95_ms=("ms", lambda s: s.quantile(0.95)),
    )
    per_day = frame.groupby(frame["ts"].dt.strftime("%Y-%m-%d")).size().rename("calls")
    return {
        "total_calls": len(frame),
        "distinct_users": _identified_users(frame),
        "anon_calls": int((frame["user"] == "anon").sum()),
        "by_server": by_server,
        "by_tool": by_tool.sort_values("calls", ascending=False),
        "per_day": per_day,
    }


def render(summary: dict[str, Any]) -> str:
    if summary["total_calls"] == 0:
        return "No tool calls in this window."
    tool_table = summary["by_tool"].round({"error_rate": 3, "p50_ms": 0, "p95_ms": 0})
    return "\n".join(
        [
            f"Total calls: {summary['total_calls']}",
            f"Distinct callers (excluding anon): {summary['distinct_users']}",
            f"Anon calls: {summary['anon_calls']}",
            "Note: league callers are Yahoo accounts (real users). Public callers are source IPs,",
            "and claude.ai sends every user's requests from shared infrastructure, so the public",
            "count understates users.",
            "",
            "By server",
            summary["by_server"].to_string(),
            "",
            "By tool",
            tool_table.to_string(),
            "",
            "Calls per day",
            summary["per_day"].to_string(),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Statline tool-call usage.")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--project", default="statline-mcp")
    args = parser.parse_args()
    try:
        entries = fetch_entries(args.project, args.days)
    except FileNotFoundError:
        print("gcloud not found; install the Google Cloud CLI.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"gcloud failed (exit {exc.returncode}): {exc.stderr.strip()}", file=sys.stderr)
        return 1
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"could not read gcloud output: {exc}", file=sys.stderr)
        return 1
    print(render(summarize(parse_entries(entries))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
