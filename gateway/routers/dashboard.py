from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from gateway.db import get_session, UsageRecord
import logging
from datetime import datetime, timedelta

router = APIRouter()
log = logging.getLogger("gateway")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Gateway Dashboard | Token Savings</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --primary: #38bdf8;
            --success: #22c55e;
            --accent: #8b5cf6;
            --border: #334155;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3rem;
        }
        h1 {
            font-size: 1.875rem;
            font-weight: 700;
            background: linear-gradient(to right, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .card:hover { transform: translateY(-2px); }
        .card-label {
            font-size: 0.875rem;
            color: var(--text-dim);
            font-weight: 500;
            margin-bottom: 0.5rem;
            display: block;
        }
        .card-value { font-size: 2rem; font-weight: 700; }
        .card-sub {
            font-size: 0.875rem;
            color: var(--success);
            font-weight: 600;
            margin-top: 0.25rem;
        }
        .table-container {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 1rem;
            overflow: hidden;
            max-height: 400px;
            overflow-y: auto;
        }
        table { width: 100%; border-collapse: collapse; text-align: left; }
        thead { background: rgba(255, 255, 255, 0.05); }
        th {
            padding: 1rem 1.5rem;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-dim);
            border-bottom: 1px solid var(--border);
        }
        td {
            padding: 1rem 1.5rem;
            font-size: 0.875rem;
            border-bottom: 1px solid var(--border);
        }
        .badge {
            padding: 0.25rem 0.625rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-cached { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
        .badge-miss { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }
        .refresh-btn {
            background: var(--primary);
            color: #000;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .refresh-btn:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>AI Gateway Insights</h1>
                <p style="color: var(--text-dim)">{{VIEW_LABEL}}</p>
            </div>
            <div>
                {{MODE_BUTTONS}}
            </div>
        </header>

        <div class="stats-grid">
            <div class="card">
                <span class="card-label">Total Input Tokens</span>
                <div class="card-value">{{TOTAL_INPUT}}</div>
                <div class="card-sub">{{TOTAL_CACHED}} Cached</div>
            </div>
            <div class="card">
                <span class="card-label">Efficiency Ratio</span>
                <div class="card-value">{{EFFICIENCY}}</div>
                <div class="card-sub">Tokens Saved</div>
            </div>
            <div class="card">
                <span class="card-label">Estimated Savings</span>
                <div class="card-value">{{SAVINGS_USD}}</div>
                <div class="card-sub">USD Predicted</div>
            </div>
            <div class="card">
                <span class="card-label">Total Requests</span>
                <div class="card-value">{{REQUEST_COUNT}}</div>
                <div class="card-sub">API Calls</div>
            </div>
        </div>

        <h2 style="margin-bottom: 1.5rem; font-size: 1.25rem;">Token Savings Breakdown</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
            <div class="card" style="border-left: 4px solid #10b981;">
                <span class="card-label">Anthropic Cache</span>
                <div class="card-value">{{TOTAL_CACHED}} tokens</div>
                <div class="card-sub" style="color: #10b981;">Cached by Anthropic</div>
            </div>
            <div class="card" style="border-left: 4px solid #3b82f6;">
                <span class="card-label">Gateway Savings</span>
                <div class="card-value">{{GATEWAY_SAVED}} tokens</div>
                <div class="card-sub" style="color: #3b82f6;">Pruning + Stripping</div>
            </div>
            <div class="card" style="border-left: 4px solid #8b5cf6;">
                <span class="card-label">Total Efficiency</span>
                <div class="card-value">{{EFFICIENCY}}</div>
                <div class="card-sub" style="color: #8b5cf6;">Combined savings</div>
            </div>
            <div class="card" style="border-left: 4px solid #f59e0b;">
                <span class="card-label">Cost Saved</span>
                <div class="card-value">{{SAVINGS_USD}}</div>
                <div class="card-sub" style="color: #f59e0b;">Estimated USD</div>
            </div>
        </div>

        <h2 style="margin-bottom: 1.5rem; font-size: 1.25rem;">Recent Requests</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Model</th>
                        <th>Status</th>
                        <th>Input</th>
                        <th>Cached</th>
                        <th>Gateway</th>
                        <th>Output</th>
                        <th>Savings</th>
                    </tr>
                </thead>
                <tbody>
                    {{RECENT_ROWS}}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

NO_REDIS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Gateway -- Setup Required</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .box { background: #1e293b; border: 1px solid #334155; border-radius: 1rem; padding: 2.5rem; max-width: 520px; text-align: center; }
        h1 { font-size: 1.5rem; margin-bottom: 0.5rem; color: #38bdf8; }
        p { color: #94a3b8; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="box">
        <h1>Redis Not Connected</h1>
        <p>The dashboard requires Redis to store real-time stats. Please add REDIS_URL to your environment variables.</p>
    </div>
</body>
</html>
"""


def _get_stats_from_redis(full: bool) -> dict:
    """Read dashboard stats directly from Redis counters. O(1) operation."""
    from gateway.cache import rds
    
    defaults = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "gateway_saved": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }
    
    if not rds:
        return defaults
    
    try:
        if full:
            data = rds.hgetall("dashboard:stats:all_time") or {}
        else:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
            d1 = rds.hgetall(f"dashboard:stats:{today}") or {}
            d2 = rds.hgetall(f"dashboard:stats:{yesterday}") or {}
            data = {}
            for k in ["input_tokens", "output_tokens", "cached_tokens", "gateway_saved", "request_count"]:
                data[k] = int(d1.get(k, 0)) + int(d2.get(k, 0))
            data["cost_usd"] = float(d1.get("cost_usd", 0)) + float(d2.get("cost_usd", 0))
        
        return {
            "input_tokens": int(data.get("input_tokens", 0)),
            "output_tokens": int(data.get("output_tokens", 0)),
            "cached_tokens": int(data.get("cached_tokens", 0)),
            "gateway_saved": int(data.get("gateway_saved", 0)),
            "cost_usd": float(data.get("cost_usd", 0)),
            "request_count": int(data.get("request_count", 0)),
        }
    except Exception as e:
        log.warning("Failed to read dashboard stats from Redis: %r", e)
        return defaults


async def _get_recent_requests(limit: int = 20) -> list[dict]:
    """Fetch recent requests from DB. Fast query using timestamp index."""
    from gateway.db import async_session_factory
    
    if not async_session_factory:
        return []
    
    try:
        async with get_session() as session:
            result = await session.execute(
                select(UsageRecord).order_by(UsageRecord.timestamp.desc()).limit(limit)
            )
            rows = []
            for r in result.scalars().all():
                total_in = r.input_tokens or 0
                cache_r = r.cache_read_input_tokens or 0
                gw_saved = r.gateway_tokens_saved or 0
                total_tok = total_in + cache_r + gw_saved
                sav_pct = ((cache_r + gw_saved) / total_tok * 100) if total_tok > 0 else 0.0
                ts = r.timestamp.strftime("%H:%M:%S") if r.timestamp else "00:00:00"
                rows.append({
                    "timestamp": ts,
                    "model": (r.model or "unknown").replace("claude-3-5-", "").replace("claude-", ""),
                    "input": total_in,
                    "cache_read": cache_r,
                    "gateway_saved": gw_saved,
                    "output": r.output_tokens or 0,
                    "savings_pct": f"{sav_pct:.1f}",
                })
            return rows
    except Exception as e:
        log.warning("Failed to fetch recent requests: %r", e)
        return []


def _render_dashboard(stats: dict, recent: list[dict], full: bool) -> str:
    """Render dashboard HTML with stats and recent requests."""
    total_input = stats["input_tokens"]
    cached = stats["cached_tokens"]
    gateway_saved = stats["gateway_saved"]
    request_count = stats["request_count"]
    
    total_processed = total_input + cached + gateway_saved
    efficiency = ((cached + gateway_saved) / total_processed * 100) if total_processed > 0 else 0.0
    
    cache_savings = cached * 0.001 * 0.003
    gateway_savings = gateway_saved * 0.001 * 0.003
    savings_usd = cache_savings + gateway_savings
    
    if full:
        view_label = "All-Time Statistics"
        mode_buttons = (
            '<button class="refresh-btn" style="background: var(--bg); color: var(--text); border: 1px solid var(--border); margin-right: 0.5rem;" '
            'onclick="window.location.href=\'/dashboard\'">24h View</button>'
            '<button class="refresh-btn" onclick="window.location.reload()">Refresh</button>'
        )
    else:
        view_label = "Last 24 Hours"
        mode_buttons = (
            '<button class="refresh-btn" style="background: var(--accent); color: white; margin-right: 0.5rem;" '
            'onclick="window.location.href=\'/dashboard?full=true\'">All-Time View</button>'
            '<button class="refresh-btn" onclick="window.location.reload()">Refresh</button>'
        )
    
    rows_html = ""
    for r in recent:
        badge = '<span class="badge badge-cached">CACHED</span>' if r["cache_read"] > 0 else '<span class="badge badge-miss">MISS</span>'
        rows_html += (
            f"<tr>"
            f"<td>{r['timestamp']}</td>"
            f'<td style="font-family: monospace; font-size: 0.75rem;">{r["model"]}</td>'
            f"<td>{badge}</td>"
            f"<td>{r['input']:,}</td>"
            f'<td style="color: #10b981;">{r["cache_read"]:,}</td>'
            f'<td style="color: #3b82f6;">{r["gateway_saved"]:,}</td>'
            f"<td>{r['output']:,}</td>"
            f"<td>{r['savings_pct']}%</td>"
            f"</tr>\n"
        )
    
    if not rows_html:
        rows_html = '<tr><td colspan="8" style="text-align: center; color: var(--text-dim);">No requests yet. Make some API calls to see data here.</td></tr>'
    
    html = DASHBOARD_HTML
    html = html.replace("{{VIEW_LABEL}}", view_label)
    html = html.replace("{{MODE_BUTTONS}}", mode_buttons)
    html = html.replace("{{TOTAL_INPUT}}", f"{total_input:,}")
    html = html.replace("{{TOTAL_CACHED}}", f"{cached:,}")
    html = html.replace("{{GATEWAY_SAVED}}", f"{gateway_saved:,}")
    html = html.replace("{{EFFICIENCY}}", f"{efficiency:.1f}%")
    html = html.replace("{{SAVINGS_USD}}", f"${savings_usd:.4f}")
    html = html.replace("{{REQUEST_COUNT}}", f"{request_count:,}")
    html = html.replace("{{RECENT_ROWS}}", rows_html)
    
    return html


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request, full: bool = False):
    from gateway.cache import rds
    
    if not rds:
        return HTMLResponse(content=NO_REDIS_HTML, status_code=200)
    
    stats = _get_stats_from_redis(full)
    recent = await _get_recent_requests(limit=20)
    html = _render_dashboard(stats, recent, full)
    
    return HTMLResponse(content=html)
