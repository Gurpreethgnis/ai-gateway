from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func
from gateway.db import get_session, UsageRecord, Project
from gateway.security import extract_gateway_api_key
import json
from datetime import datetime, timedelta

router = APIRouter()

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

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding: 2rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

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
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s;
        }

        .card:hover {
            transform: translateY(-2px);
        }

        .card-label {
            font-size: 0.875rem;
            color: var(--text-dim);
            font-weight: 500;
            margin-bottom: 0.5rem;
            display: block;
        }

        .card-value {
            font-size: 2rem;
            font-weight: 700;
        }

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
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        thead {
            background: rgba(255, 255, 255, 0.05);
        }

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

        .badge-cached {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
        }

        .badge-miss {
            background: rgba(148, 163, 184, 0.2);
            color: #cbd5e1;
        }

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

        .refresh-btn:hover {
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>AI Gateway Insights</h1>
                <p style="color: var(--text-dim)">Real-time Claude token savings dashboard</p>
            </div>
            <button class="refresh-btn" onclick="window.location.reload()">Refresh Data</button>
        </header>

        <div class="stats-grid">
            <div class="card">
                <span class="card-label">Total Input Tokens</span>
                <div class="card-value">{{ total_input }}</div>
                <div class="card-sub">{{ total_cached }} Cached</div>
            </div>
            <div class="card">
                <span class="card-label">Efficiency Ratio</span>
                <div class="card-value">{{ efficiency }}%</div>
                <div class="card-sub">Tokens Saved</div>
            </div>
            <div class="card">
                <span class="card-label">Estimated Savings</span>
                <div class="card-value">${{ savings_usd }}</div>
                <div class="card-sub">USD Predicted</div>
            </div>
            <div class="card">
                <span class="card-label">Active Connections</span>
                <div class="card-value">{{ active_connections }}</div>
                <div class="card-sub">In Redis Queue</div>
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
                        <th>Output</th>
                        <th>Savings</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in recent %}
                    <tr>
                        <td>{{ row.timestamp }}</td>
                        <td style="font-family: monospace; font-size: 0.75rem;">{{ row.model }}</td>
                        <td>
                            {% if row.cache_read > 0 %}
                            <span class="badge badge-cached">CACHED</span>
                            {% else %}
                            <span class="badge badge-miss">MISS</span>
                            {% endif %}
                        </td>
                        <td>{{ row.input }}</td>
                        <td style="color: var(--primary)">{{ row.cache_read }}</td>
                        <td>{{ row.output }}</td>
                        <td>{{ row.savings_pct }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    from gateway.db import async_session_factory
    if async_session_factory is None:
        return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Gateway — Setup Required</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .box { background: #1e293b; border: 1px solid #334155; border-radius: 1rem; padding: 2.5rem; max-width: 520px; text-align: center; }
        h1 { font-size: 1.5rem; margin-bottom: 0.5rem; color: #38bdf8; }
        p { color: #94a3b8; line-height: 1.6; }
        .steps { text-align: left; background: #0f172a; border-radius: 0.75rem; padding: 1.25rem 1.5rem; margin-top: 1.5rem; }
        .steps li { color: #cbd5e1; margin: 0.5rem 0; }
        code { background: #334155; padding: 0.1rem 0.4rem; border-radius: 0.3rem; font-size: 0.875rem; color: #38bdf8; }
    </style>
</head>
<body>
    <div class="box">
        <h1>📊 Database Not Connected</h1>
        <p>The dashboard needs a PostgreSQL database to store and display your token savings.</p>
        <div class="steps">
            <ol>
                <li>In Railway, click <strong>+ New → Database → PostgreSQL</strong></li>
                <li>Railway will auto-add <code>DATABASE_URL</code> to your service</li>
                <li>Your gateway will restart and begin tracking stats</li>
                <li>Come back here to see your token savings! 🎉</li>
            </ol>
        </div>
    </div>
</body>
</html>
""", status_code=200)

    try:
        async with get_session() as session:
            # Aggregates
            try:
                total_q = await session.execute(
                    select(
                        func.sum(UsageRecord.input_tokens),
                        func.sum(UsageRecord.cost_usd)
                    )
                )
                result = total_q.fetchone()
                if result:
                    total_input, total_cost = result
                else:
                    total_input, total_cost = 0, 0
            except Exception as e:
                import logging
                logging.getLogger("gateway").error("Dashboard aggregate query failed: %r", e)
                total_input, total_cost = 0, 0

            total_input = total_input or 0
            total_cost = total_cost or 0
            # Calculate cache efficiency and savings
            cache_result = await session.execute(
                select(func.sum(UsageRecord.cache_read_input_tokens))
            )
            total_cached = cache_result.scalar() or 0
            
            total_processed = total_input + total_cached
            efficiency = (total_cached / total_processed * 100) if total_processed > 0 else 0
            
            # Estimate savings: cached tokens cost less to process
            cache_cost_savings = total_cached * 0.001 * 0.003  # rough estimate
            savings_usd = cache_cost_savings
            
            active_connections = 0
            try:
                from gateway.cache import rds
                if rds:
                    active_connections = rds.zcard("concurrency:anthropic:sonnet") or 0
            except:
                pass

            # Recent records
            processed_recent = []
            try:
                recent_q = await session.execute(
                    select(UsageRecord).order_by(UsageRecord.timestamp.desc()).limit(20)
                )
                recent_rows = recent_q.scalars().all()
                for r in recent_rows:
                    total_in = r.input_tokens or 0
                    cache_r = r.cache_read_input_tokens or 0
                    total_tokens = total_in + cache_r
                    savings_pct = (cache_r / total_tokens * 100) if total_tokens > 0 else 0
                    
                    ts = "00:00:00"
                    if r.timestamp:
                        ts = r.timestamp.strftime("%H:%M:%S")

                    processed_recent.append({
                        "timestamp": ts,
                        "model": (r.model or "unknown").replace("claude-3-5-", ""),
                        "input": total_in,
                        "cache_read": cache_r,
                        "output": r.output_tokens or 0,
                        "savings_pct": savings_pct
                    })
            except Exception as e:
                import logging
                logging.getLogger("gateway").error("Dashboard recent rows query failed: %r", e)

        html = DASHBOARD_HTML
        html = html.replace("{{ total_input }}", f"{total_input:,}")
        html = html.replace("{{ total_cached }}", f"{total_cached:,}")
        html = html.replace("{{ efficiency }}", str(efficiency))
        html = html.replace("{{ savings_usd }}", str(savings_usd))
        html = html.replace("{{ active_connections }}", str(active_connections))
        
        rows_html = ""
        for r in processed_recent:
            badge = '<span class="badge badge-cached">CACHED</span>' if r["cache_read"] > 0 else '<span class="badge badge-miss">MISS</span>'
            rows_html += f"""
            <tr>
                <td>{r['timestamp']}</td>
                <td style="font-family: monospace; font-size: 0.75rem;">{r['model']}</td>
                <td>{badge}</td>
                <td>{r['input']}</td>
                <td style="color: var(--primary)">{r['cache_read']}</td>
                <td>{r['output']}</td>
                <td>{r['savings_pct']}%</td>
            </tr>
            """
        
        parts_for = html.split('{% for row in recent %}')
        parts_end = html.split('{% endfor %}')
        
        if len(parts_for) > 1 and len(parts_end) > 1:
            final_html = parts_for[0] + rows_html + parts_end[1]
        else:
            final_html = html # Fallback if split fails
            
        return final_html

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        import logging
        logging.getLogger("gateway").error("CRITICAL DASHBOARD ERROR: %s", err_msg)
        return HTMLResponse(
            content=f"<html><body><h1>Dashboard Error</h1><pre>{err_msg}</pre></body></html>",
            status_code=500
        )
