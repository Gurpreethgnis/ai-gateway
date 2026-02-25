from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from gateway.db import get_session, UsageRecord
from gateway import config
from gateway.routers.auth import get_current_user
import logging
from datetime import datetime, timedelta

router = APIRouter()
log = logging.getLogger("gateway")


async def require_dashboard_auth(request: Request):
    """Check dashboard authentication if enabled."""
    if not config.ENABLE_DASHBOARD_AUTH:
        return None
    
    user = await get_current_user(request)
    if not user:
        return None
    return user

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
        .logout-btn {
            background: transparent;
            color: var(--text-dim);
            border: 1px solid var(--border);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
            cursor: pointer;
            margin-left: 0.5rem;
            transition: all 0.2s;
        }
        .logout-btn:hover { border-color: var(--primary); color: var(--text); }
        .provider-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }
        .provider-card {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 0.75rem;
        }
        .provider-title {
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: capitalize;
            margin-bottom: 0.25rem;
        }
        .trace-list {
            display: grid;
            gap: 0.5rem;
            max-height: 220px;
            overflow-y: auto;
            margin-top: 0.75rem;
        }
        .trace-item {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
            color: var(--text-dim);
        }
        .warning-banner {
            margin-top: 0.75rem;
            padding: 0.5rem 0.75rem;
            border: 1px solid rgba(245, 158, 11, 0.4);
            background: rgba(245, 158, 11, 0.12);
            color: #fbbf24;
            border-radius: 0.5rem;
            font-size: 0.8rem;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>AI Gateway Insights</h1>
                <p style="color: var(--text-dim)">{{VIEW_LABEL}}</p>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                {{MODE_BUTTONS}}
                {{USER_INFO}}
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

        <!-- Routing Preferences Section -->
        <div id="routing-prefs" class="card" style="margin-bottom: 2rem; padding: 1.5rem;">
            <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">Routing Preferences</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div>
                    <label class="card-label">Cost ↔ Quality Trade-off</label>
                    <input type="range" id="cost-quality" min="0" max="100" value="30" 
                           style="width: 100%; accent-color: var(--primary);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-dim);">
                        <span>💰 Cheapest</span>
                        <span id="cost-quality-val">30%</span>
                        <span>⭐ Best Quality</span>
                    </div>
                </div>
                <div>
                    <label class="card-label">Speed ↔ Quality Trade-off</label>
                    <input type="range" id="speed-quality" min="0" max="100" value="50"
                           style="width: 100%; accent-color: var(--primary);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-dim);">
                        <span>⚡ Fastest</span>
                        <span id="speed-quality-val">50%</span>
                        <span>⭐ Best Quality</span>
                    </div>
                </div>
            </div>
            <div style="margin-top: 1.5rem; display: flex; align-items: center; gap: 1rem;">
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                    <input type="checkbox" id="cascade-enabled" checked style="accent-color: var(--primary);">
                    <span style="font-size: 0.875rem;">Enable automatic fallback (cascade)</span>
                </label>
                <button onclick="savePreferences()" class="refresh-btn">Save Preferences</button>
            </div>
            <div id="capability-warning" class="warning-banner"></div>
        </div>

        <div class="card" style="margin-bottom: 2rem; padding: 1.5rem;">
            <h2 style="font-size: 1.25rem; margin-bottom: 0.25rem;">Provider Status</h2>
            <p style="color: var(--text-dim); font-size: 0.85rem;">Live availability, capabilities, model count, and cache strategy per provider.</p>
            <div id="provider-grid" class="provider-grid">
                <div style="color: var(--text-dim);">Loading provider status...</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 2rem; padding: 1.5rem;">
            <h2 style="font-size: 1.25rem; margin-bottom: 0.25rem;">Routing Trace</h2>
            <p style="color: var(--text-dim); font-size: 0.85rem;">Recent routing decisions across local/cloud providers.</p>
            <div id="routing-trace" class="trace-list">
                <div style="color: var(--text-dim);">Loading routing trace...</div>
            </div>
        </div>
        <script>
            document.getElementById('cost-quality').addEventListener('input', function() {
                document.getElementById('cost-quality-val').textContent = this.value + '%';
                evaluateCapabilityWarning();
            });
            document.getElementById('speed-quality').addEventListener('input', function() {
                document.getElementById('speed-quality-val').textContent = this.value + '%';
                evaluateCapabilityWarning();
            });
            document.getElementById('cascade-enabled').addEventListener('change', function() {
                evaluateCapabilityWarning();
            });

            let dashboardModels = [];
            async function loadPreferences() {
                try {
                    const resp = await fetch('/api/preferences');
                    if (!resp.ok) return;
                    const prefs = await resp.json();
                    const costEl = document.getElementById('cost-quality');
                    const speedEl = document.getElementById('speed-quality');
                    const cascadeEl = document.getElementById('cascade-enabled');

                    if (typeof prefs.cost_quality_bias === 'number') {
                        costEl.value = Math.round(prefs.cost_quality_bias * 100);
                        document.getElementById('cost-quality-val').textContent = costEl.value + '%';
                    }
                    if (typeof prefs.speed_quality_bias === 'number') {
                        speedEl.value = Math.round(prefs.speed_quality_bias * 100);
                        document.getElementById('speed-quality-val').textContent = speedEl.value + '%';
                    }
                    if (typeof prefs.cascade_enabled === 'boolean') {
                        cascadeEl.checked = prefs.cascade_enabled;
                    }

                    evaluateCapabilityWarning();
                } catch (e) { console.warn('Load preferences failed:', e); }
            }

            async function loadProviderSummary() {
                const container = document.getElementById('provider-grid');
                try {
                    const resp = await fetch('/api/providers/summary');
                    if (!resp.ok) {
                        container.innerHTML = '<div style="color: var(--text-dim);">Provider summary unavailable.</div>';
                        return;
                    }
                    const data = await resp.json();
                    const providers = data.providers || [];
                    if (providers.length === 0) {
                        container.innerHTML = '<div style="color: var(--text-dim);">No provider data.</div>';
                        return;
                    }

                    container.innerHTML = providers.map(p => {
                        const availBadge = p.available ? '<span class="badge badge-cached">UP</span>' : '<span class="badge badge-miss">DOWN</span>';
                        return `<div class="provider-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                                <div class="provider-title">${p.provider}</div>
                                ${availBadge}
                            </div>
                            <div style="font-size:0.78rem; color: var(--text-dim);">Configured: ${p.configured ? 'Yes' : 'No'}</div>
                            <div style="font-size:0.78rem; color: var(--text-dim);">Models: ${p.model_count}</div>
                            <div style="font-size:0.78rem; color: var(--text-dim);">Cache: ${p.cache_strategy}</div>
                            <div style="font-size:0.78rem; color: var(--text-dim);">Tools: ${p.supports_tools ? 'Yes' : 'No'} | Vision: ${p.supports_vision ? 'Yes' : 'No'}</div>
                        </div>`;
                    }).join('');
                } catch (e) {
                    container.innerHTML = '<div style="color: var(--text-dim);">Provider summary unavailable.</div>';
                }
            }

            async function loadRoutingTrace() {
                const container = document.getElementById('routing-trace');
                try {
                    const resp = await fetch('/api/routing/trace?limit=20');
                    if (!resp.ok) {
                        container.innerHTML = '<div style="color: var(--text-dim);">Routing trace unavailable.</div>';
                        return;
                    }
                    const data = await resp.json();
                    const events = data.events || [];
                    if (events.length === 0) {
                        container.innerHTML = '<div style="color: var(--text-dim);">No routing events yet.</div>';
                        return;
                    }

                    container.innerHTML = events.map(e => {
                        const ts = e.timestamp ? new Date(e.timestamp * 1000).toLocaleTimeString() : '--:--:--';
                        const escalated = e.escalated ? ' | escalated' : '';
                        const reason = e.escalation_reason ? ` | ${e.escalation_reason}` : '';
                        return `<div class="trace-item">${ts} — ${e.provider || 'unknown'} / ${e.model || 'unknown'}${escalated}${reason}</div>`;
                    }).join('');
                } catch (e) {
                    container.innerHTML = '<div style="color: var(--text-dim);">Routing trace unavailable.</div>';
                }
            }

            function evaluateCapabilityWarning() {
                const warning = document.getElementById('capability-warning');
                if (!warning) return;

                const enabledModels = (dashboardModels || []).filter(m => m.is_enabled);
                const anyToolsCapable = enabledModels.some(m => (m.capabilities || []).includes('tools'));
                if (!anyToolsCapable) {
                    warning.style.display = 'block';
                    warning.textContent = 'No enabled models currently support tools. Tool-heavy requests may fail or degrade.';
                    return;
                }

                warning.style.display = 'none';
                warning.textContent = '';
            }

>>>>>>> 6869e71 (Close stage gaps and provider routing)
            async function savePreferences() {
                const prefs = {
                    cost_quality_bias: document.getElementById('cost-quality').value / 100,
                    speed_quality_bias: document.getElementById('speed-quality').value / 100,
                    cascade_enabled: document.getElementById('cascade-enabled').checked,
                    max_cascade_attempts: 3
                };
                try {
                    const resp = await fetch('/api/preferences', {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(prefs)
                    });
                    if (resp.ok) alert('Preferences saved!');
                    else alert('Failed to save preferences');
                } catch (e) { alert('Error: ' + e.message); }
            }
        </script>

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

        <!-- Model Manager Section -->
        <div id="model-manager" class="card" style="margin-top: 2rem; padding: 1.5rem;">
            <h2 style="font-size: 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>🤖</span> Model Manager
            </h2>
            <p style="color: var(--text-dim); margin-bottom: 1rem; font-size: 0.875rem;">
                Enable or disable models for routing. Disabled models won't be selected by the smart router.
            </p>
            <div id="model-list" style="display: grid; gap: 0.75rem;">
                <div style="text-align: center; color: var(--text-dim); padding: 2rem;">Loading models...</div>
            </div>
        </div>

        <!-- Ollama Controls Section -->
        <div id="ollama-controls" class="card" style="margin-top: 2rem; padding: 1.5rem;">
            <h2 style="font-size: 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>🦙</span> Ollama Controls
            </h2>
            <p style="color: var(--text-dim); margin-bottom: 1rem; font-size: 0.875rem;">
                Manage local Ollama models. Pull new models or delete existing ones.
            </p>
            
            <!-- Pull Model Form -->
            <div style="display: flex; gap: 0.5rem; margin-bottom: 1.5rem;">
                <input type="text" id="ollama-model-name" placeholder="e.g., llama3.2, codellama:7b, mistral"
                       style="flex: 1; padding: 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: 0.5rem; color: var(--text);">
                <button onclick="pullOllamaModel()" class="refresh-btn">Pull Model</button>
            </div>
            
            <!-- Progress Bar -->
            <div id="ollama-progress" style="display: none; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-dim); margin-bottom: 0.25rem;">
                    <span id="ollama-progress-text">Downloading...</span>
                    <span id="ollama-progress-pct">0%</span>
                </div>
                <div style="background: var(--bg); border-radius: 0.5rem; height: 8px; overflow: hidden;">
                    <div id="ollama-progress-bar" style="background: var(--primary); height: 100%; width: 0%; transition: width 0.3s;"></div>
                </div>
            </div>
            
            <!-- Local Models List -->
            <h3 style="font-size: 1rem; margin-bottom: 0.75rem; color: var(--text-dim);">Local Models</h3>
            <div id="ollama-models" style="display: grid; gap: 0.5rem;">
                <div style="text-align: center; color: var(--text-dim); padding: 1rem;">Loading Ollama models...</div>
            </div>
        </div>

        <!-- Model Manager Script -->
        <script>
            // Logout function
            async function logout() {
                try {
                    await fetch('/auth/logout', { method: 'POST' });
                    window.location.href = '/auth/login';
                } catch (e) {
                    window.location.href = '/auth/login';
                }
            }
            
            // Load models and preferences on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadPreferences();
                loadModels();
                loadOllamaModels();
                loadProviderSummary();
                loadRoutingTrace();
                setInterval(loadProviderSummary, 15000);
                setInterval(loadRoutingTrace, 5000);
            });
            
            async function loadModels() {
                const container = document.getElementById('model-list');
                try {
                    const resp = await fetch('/api/models');
                    if (!resp.ok) {
                        container.innerHTML = '<div style="color: var(--error);">Failed to load models. API returned: ' + resp.status + '</div>';
                        return;
                    }
                    const models = await resp.json();
                    dashboardModels = models || [];
                    
                    if (!models || models.length === 0) {
                        container.innerHTML = '<div style="color: var(--text-dim);">No models configured.</div>';
                        return;
                    }
                    
                    // Group by provider
                    const byProvider = {};
                    models.forEach(m => {
                        if (!byProvider[m.provider]) byProvider[m.provider] = [];
                        byProvider[m.provider].push(m);
                    });
                    
                    let html = '';
                    for (const [provider, providerModels] of Object.entries(byProvider)) {
                        html += `<div style="margin-bottom: 1rem;">
                            <div style="font-weight: 600; color: var(--primary); margin-bottom: 0.5rem; text-transform: capitalize;">${provider}</div>`;
                        providerModels.forEach(m => {
                            const checked = m.is_enabled ? 'checked' : '';
                            const cost = m.cost_per_1k_input ? `$${m.cost_per_1k_input.toFixed(4)}/1K` : 'Free';
                            html += `
                            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem; background: var(--bg); border-radius: 0.5rem; margin-bottom: 0.25rem;">
                                <div style="display: flex; align-items: center; gap: 0.75rem;">
                                    <label class="toggle-switch">
                                        <input type="checkbox" ${checked} onchange="toggleModel('${m.id}', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <div>
                                        <div style="font-weight: 500; font-size: 0.875rem;">${m.display_name || m.id}</div>
                                        <div style="font-size: 0.75rem; color: var(--text-dim);">Quality: ${(m.quality_rating * 100).toFixed(0)}% | ${cost}</div>
                                    </div>
                                </div>
                            </div>`;
                        });
                        html += '</div>';
                    }
                    container.innerHTML = html;
                    evaluateCapabilityWarning();
                } catch (e) {
                    container.innerHTML = '<div style="color: var(--error);">Error loading models: ' + e.message + '</div>';
                }
            }
            
            async function toggleModel(modelId, enabled) {
                try {
                    const resp = await fetch(`/api/models/${encodeURIComponent(modelId)}/enabled`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ is_enabled: enabled })
                    });
                    if (!resp.ok) {
                        alert('Failed to update model');
                        loadModels(); // Reload to reset state
                    }
                } catch (e) {
                    alert('Error: ' + e.message);
                    loadModels();
                }
            }
            
            async function loadOllamaModels() {
                const container = document.getElementById('ollama-models');
                try {
                    const resp = await fetch('/api/ollama/models');
                    if (!resp.ok) {
                        if (resp.status === 503) {
                            container.innerHTML = '<div style="color: var(--text-dim);">Ollama not available. Start Ollama to manage local models.</div>';
                        } else {
                            container.innerHTML = '<div style="color: var(--error);">Failed to load Ollama models.</div>';
                        }
                        return;
                    }
                    const data = await resp.json();
                    const models = data.models || [];
                    
                    if (models.length === 0) {
                        container.innerHTML = '<div style="color: var(--text-dim);">No local Ollama models. Pull a model to get started.</div>';
                        return;
                    }
                    
                    let html = '';
                    models.forEach(m => {
                        const size = m.size ? `${(m.size / 1e9).toFixed(1)} GB` : 'Unknown';
                        html += `
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem; background: var(--bg); border-radius: 0.5rem;">
                            <div>
                                <div style="font-weight: 500; font-size: 0.875rem;">${m.name}</div>
                                <div style="font-size: 0.75rem; color: var(--text-dim);">Size: ${size}</div>
                            </div>
                            <button onclick="deleteOllamaModel('${m.name}')" style="background: rgba(239, 68, 68, 0.2); color: #ef4444; border: none; padding: 0.375rem 0.75rem; border-radius: 0.375rem; font-size: 0.75rem; cursor: pointer;">
                                Delete
                            </button>
                        </div>`;
                    });
                    container.innerHTML = html;
                } catch (e) {
                    container.innerHTML = '<div style="color: var(--text-dim);">Ollama not available.</div>';
                }
            }
            
            let pullInProgress = false;
            
            async function pullOllamaModel() {
                if (pullInProgress) return;
                
                const input = document.getElementById('ollama-model-name');
                const modelName = input.value.trim();
                if (!modelName) {
                    alert('Please enter a model name');
                    return;
                }
                
                pullInProgress = true;
                const progress = document.getElementById('ollama-progress');
                const progressBar = document.getElementById('ollama-progress-bar');
                const progressText = document.getElementById('ollama-progress-text');
                const progressPct = document.getElementById('ollama-progress-pct');
                
                progress.style.display = 'block';
                progressText.textContent = 'Starting download...';
                progressPct.textContent = '0%';
                progressBar.style.width = '0%';
                
                try {
                    const resp = await fetch('/api/ollama/pull', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model_name: modelName })
                    });
                    
                    if (resp.ok) {
                        progressText.textContent = 'Pull started!';
                        progressPct.textContent = '...';
                        
                        // Poll for progress
                        let attempts = 0;
                        const pollInterval = setInterval(async () => {
                            attempts++;
                            try {
                                const statusResp = await fetch('/api/ollama/models');
                                if (statusResp.ok) {
                                    const data = await statusResp.json();
                                    const found = (data.models || []).find(m => m.name.startsWith(modelName.split(':')[0]));
                                    if (found) {
                                        clearInterval(pollInterval);
                                        progressText.textContent = 'Complete!';
                                        progressPct.textContent = '100%';
                                        progressBar.style.width = '100%';
                                        setTimeout(() => {
                                            progress.style.display = 'none';
                                            input.value = '';
                                            loadOllamaModels();
                                        }, 1500);
                                        pullInProgress = false;
                                        return;
                                    }
                                }
                            } catch (e) {}
                            
                            // Update progress animation
                            const pct = Math.min(90, attempts * 5);
                            progressBar.style.width = pct + '%';
                            progressPct.textContent = pct + '%';
                            progressText.textContent = 'Downloading ' + modelName + '...';
                            
                            if (attempts > 120) { // 2 min timeout
                                clearInterval(pollInterval);
                                progressText.textContent = 'Taking longer than expected...';
                                pullInProgress = false;
                            }
                        }, 1000);
                    } else {
                        const err = await resp.json();
                        progressText.textContent = 'Failed: ' + (err.detail || 'Unknown error');
                        progressBar.style.width = '0%';
                        progressBar.style.background = '#ef4444';
                        pullInProgress = false;
                    }
                } catch (e) {
                    progressText.textContent = 'Error: ' + e.message;
                    pullInProgress = false;
                }
            }
            
            async function deleteOllamaModel(name) {
                if (!confirm(`Delete model "${name}"? This cannot be undone.`)) return;
                
                try {
                    const resp = await fetch(`/api/ollama/models/${encodeURIComponent(name)}`, {
                        method: 'DELETE'
                    });
                    if (resp.ok) {
                        loadOllamaModels();
                    } else {
                        const err = await resp.json();
                        alert('Failed to delete: ' + (err.detail || 'Unknown error'));
                    }
                } catch (e) {
                    alert('Error: ' + e.message);
                }
            }
        </script>
        
        <!-- Toggle Switch Styles -->
        <style>
            .toggle-switch {
                position: relative;
                display: inline-block;
                width: 40px;
                height: 22px;
            }
            .toggle-switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            .toggle-slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: var(--border);
                transition: 0.3s;
                border-radius: 22px;
            }
            .toggle-slider:before {
                position: absolute;
                content: "";
                height: 16px;
                width: 16px;
                left: 3px;
                bottom: 3px;
                background-color: var(--text);
                transition: 0.3s;
                border-radius: 50%;
            }
            .toggle-switch input:checked + .toggle-slider {
                background-color: var(--success);
            }
            .toggle-switch input:checked + .toggle-slider:before {
                transform: translateX(18px);
            }
            .error { color: #ef4444; }
        </style>
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
    """Fetch recent requests from DB with strict timeout. Returns empty list on any issue."""
    import asyncio
    from gateway.db import async_session_factory, db_ready
    
    if not async_session_factory or not db_ready:
        return []
    
    async def _query():
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
    
    timeout_s = 15.0
    try:
        import os
        timeout_s = float(os.getenv("DASHBOARD_RECENT_REQUESTS_TIMEOUT", "15"))
    except (ValueError, TypeError):
        pass
    try:
        return await asyncio.wait_for(_query(), timeout=timeout_s)
    except asyncio.TimeoutError:
        log.warning("Recent requests query timed out (%.0fs) - skipping table", timeout_s)
        return []
    except Exception as e:
        log.warning("Failed to fetch recent requests: %r", e)
        return []


def _render_dashboard(stats: dict, recent: list[dict], full: bool, user: dict = None) -> str:
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
    
    # User info for header when auth is enabled
    if user and config.ENABLE_DASHBOARD_AUTH:
        user_email = user.get("email", "User")
        user_info = f'<span style="color: var(--text-dim); font-size: 0.875rem;">{user_email}</span><button class="logout-btn" onclick="logout()">Logout</button>'
    else:
        user_info = ""
    
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
    html = html.replace("{{USER_INFO}}", user_info)
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
    
    # Check auth if enabled
    user = None
    if config.ENABLE_DASHBOARD_AUTH:
        user = await get_current_user(request)
        if not user:
            return RedirectResponse(url="/auth/login", status_code=302)
    
    if not rds:
        return HTMLResponse(content=NO_REDIS_HTML, status_code=200)
    
    stats = _get_stats_from_redis(full)
    recent = await _get_recent_requests(limit=20)
    html = _render_dashboard(stats, recent, full, user)
    
    return HTMLResponse(content=html)
