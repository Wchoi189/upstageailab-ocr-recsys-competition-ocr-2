        const vscode = acquireVsCodeApi();
        let currentData = { recentCalls: [], total: 0 };
        let settings = {
            autoRefresh: true,
            notifications: true,
            compactView: false,
            maxEvents: 100
        };

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab;
                switchTab(tabName);
            });
        });

        function switchTab(tabName) {
            // Update buttons
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

            // Update content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`tab-${tabName}`).classList.add('active');

            // Load tab-specific data
            if (tabName === 'tools') {
                updateToolsTab();
            } else if (tabName === 'policies') {
                updateViolationsTab();
                // updatePoliciesTab is handled by message updates, but we should ensure it renders
                // We'll rely on the existing data or trigger a render if we have it
                if (window.lastPolicies) {
                    updatePoliciesList(window.lastPolicies);
                }
            }
        }

        function refresh() {
            vscode.postMessage({ command: 'refresh' });
        }

        function formatTimeAgo(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const diffMs = now - date;
            const diffSec = Math.floor(diffMs / 1000);

            if (diffSec < 60) return diffSec + 's ago';
            if (diffSec < 3600) return Math.floor(diffSec / 60) + 'm ago';
            if (diffSec < 86400) return Math.floor(diffSec / 3600) + 'h ago';
            return date.toLocaleDateString();
        }

        function updateConnectionStatus(connected, error) {
            const indicator = document.getElementById('healthIndicator');
            const text = document.getElementById('healthText');
            const lastUpdate = document.getElementById('lastUpdate');

            if (connected) {
                indicator.className = 'health-indicator connected';
                text.textContent = 'Connected';
            } else {
                indicator.className = 'health-indicator disconnected';
                text.textContent = error || 'Disconnected';
            }

            lastUpdate.textContent = 'Last update: ' + new Date().toLocaleTimeString();
        }

        function updateDashboard(data) {
            currentData = data;

            document.getElementById('total').textContent = data.total;
            document.getElementById('success').textContent = data.success;
            document.getElementById('errors').textContent = data.errors;
            document.getElementById('violations').textContent = data.policyViolations;
            document.getElementById('avgDuration').textContent = data.avgDurationMs;

            // Session Tokens
            document.getElementById('sessionInput').textContent = data.session_input_tokens || 0;
            document.getElementById('sessionOutput').textContent = data.session_output_tokens || 0;
            document.getElementById('sessionTotal').textContent = data.session_total_tokens || 0;

            // Update last update time
            document.getElementById('lastUpdate').textContent = 'Last update: ' + new Date().toLocaleTimeString();

            // Violations panel
            const violationsSection = document.getElementById('violations-section');
            const violationsList = document.getElementById('violations-list');
            const violations = data.recentCalls.filter(c => c.status === 'policy_violation');

            if (violations.length > 0) {
                violationsSection.style.display = 'block';
                violationsList.innerHTML = violations.slice(0, 5).map(v =>
                    '<div class="violation-item"><strong class="tool-name">' + v.tool_name + '</strong>: ' +
                    (v.policy || 'Unknown policy') +
                    ' <span class="time-ago">' + formatTimeAgo(v.timestamp) + '</span></div>'
                ).join('');
            } else {
                violationsSection.style.display = 'none';
            }

            // Recent calls table
            const container = document.getElementById('calls-container');
            if (data.recentCalls.length === 0) {
                container.innerHTML = '<div class="empty-state">No telemetry data yet. Make some MCP tool calls!</div>';
                return;
            }

            container.innerHTML = '<table class="log-table">' +
                '<thead><tr><th>Tool</th><th>Status</th><th>Duration</th><th>Context/Metadata</th><th>Time</th></tr></thead>' +
                '<tbody>' +
                data.recentCalls.map((call, idx) => {
                    // Extract metadata badges
                    let metaBadges = '';
                    if (call.metadata) {
                        const interestingKeys = ['bundle_name', 'artifact_type', 'query', 'action'];
                        interestingKeys.forEach(k => {
                             if (call.metadata[k]) {
                                metaBadges += `<span class="badge info-badge" style="margin-right:4px; font-size:0.8em; padding:2px 6px; background:var(--bg-secondary); border-radius:4px" title="${k}: ${call.metadata[k]}">${k}: ${call.metadata[k]}</span>`;
                             }
                        });
                        // Fallback
                        if (!metaBadges && Object.keys(call.metadata).length > 0) {
                             const k = Object.keys(call.metadata)[0];
                             metaBadges += `<span class="badge info-badge" style="font-size:0.8em; padding:2px 6px; background:var(--bg-secondary); border-radius:4px">${k}: ${call.metadata[k]}</span>`;
                        }
                    }

                    return `<tr class="clickable-row" onclick="showCallDetails(${idx})">` +
                    '<td class="tool-name">' + call.tool_name + '</td>' +
                    '<td><span class="status-badge status-' + call.status + '">' + call.status + '</span></td>' +
                    '<td>' + (call.duration_ms ? call.duration_ms + 'ms' : '-') + '</td>' +
                    '<td>' + metaBadges + '</td>' +
                    '<td class="time-ago">' + formatTimeAgo(call.timestamp) + '</td>' +
                    '</tr>';
                }).join('') +
                '</tbody></table>';
        }

        function updateBundles(bundles) {
            const container = document.getElementById('bundles-container');
            if (!bundles || bundles.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📦</div><div>No bundles found</div></div>';
                return;
            }
            container.innerHTML = '<div class="bundles-grid">' +
                bundles.map(b =>
                    '<div class="bundle-card">' +
                    '<div class="bundle-name">📦 ' + b.name + '</div>' +
                    '<div class="bundle-desc">' + b.description + '</div>' +
                    '<div class="bundle-files">' + b.fileCount + ' files</div>' +
                    '</div>'
                ).join('') +
                '</div>';
        }

        // Tools Tab Analytics
        function updateToolsTab() {
            if (!currentData.recentCalls || currentData.recentCalls.length === 0) return;

            const toolStats = {};
            currentData.recentCalls.forEach(call => {
                if (!toolStats[call.tool_name]) {
                    toolStats[call.tool_name] = {
                        name: call.tool_name,
                        count: 0,
                        totalDuration: 0,
                        errors: 0,
                        successes: 0
                    };
                }
                const stats = toolStats[call.tool_name];
                stats.count++;
                if (call.duration_ms) stats.totalDuration += call.duration_ms;
                if (call.status === 'error') stats.errors++;
                if (call.status === 'success') stats.successes++;
            });

            const toolsArray = Object.values(toolStats);
            const container = document.getElementById('tools-grid');

            if (toolsArray.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🔧</div><div>No tool data available</div></div>';
                return;
            }

            container.innerHTML = toolsArray.map(tool => {
                const avgDuration = tool.totalDuration > 0 ? (tool.totalDuration / tool.count).toFixed(2) : 0;
                const successRate = ((tool.successes / tool.count) * 100).toFixed(0);
                return `
                    <div class="tool-card">
                        <div class="tool-header">
                            <strong class="tool-name">${tool.name}</strong>
                            <span class="tool-badge">${tool.count}</span>
                        </div>
                        <div class="tool-usage">
                            Success Rate: <strong>${successRate}%</strong><br>
                            Avg Duration: <strong>${avgDuration}ms</strong><br>
                            Errors: <span style="color: var(--error)">${tool.errors}</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Violations Tab
        function updateViolationsTab() {
            if (!currentData.recentCalls) return;

            const violations = currentData.recentCalls.filter(c => c.status === 'policy_violation');
            const container = document.getElementById('violations-detailed');

            if (violations.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">✓</div><div>No policy violations found</div></div>';
                return;
            }

            container.innerHTML = '<div class="violations-panel">' + violations.map(v =>
                `<div class="violation-item">
                    <strong class="tool-name">${v.tool_name}</strong>: ${v.policy || 'Unknown policy'}
                    <span class="time-ago"> - ${formatTimeAgo(v.timestamp)}</span>
                </div>`
            ).join('') + '</div>';
        }

        // Search and Filter
        document.getElementById('searchInput')?.addEventListener('input', (e) => {
            filterCalls(e.target.value, document.getElementById('statusFilter').value);
        });

        document.getElementById('statusFilter')?.addEventListener('change', (e) => {
            filterCalls(document.getElementById('searchInput').value, e.target.value);
        });

        function filterCalls(searchTerm, status) {
            if (!currentData.recentCalls) return;

            let filtered = currentData.recentCalls;

            if (searchTerm) {
                filtered = filtered.filter(call =>
                    call.tool_name.toLowerCase().includes(searchTerm.toLowerCase())
                );
            }

            if (status !== 'all') {
                filtered = filtered.filter(call => call.status === status);
            }

            renderCalls(filtered);
        }

        function renderCalls(calls) {
            const container = document.getElementById('calls-container');
            if (calls.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📋</div><div>No matching calls found</div></div>';
                return;
            }

            container.innerHTML = '<table class="log-table">' +
                '<thead><tr><th>Tool</th><th>Status</th><th>Duration</th><th>Context/Metadata</th><th>Time</th></tr></thead>' +
                '<tbody>' +
                calls.map((call, idx) => {
                    // Extract metadata badges
                    let metaBadges = '';
                    if (call.metadata) {
                        const interestingKeys = ['bundle_name', 'artifact_type', 'query', 'action'];
                        interestingKeys.forEach(k => {
                             if (call.metadata[k]) {
                                metaBadges += `<span class="badge info-badge" style="margin-right:4px; font-size:0.8em; padding:2px 6px; background:var(--bg-secondary); border-radius:4px" title="${k}: ${call.metadata[k]}">${k}: ${call.metadata[k]}</span>`;
                             }
                        });
                        // Fallback
                        if (!metaBadges && Object.keys(call.metadata).length > 0) {
                             const k = Object.keys(call.metadata)[0];
                             metaBadges += `<span class="badge info-badge" style="font-size:0.8em; padding:2px 6px; background:var(--bg-secondary); border-radius:4px">${k}: ${call.metadata[k]}</span>`;
                        }
                    }

                    return `<tr class="clickable-row" onclick="showCallDetails(${idx})">` +
                    '<td class="tool-name">' + call.tool_name + '</td>' +
                    '<td><span class="status-badge status-' + call.status + '">' + call.status + '</span></td>' +
                    '<td>' + (call.duration_ms ? call.duration_ms + 'ms' : '-') + '</td>' +
                    '<td>' + metaBadges + '</td>' +
                    '<td class="time-ago">' + formatTimeAgo(call.timestamp) + '</td>' +
                    '</tr>';
                }).join('') +
                '</tbody></table>';
        }

        // Modal functions
        function showCallDetails(index) {
            const calls = getFilteredCalls();
            const call = calls[index];
            if (!call) return;

            document.getElementById('modalTitle').textContent = `${call.tool_name} - Details`;
            document.getElementById('modalBody').innerHTML = `
                <div class="detail-group">
                    <div class="detail-label">Tool Name</div>
                    <div class="detail-value">${call.tool_name}</div>
                </div>
                <div class="detail-group">
                    <div class="detail-label">Status</div>
                    <div class="detail-value">
                        <span class="status-badge status-${call.status}">${call.status}</span>
                    </div>
                </div>
                <div class="detail-group">
                    <div class="detail-label">Timestamp</div>
                    <div class="detail-value">${new Date(call.timestamp).toLocaleString()}</div>
                </div>
                <div class="detail-group">
                    <div class="detail-label">Duration</div>
                    <div class="detail-value">${call.duration_ms ? call.duration_ms + ' ms' : 'N/A'}</div>
                </div>
                <div class="detail-group">
                    <div class="detail-label">Arguments Hash</div>
                    <div class="detail-value code">${call.args_hash}</div>
                </div>
                ${call.module ? `
                <div class="detail-group">
                    <div class="detail-label">Module</div>
                    <div class="detail-value code">${call.module}</div>
                </div>
                ` : ''}
                ${call.policy ? `
                <div class="detail-group">
                    <div class="detail-label">Policy Violated</div>
                    <div class="detail-value code">${call.policy}</div>
                </div>
                ` : ''}
                ${call.error ? `
                <div class="detail-group">
                    <div class="detail-label">Error Message</div>
                    <div class="detail-value code">${call.error}</div>
                </div>
                ` : ''}
            `;

            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        // Close modal on overlay click
        document.getElementById('modal').addEventListener('click', closeModal);

        // Close modal on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeModal();
            }
        });

        function getFilteredCalls() {
            const searchTerm = document.getElementById('searchInput')?.value || '';
            const status = document.getElementById('statusFilter')?.value || 'all';

            let filtered = currentData.recentCalls || [];

            if (searchTerm) {
                filtered = filtered.filter(call =>
                    call.tool_name.toLowerCase().includes(searchTerm.toLowerCase())
                );
            }

            if (status !== 'all') {
                filtered = filtered.filter(call => call.status === status);
            }

            return filtered;
        }

        // Settings
        function toggleSetting(element, setting) {
            element.classList.toggle('on');
            settings[setting] = element.classList.contains('on');
            console.log('Setting changed:', setting, settings[setting]);
        }

        function updateMaxEvents(value) {
            settings.maxEvents = parseInt(value);
            console.log('Max events set to:', value);
        }

        function clearData() {
            if (confirm('Are you sure you want to clear all cached data?')) {
                currentData = { recentCalls: [], total: 0 };
                updateDashboard({ total: 0, success: 0, errors: 0, policyViolations: 0, avgDurationMs: 0, recentCalls: [] });
            }
        }

        // Export functionality
        function exportData() {
            const dataStr = JSON.stringify(currentData.recentCalls, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `mcp-telemetry-${new Date().toISOString().split('T')[0]}.json`;
            link.click();
            URL.revokeObjectURL(url);
        }

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.command === 'update') {
                updateDashboard(message.data);
                updateConnectionStatus(true);
            } else if (message.command === 'updateBundles') {
                updateBundles(message.bundles);
            } else if (message.command === 'updatePolicies') {
                window.lastPolicies = message.policies;
                updatePoliciesList(message.policies);
            } else if (message.command === 'connectionStatus') {
                updateConnectionStatus(message.connected, message.error);
            }
        });

        function updatePoliciesList(policies) {
            const container = document.getElementById('policies-container');
            const searchTerm = document.getElementById('policySearch')?.value.toLowerCase() || '';

            if (!policies || policies.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📜</div><div>No policies found</div></div>';
                return;
            }

            const filtered = policies.filter(p =>
                p.name.toLowerCase().includes(searchTerm) ||
                p.category.toLowerCase().includes(searchTerm)
            );

            if (filtered.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📜</div><div>No matching policies</div></div>';
                return;
            }

            // Group by category
            const grouped = {};
            filtered.forEach(p => {
                if (!grouped[p.category]) grouped[p.category] = [];
                grouped[p.category].push(p);
            });

            let html = '<div class="policies-grid" style="display:grid;gap:16px;">';

            for (const [category, items] of Object.entries(grouped)) {
                html += `
                    <div class="policy-group">
                        <h3 style="margin-bottom:8px;font-size:0.9em;color:var(--text-muted);border-bottom:1px solid var(--border);padding-bottom:4px;">${category}</h3>
                        <div style="display:grid;gap:8px;">
                            ${items.map(p => `
                                <div class="stat-card" style="text-align:left;padding:12px;">
                                    <div style="font-weight:600;margin-bottom:4px;">${p.name}</div>
                                    <div style="font-size:0.8em;color:var(--text-muted);word-break:break-all;">${p.path}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }
            html += '</div>';
            container.innerHTML = html;
        }

        // Add search listener for policies
        document.getElementById('policySearch')?.addEventListener('input', () => {
            if (window.lastPolicies) updatePoliciesList(window.lastPolicies);
        });

        // Signal that the webview is ready to receive data
        vscode.postMessage({ command: 'webviewReady' });
