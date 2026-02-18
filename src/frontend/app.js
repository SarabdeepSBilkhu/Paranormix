/**
 * Paranormix - Diagnostic Suite Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const chatContainer = document.getElementById('chatContainer');
    const diagnosticDashboard = document.getElementById('diagnosticDashboard');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const resetBtn = document.getElementById('resetBtn');
    const dashboardTemplate = document.getElementById('dashboardTemplate');

    let sessionId = null;

    // --- Core Logic ---

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Is this the first message of the investigation?
        const isInitial = !sessionId;

        userInput.value = '';
        appendMessage('user', message);

        setLoading(true);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    user_message: message
                })
            });

            const data = await response.json();
            
            if (data.session_id) sessionId = data.session_id;

            if (isInitial && data.ml_data) {
                renderDashboard(data.ml_data);
            }

            appendMessage('ai', data.response);
        } catch (err) {
            console.error('Diagnostic error:', err);
            appendMessage('ai', 'Error: System failed to generate diagnostic report.');
        } finally {
            setLoading(false);
        }
    }

    // --- UI Rendering ---

    function renderDashboard(mlData) {
        // Clear and show dashboard
        diagnosticDashboard.innerHTML = '';
        diagnosticDashboard.classList.remove('dashboard-hidden');
        
        const report = dashboardTemplate.content.cloneNode(true);
        const charts = mlData.chart_data;

        // Header
        report.getElementById('dashPrediction').textContent = mlData.prediction;
        const certainty = report.getElementById('dashCertainty');
        certainty.textContent = `CERTAINTY: ${mlData.certainty}`;
        certainty.className = `certainty-pill cert-${mlData.certainty.toLowerCase()}`;

        // 1. Class Score Bar Chart
        const scoreContainer = report.getElementById('chartClassScore');
        Object.entries(charts.class_scores).forEach(([cls, score]) => {
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-label">${cls}</div>
                <div class="bar-outer"><div class="bar-inner" style="width: ${score * 100}%"></div></div>
            `;
            scoreContainer.appendChild(row);
        });

        // 2. Certainty Drivers
        const driverContainer = report.getElementById('chartDrivers');
        const driverMap = {
            "multi_class_overlap": "High Probability Overlap",
            "modifier_presence": "Interpretive Bias Detected",
            "signal_contradiction": "Conflicting Evidence Patterns"
        };
        Object.entries(charts.certainty_drivers).forEach(([key, active]) => {
            const item = document.createElement('div');
            item.className = `check-item ${active ? 'active' : ''}`;
            item.innerHTML = `
                <div class="check-box ${active ? 'checked' : ''}"></div>
                <span>${driverMap[key] || key}</span>
            `;
            driverContainer.appendChild(item);
        });

        // 3. Signal Contribution (Stacked Bar)
        const contribChart = report.getElementById('chartContribution');
        const contribs = charts.signal_contributions || {}; 
        const total = Object.values(contribs).reduce((a, b) => a + b, 0) || 1;
        
        Object.entries(contribs).forEach(([cat, val]) => {
            const p = (val / total) * 100;
            if (p > 0) {
                const seg = document.createElement('div');
                seg.className = `contrib-segment seg-${cat}`;
                seg.style.width = `${p}%`;
                seg.textContent = val > 0 ? cat[0].toUpperCase() : '';
                seg.title = `${cat}: ${val} signals`;
                contribChart.appendChild(seg);
            }
        });

        // 4. Competing Margin
        const marginContainer = report.getElementById('chartMargins');
        // Sort margins to show closest competitors
        const sortedMargins = Object.entries(charts.margins)
            .sort((a,b) => a[1] - b[1])
            .slice(0, 3);

        sortedMargins.forEach(([cls, gap]) => {
            const row = document.createElement('div');
            row.className = 'margin-row';
            // Gap of 0.1 becomes a 90% wide bar (inverted to show proximity)
            const gapWidth = Math.max(0, (1 - gap) * 100);
            row.innerHTML = `
                <div class="margin-label">${cls}</div>
                <div class="margin-bar-outer"><div class="margin-bar-inner" style="width: ${gapWidth}%"></div></div>
            `;
            marginContainer.appendChild(row);
        });

        // 5. Global Heatmap
        const heatContainer = report.getElementById('chartHeatmap');
        const cm = charts.global_cm;
        
        // Header labels
        heatContainer.appendChild(document.createElement('div')); // Empty corner
        cm.labels.forEach(l => {
            const cell = document.createElement('div');
            cell.className = 'heat-cell heat-label';
            cell.textContent = l.slice(0, 3);
            heatContainer.appendChild(cell);
        });

        cm.matrix.forEach((row, i) => {
            const label = document.createElement('div');
            label.className = 'heat-cell heat-label';
            label.textContent = cm.labels[i].slice(0, 3);
            heatContainer.appendChild(label);

            row.forEach(val => {
                const cell = document.createElement('div');
                cell.className = 'heat-cell';
                const opacity = Math.min(1, val/200);
                cell.style.background = `rgba(56, 189, 248, ${opacity})`;
                cell.textContent = val;
                heatContainer.appendChild(cell);
            });
        });

        diagnosticDashboard.appendChild(report);
    }

    function appendMessage(role, text) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        
        // Format bold text
        const formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        div.innerHTML = formatted;
        
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function setLoading(isLoading) {
        sendBtn.disabled = isLoading;
        userInput.disabled = isLoading;
        if (isLoading) {
            sendBtn.innerHTML = '<div class="loader"></div>';
        } else {
            sendBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>';
        }
    }

    function resetInvestigation() {
        sessionId = null;
        chatContainer.innerHTML = '';
        diagnosticDashboard.innerHTML = '';
        diagnosticDashboard.classList.add('dashboard-hidden');
        userInput.value = '';
        appendMessage('ai', 'Diagnostic Terminal active. Submit narrative for multi-axial analysis.');
    }

    // --- Listeners ---
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    resetBtn.addEventListener('click', resetInvestigation);

    // Welcome message
    appendMessage('ai', 'Diagnostic Terminal active. Submit narrative for multi-axial analysis.');
});
