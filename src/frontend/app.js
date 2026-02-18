/**
 * Paranormix - Diagnostic Suite Logic V2
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

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server Error: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.session_id) sessionId = data.session_id;

            if (isInitial && data.ml_data) {
                renderDashboard(data.ml_data);
            }

            appendMessage('ai', data.response || "NO_RESPONSE_RECEIVED");
        } catch (err) {
            console.error('Diagnostic error:', err);
            appendMessage('ai', `SYSTEM_ERROR: ${err.message || 'Failed to establish diagnostic link.'}`);
        } finally {
            setLoading(false);
        }
    }

    // --- UI Rendering ---

    function renderDashboard(mlData) {
        diagnosticDashboard.innerHTML = '';
        diagnosticDashboard.classList.remove('dashboard-hidden');
        
        const report = dashboardTemplate.content.cloneNode(true);
        const charts = mlData.chart_data;

        // Header
        report.getElementById('dashPrediction').textContent = mlData.prediction;
        const certainty = report.getElementById('dashCertainty');
        certainty.textContent = mlData.certainty;
        certainty.className = `cert-pill cert-${mlData.certainty.toLowerCase()}`;

        // 1. Class Score Bar Chart
        const scoreContainer = report.getElementById('chartClassScore');
        Object.entries(charts.class_scores).forEach(([cls, score]) => {
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-lbl">${cls.toUpperCase()}</div>
                <div class="bar-out"><div class="bar-in" style="width: ${score * 100}%"></div></div>
            `;
            scoreContainer.appendChild(row);
        });

        // 2. Certainty Drivers
        const driverContainer = report.getElementById('chartDrivers');
        const driverMap = {
            "multi_class_overlap": "PROBABILITY_OVERLAP",
            "resolution_boundary": "RESOLUTION_BOUNDARY",
            "signal_conflict": "SIGNAL_CONFLICT"
        };
        Object.entries(charts.certainty_drivers).forEach(([key, active]) => {
            const item = document.createElement('div');
            item.className = `chk-row ${active ? 'active' : ''}`;
            item.innerHTML = `
                <div class="chk-box ${active ? 'active' : ''}"></div>
                <span>${driverMap[key] || key.toUpperCase()}</span>
            `;
            driverContainer.appendChild(item);
        });

        // 3. Signal Contribution (Stacked Bar)
        const contribChart = report.getElementById('chartContribution');
        const contribs = charts.signal_contributions || {}; 
        const total = Object.values(contribs).reduce((a, b) => a + b, 0) || 1;
        
        const catMap = { 
            "Pattern_A": "Kinetic / Physical", 
            "Pattern_B": "Sensory / Temp", 
            "Pattern_C": "Cognitive / Information", 
            "Pattern_D": "Visual / Optical" 
        };
        Object.entries(contribs).forEach(([cat, val]) => {
            const p = (val / total) * 100;
            if (p > 0) {
                const seg = document.createElement('div');
                seg.className = `seg ${catMap[cat] || 'phys'}`;
                seg.style.width = `${p}%`;
                seg.title = `${cat}: ${val} signals`;
                contribChart.appendChild(seg);
            }
        });

        // 4. Competing Margin
        const marginContainer = report.getElementById('chartMargins');
        const sortedMargins = Object.entries(charts.margins)
            .sort((a,b) => a[1] - b[1])
            .slice(0, 3);

        sortedMargins.forEach(([cls, gap]) => {
            const row = document.createElement('div');
            row.className = 'bar-row'; 
            const gapWidth = Math.max(0, (1 - gap) * 100);
            row.innerHTML = `
                <div class="bar-lbl">${cls.toUpperCase()}</div>
                <div class="bar-out"><div class="bar-in" style="width: ${gapWidth}%; opacity: 0.5;"></div></div>
            `;
            marginContainer.appendChild(row);
        });

        // 5. Global Heatmap
        const heatContainer = report.getElementById('chartHeatmap');
        const cm = charts.global_cm;
        
        heatContainer.appendChild(document.createElement('div')); // Empty corner
        cm.labels.forEach(l => {
            const cell = document.createElement('div');
            cell.className = 'h-cell h-lbl';
            cell.textContent = l.slice(0, 3).toUpperCase();
            heatContainer.appendChild(cell);
        });

        cm.matrix.forEach((row, i) => {
            const label = document.createElement('div');
            label.className = 'h-cell h-lbl';
            label.textContent = cm.labels[i].slice(0, 3).toUpperCase();
            heatContainer.appendChild(label);

            row.forEach(val => {
                const cell = document.createElement('div');
                cell.className = 'h-cell';
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
        
        // Safety check for undefined/null text
        const safeText = String(text || "");
        
        // Format bold text
        const formatted = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
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
            sendBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>';
        }
    }

    function resetInvestigation() {
        sessionId = null;
        chatContainer.innerHTML = '';
        diagnosticDashboard.innerHTML = '';
        diagnosticDashboard.classList.add('dashboard-hidden');
        userInput.value = '';
        appendMessage('ai', 'INVESTIGATION_TERMINATED. Terminal ready for new signal input.');
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

    appendMessage(
        "ai",
        "Hello! I'm here to help you analyze your narrative. When you're ready, paste your story below and I'll begin the investigation."
    );
});
