document.addEventListener('DOMContentLoaded', () => {

    const chatContainer = document.getElementById('chatContainer');
    const diagnosticDashboard = document.getElementById('diagnosticDashboard');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const resetBtn = document.getElementById('resetBtn');
    const dashboardTemplate = document.getElementById('dashboardTemplate');
    const analystStatus = document.getElementById('analystStatus');

    let sessionId = null;

    async function sendMessage(manualText = null) {
        const message = manualText || userInput.value.trim();
        if (!message) return;

        const isInitial = !sessionId;

        appendMessage('user', message);
        userInput.value = '';
        setLoading(true);

        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: sessionId,
                    user_message: message
                })
            });

            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(errorText || `Status ${res.status}`);
            }

            const data = await res.json();

            if (data.session_id) sessionId = data.session_id;

            appendMessage('ai', data.response || data.detail || "No response");

            // Render dashboard ONLY on initial
            if (isInitial && data.ml_data) {
                renderDashboard(data.ml_data);
            }

        } catch (e) {
            appendMessage('ai', 'Error: ' + e.message);
        } finally {
            setLoading(false);
        }
    }

    function renderDashboard(data) {
        const mainGrid = document.getElementById('mainGrid');
        if (mainGrid) mainGrid.classList.add('axial-active');

        diagnosticDashboard.innerHTML = '';
        const report = dashboardTemplate.content.cloneNode(true);

        const classification = data.classification;
        const confidence = data.confidence;
        const band = data.confidence_band;

        // Prediction
        report.getElementById('dashPrediction').textContent = classification.toUpperCase();

        // Band
        const bandEl = report.getElementById('dashBand');
        bandEl.textContent = band;
        bandEl.className = `band-indicator band-${band.toLowerCase()}`;

        report.getElementById('dashConfidence').textContent =
            `Confidence: ${(confidence * 100).toFixed(0)}%`;

        // --- ML PROBS ---
        const mlBox = report.getElementById('mlProbs');
        const probs = data.ml_probs || {};

        Object.keys(probs).forEach(cls => {
            const val = probs[cls];

            const row = document.createElement('div');
            row.className = 'bar-row';

            row.innerHTML = `
                <div class="bar-lbl">
                    <span>${cls.toUpperCase()}</span>
                    <span>${(val * 100).toFixed(1)}%</span>
                </div>
                <div class="bar-out">
                    <div class="bar-in prob bar-${cls}" style="width:${val*100}%"></div>
                </div>
            `;

            mlBox.appendChild(row);
        });

        // --- EVIDENCE ---
        const evBox = report.getElementById('evidenceList');
        const evidence = data.evidence || {};

        Object.keys(evidence).forEach(cls => {
            if (evidence[cls].length > 0) {
                const div = document.createElement('div');
                div.className = 'rank-item';
                div.innerHTML = `
                    <span>${cls.toUpperCase()}</span>
                    <span>${evidence[cls].join(', ')}</span>
                `;
                evBox.appendChild(div);
            }
        });

        // --- IGNORED ---
        const igBox = report.getElementById('ignoredSignals');
        const ignored = data.ignored_signals || [];

        if (ignored.length === 0) {
            igBox.innerHTML = '<li>No weaker signals</li>';
        } else {
            ignored.forEach(s => {
                const li = document.createElement('li');
                li.textContent = s.toUpperCase();
                igBox.appendChild(li);
            });
        }

        // --- MODEL METRICS ---
        const metricsGrid = report.getElementById('modelMetricsGrid');
        const metrics = data.metrics || {};

        if (metricsGrid && Object.keys(metrics).length > 0) {
            const header = document.createElement('div');
            header.className = 'metrics-row header';
            header.innerHTML = '<span>CLS</span><span>PREC</span><span>RECL</span><span>F1</span>';
            metricsGrid.appendChild(header);

            ['material', 'environmental', 'immaterial', 'rule_bound', 'internal'].forEach(cls => {
                const m = metrics[cls] || {p:0, r:0, f1:0};
                const row = document.createElement('div');
                row.className = 'metrics-row';
                row.innerHTML = `
                    <span class="cls-lbl">${cls.slice(0, 4).toUpperCase()}</span>
                    <span>${m.p.toFixed(2)}</span>
                    <span>${m.r.toFixed(2)}</span>
                    <span class="f1-val">${m.f1.toFixed(2)}</span>
                `;
                metricsGrid.appendChild(row);
            });
        }

        diagnosticDashboard.appendChild(report);
    }

    function appendMessage(role, text) {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        // HARD SAFETY FIX
        const safeText = (text ?? "").toString();

        const formatted = safeText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*(?!\*)/g, '<em>$1</em>')
            .replace(/^\s*[\*]\s+(.*)/gm, '• $1')
            .replace(/\n/g, '<br>');

        div.innerHTML = formatted;
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function setLoading(state) {
        sendBtn.disabled = state;
        userInput.disabled = state;
        analystStatus.textContent = state ? 'Analyzing...' : 'Ready';
    }

    function reset() {
        sessionId = null;
        chatContainer.innerHTML = '';
        diagnosticDashboard.innerHTML = '';
        const mainGrid = document.getElementById('mainGrid');
        if (mainGrid) mainGrid.classList.remove('axial-active');
        appendMessage('ai', 'System reset.');
    }

    sendBtn.addEventListener('click', () => sendMessage());

    userInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    resetBtn.addEventListener('click', reset);

    appendMessage('ai', 'Enter narrative for analysis.');
});