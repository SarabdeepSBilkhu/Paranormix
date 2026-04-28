document.addEventListener('DOMContentLoaded', () => {

    const chatContainer = document.getElementById('chatContainer');
    const diagnosticDashboard = document.getElementById('diagnosticDashboard');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const resetBtn = document.getElementById('resetBtn');
    const dashboardTemplate = document.getElementById('dashboardTemplate');
    const analystStatus = document.getElementById('analystStatus');
    const suggestedPrompts = document.getElementById('suggestedPrompts');

    let sessionId = null;

    // ── Follow-up question helpers ──
    function parseFollowUps(rawText) {
        // Robust regex to find the marker, allowing for:
        // - Case insensitivity (i flag)
        // - Optional markdown bolding (**)
        // - Hyphens, underscores, or spaces
        // - Extra whitespace before/after colon
        const markerRegex = /\n?\s*(?:\*\*)?FOLLOW[-_ ]?UP[-_ ]?QUESTIONS\s*:(?:\*\*)?\s*/i;
        const match = rawText.match(markerRegex);

        console.debug('[Paranormix] Raw LLM response:', rawText);

        if (!match) {
            console.warn('[Paranormix] FOLLOW_UP_QUESTIONS block not found in response.');
            return { cleanText: rawText, questions: [] };
        }

        const idx = match.index;
        const cleanText = rawText.slice(0, idx).trimEnd();
        let jsonPart = rawText.slice(idx + match[0].length).trim();

        // Strip markdown code fences if the LLM wrapped the array
        jsonPart = jsonPart.replace(/^```[\w]*\n?/, '').replace(/```$/, '').trim();

        // Normalise curly/smart quotes to straight quotes
        jsonPart = jsonPart.replace(/[‘’]/g, "'").replace(/[“”]/g, '"');

        console.debug('[Paranormix] Parsed JSON part:', jsonPart);

        try {
            const questions = JSON.parse(jsonPart);
            return { cleanText, questions: Array.isArray(questions) ? questions : [] };
        } catch (e) {
            console.error('[Paranormix] Failed to parse follow-up questions:', e, '\nRaw:', jsonPart);
            // Return cleanText even if parsing fails to avoid showing raw block to user
            return { cleanText, questions: [] };
        }
    }

    function renderFollowUps(questions) {
        suggestedPrompts.innerHTML = '';
        if (!questions || questions.length === 0) return;

        const label = document.createElement('span');
        label.className = 'chip-label';
        label.textContent = 'Ask:';
        suggestedPrompts.appendChild(label);

        questions.forEach(q => {
            const chip = document.createElement('button');
            chip.className = 'prompt-chip';
            chip.textContent = q;
            chip.addEventListener('click', () => {
                userInput.value = q;
                sendMessage();
            });
            suggestedPrompts.appendChild(chip);
        });
    }

    async function sendMessage(manualText = null) {
        const message = manualText || userInput.value.trim();
        if (!message) return;

        const isInitial = !sessionId;

        appendMessage('user', message);
        userInput.value = '';
        suggestedPrompts.innerHTML = ''; // clear chips while waiting
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

            const rawResponse = data.response || data.detail || 'No response';
            const { cleanText, questions } = parseFollowUps(rawResponse);

            appendMessage('ai', cleanText);
            renderFollowUps(questions);

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
        
        console.log("Rendering Dashboard. ML Probs:", probs);

        const probKeys = Object.keys(probs);
        if (probKeys.length === 0) {
            mlBox.innerHTML = '<div class="text-muted" style="font-size:0.75rem; padding: 10px;">No probability data available.</div>';
        } else {
            probKeys.forEach(cls => {
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
        }

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

        // --- METADATA ---
        const wordCountEl = report.getElementById('dashWordCount');
        if (wordCountEl && data.metadata) {
            wordCountEl.textContent = data.metadata.words || 0;
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
        suggestedPrompts.innerHTML = '';
        const mainGrid = document.getElementById('mainGrid');
        if (mainGrid) mainGrid.classList.remove('axial-active');
        appendMessage('ai', 'Case file cleared. Submit a new account to begin.');
    }

    sendBtn.addEventListener('click', () => sendMessage());

    userInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    resetBtn.addEventListener('click', reset);

    appendMessage('ai', 'Case file open. Describe the occurrence, and I will take it from there.');
});