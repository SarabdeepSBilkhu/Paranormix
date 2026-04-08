/**
 * Paranormix — XAI Diagnostic Suite V4
 * Version: 4.1.2 — Deterministic Classification Terminal
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log("XAI Terminal V4.1.2 Initializing...");

    // Structural Elements
    const chatContainer = document.getElementById('chatContainer');
    const diagnosticDashboard = document.getElementById('diagnosticDashboard');
    const metricPanel = document.getElementById('metricPanel');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const resetBtn = document.getElementById('resetBtn');
    const dashboardTemplate = document.getElementById('dashboardTemplate');
    const suggestedPrompts = document.getElementById('suggestedPrompts');
    
    // Metadata Elements
    const sessionHeader = document.getElementById('sessionHeader');
    const sessionTimestamp = document.getElementById('sessionTimestamp');
    const analystStatus = document.getElementById('analystStatus');

    let sessionId = null;

    // --- Core Interaction ---

    async function sendMessage(manualText = null) {
        const message = manualText || userInput.value.trim();
        if (!message) return;

        const isInitial = !sessionId;

        userInput.value = '';
        appendMessage('user', message);
        setLoading(true);
        clearPrompts();

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
                throw new Error(errData.detail || `SYSTEM_FAILURE: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.session_id) sessionId = data.session_id;

            if (isInitial && data.ml_data) {
                try {
                    renderDashboard(data.ml_data);
                    updateSessionMetadata(data.ml_data);
                    renderPrompts(data.ml_data);
                } catch (renderError) {
                    console.error("Layout Rendering Failure:", renderError);
                    console.log("Faulty mlData state:", JSON.stringify(data.ml_data));
                    throw new Error("UI Component Error during axial mapping. Check console for trace.");
                }
            }

            appendMessage('ai', data.response);
        } catch (err) {
            console.error('Diagnostic error:', err);
            appendMessage('ai', `CRITICAL_SYSTEM_ERROR: ${err.message}`);
        } finally {
            setLoading(false);
        }
    }

    // --- XAI Dashboard Rendering ---

    function renderDashboard(mlData) {
        if (!mlData) {
            console.error("Dashboard invoked with null mlData.");
            return;
        }

        const mainGrid = document.getElementById('mainGrid');
        if (mainGrid) mainGrid.classList.add('axial-active');
        
        diagnosticDashboard.innerHTML = '';
        
        const report = dashboardTemplate.content.cloneNode(true);

        // 1. Core Output: Classification & Confidence (Strict Safety Wrappers)
        const classification = String(mlData.classification || 'internal');
        const confidenceBand = String(mlData.confidence_band || 'Low');
        const confidenceScore = Number(mlData.confidence || 0);

        const predictionElem = report.getElementById('dashPrediction');
        if (predictionElem) predictionElem.textContent = classification.toUpperCase();
        
        const bandElem = report.getElementById('dashBand');
        if (bandElem) {
            bandElem.textContent = confidenceBand.toUpperCase();
            bandElem.className = `band-indicator band-${confidenceBand.toLowerCase()}`;
        }

        const confidenceElem = report.getElementById('dashConfidence');
        if (confidenceElem) {
            confidenceElem.textContent = 
                `Confidence: ${(confidenceScore * 100).toFixed(0)}% — Evidence-based (deterministic)`;
        }

        // 2. Signal Flags (boolean indicators)
        const flagContainer = report.getElementById('signalFlags');
        const signals = mlData.signals || {};
        const precedenceOrder = ['material', 'environmental', 'immaterial', 'rule_bound', 'internal'];
        
        precedenceOrder.forEach(cls => {
            const clsString = String(cls);
            const detected = signals[clsString] === true;
            const isWinner = clsString === classification;
            const isOverridden = detected && !isWinner;
            
            const row = document.createElement('div');
            row.className = `bar-row ${isOverridden ? 'dimmed' : ''}`;
            row.innerHTML = `
                <div class="bar-lbl">${clsString.toUpperCase()}${isWinner ? ' ★' : ''} ${isOverridden ? '<span class="label-overridden">(overridden)</span>' : ''}</div>
                <div class="bar-out">
                    <div class="bar-in" style="width: ${detected ? 100 : 0}%; background: ${isWinner ? 'var(--band-high)' : detected ? 'var(--band-moderate)' : 'transparent'}"></div>
                </div>
            `;
            if (flagContainer) flagContainer.appendChild(row);
        });

        // 3. Evidence (extracted phrases per class)
        const evidenceContainer = report.getElementById('evidenceList');
        const evidence = mlData.evidence || {};
        let hasEvidence = false;

        precedenceOrder.forEach(cls => {
            const clsString = String(cls);
            const items = Array.isArray(evidence[clsString]) ? evidence[clsString] : [];
            if (items.length > 0) {
                hasEvidence = true;
                const div = document.createElement('div');
                div.className = 'rank-item';
                div.innerHTML = `
                    <span>${clsString.toUpperCase()}</span>
                    <span class="rank-tag">${items.join(', ')}</span>
                `;
                if (evidenceContainer) evidenceContainer.appendChild(div);
            }
        });

        if (!hasEvidence && evidenceContainer) {
            evidenceContainer.innerHTML =
                '<div class="rank-item" style="opacity:0.5; font-style:italic;">No explicit evidence detected — default fallback to INTERNAL.</div>';
        }

        // 4. Ignored Signals (precedence override)
        const ignoredList = report.getElementById('ignoredSignals');
        const ignored = Array.isArray(mlData.ignored_signals) ? mlData.ignored_signals : [];

        if (ignored.length > 0 && ignoredList) {
            ignored.forEach(sig => {
                const sigString = String(sig);
                const li = document.createElement('li');
                li.textContent = `${sigString.toUpperCase()} — detected but overridden by ${classification.toUpperCase()}`;
                ignoredList.appendChild(li);
            });
        } else if (ignoredList) {
            ignoredList.innerHTML =
                '<li style="background:none; border:none; opacity:0.5; font-style:italic;">No signals overridden. Clean classification.</li>';
        }

        // 5. Audit Details
        const wordCountElem = report.getElementById('dashWordCount');
        if (wordCountElem) {
            wordCountElem.textContent = mlData.metadata ? mlData.metadata.words : '--';
        }

        diagnosticDashboard.appendChild(report);
    }

    function updateSessionMetadata(mlData) {
        const meta = mlData.metadata || {};
        if (sessionHeader) {
            sessionHeader.innerHTML = `
                <span>SID: ${sessionId ? sessionId.slice(0, 8) : '--'}</span> | 
                <span>MODE: XAI</span> | 
                <span>W: ${meta.words || '--'}</span>
            `;
        }
        if (sessionTimestamp) {
            sessionTimestamp.textContent = `TS: ${meta.timestamp || '--'}`;
        }
    }

    function renderPrompts(mlData) {
        clearPrompts();
        const classification = String(mlData.classification || 'unknown');
        const ignored = Array.isArray(mlData.ignored_signals) ? mlData.ignored_signals : [];

        const prompts = [
            "Why was this classification chosen?",
            "What evidence supports this result?",
            "How does the precedence system work?"
        ];

        if (ignored.length > 0) {
            prompts.unshift(`Why was ${String(ignored[0])} ignored in favor of ${classification}?`);
        }

        prompts.forEach(text => {
            const chip = document.createElement('div');
            chip.className = 'prompt-chip';
            chip.textContent = text;
            chip.onclick = () => {
                userInput.value = text;
                sendMessage();
            };
            if (suggestedPrompts) suggestedPrompts.appendChild(chip);
        });
    }

    function clearPrompts() {
        if (suggestedPrompts) suggestedPrompts.innerHTML = '';
    }

    function appendMessage(role, text) {
        const div = document.createElement('div');
        div.className = `message ${role} mono`;
        
        const safeText = String(text || "");
        const formatted = safeText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
            
        div.innerHTML = formatted;
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function setLoading(isLoading) {
        if (sendBtn) sendBtn.disabled = isLoading;
        if (userInput) userInput.disabled = isLoading;
        if (analystStatus) {
            analystStatus.textContent = isLoading ? 'PROCESSING...' : 'READY';
            analystStatus.style.color = isLoading ? 'var(--band-low)' : 'var(--band-high)';
        }
    }

    function resetInvestigation() {
        sessionId = null;
        chatContainer.innerHTML = '';
        diagnosticDashboard.innerHTML = '';
        const mainGrid = document.getElementById('mainGrid');
        if (mainGrid) mainGrid.classList.remove('axial-active');
        if (sessionHeader) sessionHeader.innerHTML = '';
        if (sessionTimestamp) sessionTimestamp.textContent = '--';
        clearPrompts();
        userInput.value = '';
        appendMessage('ai', 'RESET COMPLETE. Terminal awaiting new input.');
    }

    // --- Listeners ---
    if (sendBtn) sendBtn.addEventListener('click', () => sendMessage());
    if (userInput) userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    if (resetBtn) resetBtn.addEventListener('click', resetInvestigation);

    // Initial Greeting
    appendMessage(
        "ai",
        "Paranormix XAI Terminal ready (V4.1.2). Submit a narrative for deterministic signal extraction and classification."
    );
});
