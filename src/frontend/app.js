/**
 * Paranormix - XAI Diagnostic Suite V3
 * Research Terminal Implementation
 */

document.addEventListener('DOMContentLoaded', () => {
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
                throw new Error(errData.detail || `AXIAL_FAILURE: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.session_id) sessionId = data.session_id;

            if (isInitial && data.ml_data) {
                renderDashboard(data.ml_data);
                updateSessionMetadata(data.ml_data);
                renderPrompts(data.ml_data);
            }

            appendMessage('ai', data.response);
        } catch (err) {
            console.error('Diagnostic error:', err);
            appendMessage('ai', `CRITICAL_SYSTEM_ERROR: ${err.message}`);
        } finally {
            setLoading(false);
        }
    }

    // --- Research Rendering ---

    function renderDashboard(mlData) {
        const mainGrid = document.getElementById('mainGrid');
        mainGrid.classList.add('axial-active');
        
        diagnosticDashboard.innerHTML = '';
        
        const report = dashboardTemplate.content.cloneNode(true);
        const charts = mlData.chart_data;

        // 1. Core Output & Stability
        report.getElementById('dashPrediction').textContent = mlData.prediction;
        const band = report.getElementById('dashBand');
        band.textContent = mlData.band.toUpperCase();
        band.className = `band-indicator band-${mlData.band.toLowerCase()}`;
        report.getElementById('dashStability').textContent = mlData.stability;

        // 2. Class Distribution (Sorted)
        const scoreContainer = report.getElementById('chartClassScore');
        const sortedDist = charts.sorted_distribution || [];
        sortedDist.forEach(item => {
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-lbl">${item.class}</div>
                <div class="bar-out"><div class="bar-in" style="width: ${item.p * 100}%"></div></div>
            `;
            scoreContainer.appendChild(row);
        });

        // 3. Ranked Matches
        const rankContainer = report.getElementById('rankedMatches');
        mlData.ranked_matches.forEach(match => {
            const div = document.createElement('div');
            div.className = 'rank-item';
            div.innerHTML = `
                <span>${match.class.toUpperCase()}</span>
                <span class="rank-tag">${match.label} (${(match.p * 100).toFixed(1)}%)</span>
            `;
            rankContainer.appendChild(div);
        });

        // 4. Grouped Signals
        const observedList = report.getElementById('observedSignals');
        if (mlData.observed.length > 0) {
            mlData.observed.forEach(sig => {
                const li = document.createElement('li');
                li.textContent = sig;
                observedList.appendChild(li);
            });
        } else {
            observedList.innerHTML = '<li style="background:none; border:none; opacity:0.5;">No patterns detected</li>';
        }

        const absentList = report.getElementById('absentSignals');
        mlData.absent.forEach(sig => {
            const li = document.createElement('li');
            li.textContent = sig;
            absentList.appendChild(li);
        });

        // 5. Audit Details
        report.getElementById('dashWordCount').textContent = mlData.metadata.words;

        diagnosticDashboard.appendChild(report);
    }

    function updateSessionMetadata(mlData) {
        const meta = mlData.metadata;
        sessionHeader.innerHTML = `
            <span>SID: ${sessionId.slice(0, 8)}</span> | 
            <span>AXIS: MULTI</span> | 
            <span>W: ${meta.words}</span>
        `;
        sessionTimestamp.textContent = `TS: ${meta.timestamp}`;
    }

    function renderPrompts(mlData) {
        clearPrompts();
        const prompts = [
            "What would increase the confidence band?",
            "Explain the resolution boundary for this case.",
            "How should I interpret the absent indicators?"
        ];
        
        // Contextual prompt for top contender
        if (mlData.ranked_matches.length > 1) {
            const contender = mlData.ranked_matches[1].class;
            prompts.unshift(`Why is ${contender} a contender?`);
        }

        prompts.forEach(text => {
            const chip = document.createElement('div');
            chip.className = 'prompt-chip';
            chip.textContent = text;
            chip.onclick = () => {
                userInput.value = text;
                sendMessage();
            };
            suggestedPrompts.appendChild(chip);
        });
    }

    function clearPrompts() {
        suggestedPrompts.innerHTML = '';
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
        sendBtn.disabled = isLoading;
        userInput.disabled = isLoading;
        analystStatus.textContent = isLoading ? 'PROCESSING...' : 'READY';
        analystStatus.style.color = isLoading ? 'var(--band-low)' : 'var(--band-high)';
    }

    function resetInvestigation() {
        sessionId = null;
        chatContainer.innerHTML = '';
        diagnosticDashboard.innerHTML = '';
        const mainGrid = document.getElementById('mainGrid');
        mainGrid.classList.remove('axial-active');
        sessionHeader.innerHTML = '';
        sessionTimestamp.textContent = '--';
        clearPrompts();
        userInput.value = '';
        appendMessage('ai', 'AXIAL_RESET_COMPLETE. Terminal awaiting new signal input.');
    }

    // --- Listeners ---
    sendBtn.addEventListener('click', () => sendMessage());
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    resetBtn.addEventListener('click', resetInvestigation);

    // Initial Greeting
    appendMessage(
        "ai",
        "**PARANORMIX_ANALYST_DECODER_READY.**\n\nPlease submit a subject narrative for axial capture and empirical pattern analysis. Minimum 50 characters required for statistical significance."
    );
});
