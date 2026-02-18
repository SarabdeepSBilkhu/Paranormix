/**
 * Paranormix - Pure Chatbot Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSendBtn');
    const resetBtn = document.getElementById('resetBtn');
    const turnCounter = document.getElementById('turnCounter');
    const suggestedQuestions = document.getElementById('suggestedQuestions');
    const suggestedButtons = document.getElementById('suggestedButtons');
    const reportTemplate = document.getElementById('reportTemplate');

    // State
    let sessionId = null;
    let turnCount = 0;
    let isProcessing = false;

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
        sendBtn.disabled = chatInput.value.trim().length === 0 || isProcessing;
    });

    // Send on Enter (but allow Shift+Enter for new lines)
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);
    resetBtn.addEventListener('click', resetInvestigation);

    // Live Deployment Configuration
    // In a unified full-stack deployment (Railway), the frontend is served from the same origin.
    // Use an empty string for relative paths, which is the architectural best practice.
    const API_BASE_URL = "";

    async function sendMessage(text = null) {
        // Ensure we handle the case where 'text' is a DOM Event (from event listeners)
        const message = (typeof text === 'string') ? text : chatInput.value.trim();
        
        if (!message || isProcessing) return;

        const apiUrl = `${API_BASE_URL}/chat`;

        // Clear input
        if (!text) {
            chatInput.value = '';
            chatInput.style.height = 'auto';
            sendBtn.disabled = true;
        }

        // Add user message to UI
        appendMessage('user', message);
        
        // Show loading
        isProcessing = true;
        const loadingDiv = appendMessage('ai', 'Investigating...', true);
        
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    user_message: message
                })
            });

            const data = await response.json();

            if (!response.ok) {
                const errorDetail = typeof data.detail === 'object' ? JSON.stringify(data.detail) : data.detail;
                throw new Error(errorDetail || 'The spirits are silent.');
            }

            // Update State
            sessionId = data.session_id;
            turnCount = data.turn_count;
            updateUI(data);

            // Replace loading with actual response
            loadingDiv.remove();
            appendMessage('ai', data.response, false, data.ml_data);

        } catch (error) {
            loadingDiv.textContent = `Error: ${error.message}`;
            loadingDiv.classList.add('error-bubble');
            console.error('Chat error:', error);
        } finally {
            isProcessing = false;
        }
    }

    function appendMessage(role, text, isLoading = false, mlData = null) {
        // Remove welcome message on first user message if it's there
        if (role === 'user' && chatMessages.children.length === 1 && chatMessages.children[0].textContent.includes("Welcome")) {
            chatMessages.children[0].remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${role}`;
        
        // Simple Markdown-style replacement for bold text
        if (!isLoading) {
            const formattedText = text
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\n/g, '<br>');
            bubble.innerHTML = formattedText;
        } else {
            bubble.textContent = text;
        }
        
        messageDiv.appendChild(bubble);

        // If it's the initial report, inject the ML data report
        if (mlData) {
            const report = reportTemplate.content.cloneNode(true);
            const pred = mlData.prediction.toLowerCase().split(' ')[0];
            
            const badge = report.getElementById('reportBadge');
            badge.textContent = mlData.prediction;
            badge.classList.add(`badge-${pred}`);
            
            report.getElementById('reportPrediction').textContent = `Dominant Diagnosis: ${mlData.prediction}`;
            
            // Certainty
            const certainty = report.getElementById('reportCertainty');
            certainty.textContent = mlData.certainty;
            certainty.classList.add(`certainty-${mlData.certainty.toLowerCase()}`);
            
            // Signals (Evidence)
            const evidenceContainer = report.getElementById('reportEvidence');
            const evidence = mlData.evidence || [];
            if (evidence.length > 0) {
                evidence.forEach(s => {
                    const span = document.createElement('span');
                    span.className = 'signal-tag evidence-tag';
                    span.textContent = s;
                    evidenceContainer.appendChild(span);
                });
            } else {
                evidenceContainer.innerHTML = '<span class="none">None Detected</span>';
            }

            // Modifiers
            const modifiersContainer = report.getElementById('reportModifiers');
            const modifiers = mlData.modifiers || [];
            if (modifiers.length > 0) {
                modifiers.forEach(s => {
                    const span = document.createElement('span');
                    span.className = 'signal-tag modifier-tag';
                    span.textContent = s;
                    modifiersContainer.appendChild(span);
                });
            } else {
                modifiersContainer.innerHTML = '<span class="none">None Detected</span>';
            }

            // Competing Hypotheses
            const competing = report.getElementById('reportCompeting');
            const hyps = mlData.competing || [];
            if (hyps.length > 0) {
                hyps.forEach(h => {
                    const div = document.createElement('div');
                    div.className = 'hypothesis-item';
                    div.textContent = h;
                    competing.appendChild(div);
                });
            } else {
                competing.innerHTML = '<div class="hypothesis-item none">No secondary indicators</div>';
            }

            messageDiv.appendChild(report);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return bubble;
    }

    function updateUI(data) {
        turnCounter.textContent = `Investigation depth: ${data.turn_count}/5`;
        
        // Handle suggestions
        if (data.is_initial && data.ml_data) {
            suggestedQuestions.style.display = 'block';
            suggestedButtons.innerHTML = '';
            
            const p = data.ml_data.prediction;
            const suggestions = [
                `Evidence for ${p}?`,
                `Model uncertainty?`,
                `Summary report`
            ];

            suggestions.forEach(q => {
                const btn = document.createElement('button');
                btn.className = 'suggested-btn';
                btn.textContent = q;
                btn.onclick = () => {
                    sendMessage(q);
                    suggestedQuestions.style.display = 'none';
                };
                suggestedButtons.appendChild(btn);
            });
        } else {
            suggestedQuestions.style.display = 'none';
        }
    }

    function resetInvestigation() {
        sessionId = null;
        turnCount = 0;
        chatMessages.innerHTML = '';
        turnCounter.textContent = 'Investigation depth: 0/5';
        suggestedQuestions.style.display = 'none';
        
        // Initial Local Welcome
        const welcomeText = "Welcome. I am Paranormix, your autonomous paranormal narrative investigator. Please share a narrative to begin machine learning analysis.";
        appendMessage('ai', welcomeText);
        
        chatInput.value = '';
        chatInput.style.height = 'auto';
        chatInput.focus();
    }
});
