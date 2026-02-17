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

    async function sendMessage(text = null) {
        // Ensure we handle the case where 'text' is a DOM Event (from event listeners)
        const message = (typeof text === 'string') ? text : chatInput.value.trim();
        
        if (!message || isProcessing) return;

        // Configuration for Live Deployment
        // In local dev, it uses relative paths. For GitHub Pages, it can be set via console:
        // localStorage.setItem('paranormix_api_url', 'https://your-backend.up.railway.app')
        const baseUrl = localStorage.getItem('paranormix_api_url') || '';
        const apiUrl = `${baseUrl}/chat`;

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
            report.getElementById('reportPrediction').textContent = mlData.prediction;
            report.getElementById('reportConfidence').textContent = `${(mlData.confidence * 100).toFixed(1)}%`;
            
            const signalsContainer = report.getElementById('reportSignals');
            const signals = mlData.signals || [];
            if (signals.length > 0) {
                signals.forEach(s => {
                    const span = document.createElement('span');
                    span.className = 'signal-tag';
                    span.textContent = s;
                    signalsContainer.appendChild(span);
                });
            } else {
                signalsContainer.textContent = 'Standard markers detected.';
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
                `Why was this classified as ${p}?`,
                `What signals led to this result?`,
                `How certain is the classification?`
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
        const welcomeText = "Welcome. I am Paranormix, your autonomous paranormal narrative investigator. Please share a narrative of supernatural events, and I will perform a machine learning analysis to classify the phenomena.";
        appendMessage('ai', welcomeText);
        
        chatInput.value = '';
        chatInput.style.height = 'auto';
        chatInput.focus();
    }
});
