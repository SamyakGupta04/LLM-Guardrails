const messagesContainer = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

// Auto-resize textarea
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value.trim().length > 0) {
        sendBtn.disabled = false;
        sendBtn.style.backgroundColor = '#ECECEC';
        sendBtn.style.color = '#0d0d0d';
    } else {
        sendBtn.disabled = true;
        sendBtn.style.backgroundColor = 'transparent';
        sendBtn.style.color = '#B4B4B4';
    }
});

// Handle enter key
userInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add user message
    addMessage(text, 'user');
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    const loadingId = addLoadingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        removeMessage(loadingId);

        if (response.ok) {
            addMessage(data.response, 'bot');
        } else if (data.blocked) {
            // Show guardrail popup
            showGuardrailPopup(data.issues, data.message);
        } else {
            addMessage(`Error: ${data.detail || 'Something went wrong'}`, 'bot');
        }
    } catch (error) {
        removeMessage(loadingId);
        addMessage(`Connection Error: ${error.message}. Is the backend running?`, 'bot');
    }
}

function showGuardrailPopup(issues, message) {
    // Create overlay
    const overlay = document.createElement('div');
    overlay.className = 'popup-overlay';
    overlay.id = 'guardrailPopup';

    let issuesList = issues.map(issue =>
        `<li><strong>${issue.guard}</strong>: ${issue.description}${issue.matched ? ` (${issue.matched})` : ''}</li>`
    ).join('');

    overlay.innerHTML = `
        <div class="popup-content">
            <div class="popup-header">
                <i class="fa-solid fa-shield-halved"></i>
                <h3>Message Blocked</h3>
            </div>
            <p class="popup-message">${message}</p>
            <div class="popup-issues">
                <strong>Issues detected:</strong>
                <ul>${issuesList}</ul>
            </div>
            <button class="popup-close-btn" onclick="closeGuardrailPopup()">
                <i class="fa-solid fa-xmark"></i> Close
            </button>
        </div>
    `;

    document.body.appendChild(overlay);

    // Close on overlay click
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) closeGuardrailPopup();
    });
}

function closeGuardrailPopup() {
    const popup = document.getElementById('guardrailPopup');
    if (popup) popup.remove();
}

function addMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;

    let contentHtml = text;
    if (sender === 'bot') {
        contentHtml = marked.parse(text);
    } else {
        const temp = document.createElement('div');
        temp.textContent = text;
        contentHtml = `<p>${temp.innerHTML.replace(/\n/g, '<br>')}</p>`;
    }

    msgDiv.innerHTML = `
        <div class="message-content">${contentHtml}</div>
        <div class="message-meta">${sender === 'user' ? 'You' : 'Assistant'}</div>
    `;

    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

function addLoadingIndicator() {
    const id = 'loading-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.id = id;
    msgDiv.className = 'message bot-message';
    msgDiv.innerHTML = `
        <div class="message-content">
            <p><i class="fa-solid fa-circle-notch fa-spin"></i> Thinking...</p>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}
