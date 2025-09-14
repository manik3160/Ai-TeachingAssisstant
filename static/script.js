// Chat functionality
let isLoading = false;

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const typingIndicator = document.getElementById('typingIndicator');
const loadingOverlay = document.getElementById('loadingOverlay');

// Initialize chat
document.addEventListener('DOMContentLoaded', function() {
    // Hide loading overlay after page loads
    setTimeout(() => {
        loadingOverlay.style.display = 'none';
    }, 1000);
    
    // Focus on input
    messageInput.focus();
    
    // Add event listeners
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    messageInput.addEventListener('input', function() {
        sendButton.disabled = !this.value.trim() || isLoading;
    });
});

// Send message function
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input and disable send button
    messageInput.value = '';
    sendButton.disabled = true;
    isLoading = true;
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Send message to backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Add AI response to chat
        addMessage(data.response, 'ai');
        
    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.', 'ai');
    } finally {
        isLoading = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}

// Add message to chat
function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = formatMessage(content);
    
    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageContent.appendChild(time);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    chatMessages.appendChild(messageDiv);
    
    // Remove welcome message if it exists
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    // Scroll to bottom
    scrollToBottom();
}

// Format message content (handle line breaks, links, etc.)
function formatMessage(content) {
    // Convert line breaks to <br>
    content = content.replace(/\n/g, '<br>');
    
    // Highlight video titles and timestamps
    content = content.replace(/(\d+:\d+)/g, '<span class="timestamp">$1</span>');
    
    // Make video titles bold
    content = content.replace(/(📹\s*[^<]+)/g, '<strong class="video-title">$1</strong>');
    
    return content;
}

// Show typing indicator
function showTypingIndicator() {
    typingIndicator.style.display = 'flex';
}

// Hide typing indicator
function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
}

// Scroll to bottom of chat
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Ask question from example chips
function askQuestion(question) {
    messageInput.value = question;
    sendMessage();
}

// Handle window resize
window.addEventListener('resize', function() {
    scrollToBottom();
});

// Add some CSS for formatted content
const style = document.createElement('style');
style.textContent = `
    .timestamp {
        background: #1a3a5c;
        color: #00d4ff;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.9em;
    }
    
    .video-title {
        color: #00d4ff;
        font-weight: 600;
    }
    
    .message-content strong {
        font-weight: 600;
    }
`;
document.head.appendChild(style);
