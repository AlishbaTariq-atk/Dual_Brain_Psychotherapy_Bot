"""
Test script for DBP Chatbot API
This demonstrates how to interact with the chatbot API from a web application
"""
import requests
import json

# API base URL (when running locally)
API_BASE = "http://localhost:8000"

def test_chatbot_api():
    """Test the chatbot API endpoints"""
    
    print("🤖 Testing DBP Chatbot API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    response = requests.get(f"{API_BASE}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Send a chat message
    print("\n2. Testing chat endpoint...")
    chat_data = {
        "message": "Hello, I'm feeling sad today",
        "session_id": "test_session_1"
    }
    
    response = requests.post(f"{API_BASE}/chat", json=chat_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Bot Response: {result['response']}")
        print(f"Session ID: {result['session_id']}")
        print(f"Conversation Length: {result['conversation_length']}")
    
    # Test 3: Send another message in the same session
    print("\n3. Testing follow-up message...")
    chat_data = {
        "message": "What should I do about it?",
        "session_id": "test_session_1"
    }
    
    response = requests.post(f"{API_BASE}/chat", json=chat_data)
    if response.status_code == 200:
        result = response.json()
        print(f"Bot Response: {result['response']}")
        print(f"Conversation Length: {result['conversation_length']}")
    
    # Test 4: Get session information
    print("\n4. Testing session info...")
    response = requests.get(f"{API_BASE}/sessions")
    if response.status_code == 200:
        sessions = response.json()
        print(f"Active Sessions: {len(sessions)}")
        for session in sessions:
            print(f"  - {session['session_id']}: {session['conversation_length']} messages")
    
    # Test 5: Get conversation history
    print("\n5. Testing conversation history...")
    response = requests.get(f"{API_BASE}/sessions/test_session_1/history")
    if response.status_code == 200:
        history = response.json()
        print(f"Conversation History for {history['session_id']}:")
        for i, turn in enumerate(history['conversation_history']):
            role = "User" if turn['role'] == 'user' else "Bot"
            print(f"  {i+1}. {role}: {turn['content'][:100]}...")

def example_web_integration():
    """
    Example of how to integrate with a web-based avatar application
    This shows the basic pattern your teammate would use
    """
    print("\n" + "=" * 50)
    print("🌐 Example Web Integration Pattern")
    print("=" * 50)
    
    # This is what your teammate's JavaScript might look like:
    js_example = '''
    // Example JavaScript for web integration
    async function sendMessageToBot(message, sessionId = 'avatar_session') {
        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId
                })
            });
            
            const data = await response.json();
            
            // Use the bot response to make avatar speak
            makeAvatarSpeak(data.response);
            
            return data;
        } catch (error) {
            console.error('Error communicating with chatbot:', error);
        }
    }
    
    // Example usage in avatar application
    document.getElementById('send-button').addEventListener('click', async () => {
        const userMessage = document.getElementById('user-input').value;
        const botResponse = await sendMessageToBot(userMessage);
        
        // Display user message
        displayMessage('user', userMessage);
        
        // Display bot response and make avatar speak
        displayMessage('bot', botResponse.response);
    });
    '''
    
    print("Your teammate can use this JavaScript pattern:")
    print(js_example)

if __name__ == "__main__":
    try:
        test_chatbot_api()
        example_web_integration()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("Make sure to start the API server first:")
        print("python chatbot_api.py")
    except Exception as e:
        print(f"❌ Error: {e}")
