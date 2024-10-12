import React, { useState, useEffect } from 'react';
import profiePic from '../../../assets/human6.jpg';
import UserSidebar from './UserSidebar';
import axios from 'axios';

function UserMessage() {
  const userData = JSON.parse(localStorage.getItem('user'));

  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);

  // Fetch existing messages when the component mounts
  const fetchMessages = async () => {
    try {
      const response = await axios.get(`https://medi-mind-s2fr.onrender.com/user/get-message/${userData.email}`);
      setMessages(response.data);
    } catch (error) {
      console.error('Error fetching messages:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMessages();
  }, [userData.email]);

  // Handle sending a new message
  const sendMessage = async () => {
    if (newMessage.trim() === '') return; // Avoid sending empty messages

    try {
      // Add user's message to the chat
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: userData.email, message: newMessage }
      ]);

      // Send user's message to the backend
      await axios.post('https://medi-mind-s2fr.onrender.com/user/add-message-chatbot', {
        email: userData.email,
        message: newMessage,
        from: userData.email,
        to: 'chatbot@gmail.com',
      });

      // Call the chatbot API
      const botResponse = await axios.post('https://medi-mind-s2fr.onrender.com/chat', {
        prompt: newMessage,
        patient_data: newMessage,
        chat_history: messages,
      });

      // Add chatbot's response to the chat
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'chatbot@gmail.com', message: botResponse.data.response }
      ]);

      // Save chatbot's message to the backend
      await axios.post('https://medi-mind-s2fr.onrender.com/user/add-message-chatbot', {
        email: userData.email,
        message: botResponse.data.response,
        from: 'chatbot@gmail.com',
        to: userData.email,
      });

      // Clear the input field after sending
      setNewMessage('');
    } catch (error) {
      console.error('Error sending message or fetching chatbot response:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="loader border-t-4 border-blue-500 rounded-full w-12 h-12 animate-spin"></div>
        <p className="ml-4 text-xl">Loading messages...</p>
      </div>
    );
  }

  return (
    <section className='bg-slate-300 flex justify-center items-center'>
      <div className='h-[80%] w-[80%] bg-white shadow-xl p-2 flex'>
        <UserSidebar profiePic={profiePic} userName={userData.userName} />
        <div className="w-[70%] ms-24 p-4 flex flex-col justify-start gap-5">
          <p className="font-semibold text-3xl">Messages</p>
          <div className="flex flex-col w-full h-[1000px] bg-gray-200 p-4 rounded-lg overflow-auto shadow-md">
            {messages.length > 0 ? (
              messages.map((msg, index) => (
            <div
              key={index}
              className={`p-2 mb-2 rounded-lg ${msg.from === userData.email ? 'bg-blue-300 self-end' : 'bg-gray-300 self-start'}`}
            >
              <p>
                <strong>{msg.from === userData.email ? userData.email : 'Chatbot'}</strong>:
                {msg.message.includes('\n') ? (
                  <span dangerouslySetInnerHTML={{ __html: msg.message.replace(/\n/g, '<br/>') }} />
                ) : (
                  msg.message
                )}
              </p>
            </div>
              ))
            ) : (
              <p>No messages yet.</p>
            )}
          </div>
          <div className="flex mt-4">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              className="flex-grow p-2 border border-gray-300 rounded-md"
              placeholder="Type your message..."
            />
            <button
              onClick={sendMessage}
              className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-md"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

export default UserMessage;
