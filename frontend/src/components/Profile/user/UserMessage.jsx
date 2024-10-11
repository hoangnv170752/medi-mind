import React, { useState, useEffect } from 'react';
import profiePic from '../../../assets/human6.jpg';
import UserSidebar from './UserSidebar';
import axios from 'axios';

function UserMessage() {
  const userData = JSON.parse(localStorage.getItem('user'));

  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true); // Estado para la pantalla de carga

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await axios.get(`http://103.116.8.27:4451/user/get-messages/${userData.email}`);
        setMessages(response.data);
      } catch (error) {
        console.error('Error fetching messages:', error);
      } finally {
        setLoading(false); // Una vez cargados los mensajes, desactivamos la pantalla de carga
      }
    };

    fetchMessages();
  }, [userData.email]);

  const sendMessage = async () => {
    if (newMessage.trim() === '') return; // No enviar mensajes vacíos

    try {
      const response = await axios.post(`http://103.116.8.27:4451/user/send-message`, {
        email: userData.email,
        message: newMessage
      });
      setMessages([...messages, response.data]); // Añade el nuevo mensaje al estado actual
      setNewMessage(''); // Limpia el campo de texto
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  if (loading) {
    // Mostrar pantalla de carga mientras los datos se obtienen
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
          <div className="flex flex-col w-full h-[400px] bg-gray-200 p-4 rounded-lg overflow-auto shadow-md">
            {messages.length > 0 ? (
              messages.map((msg, index) => (
                <div key={index} className={`p-2 mb-2 rounded-lg ${msg.sender === userData.email ? 'bg-blue-300 self-end' : 'bg-gray-300 self-start'}`}>
                  <p><strong>{msg.sender === userData.email ? 'You' : msg.sender}</strong>: {msg.message}</p>
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
