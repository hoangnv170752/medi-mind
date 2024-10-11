import React, { useState, useEffect, useRef } from 'react';
import profiePic from '../../../assets/doct2.jpg';
import DoctorSidebar from './DoctorSidebar';
import Swal from 'sweetalert2';
import axios from 'axios';

function DoctorReview() {
  const [userData, setUserData] = useState({});
  const [email, setEmail] = useState('');
  const [nurses, setNurses] = useState([]);
  const [message, setMessage] = useState('');
  const [from, setFrom] = useState('');
  const [messages, setMessages] = useState([]); 
  const [loading, setLoading] = useState(true); 
  const messagesEndRef = useRef(null); 

  useEffect(() => {
    const fetchInfo = async () => {
      const user = JSON.parse(localStorage.getItem('user'));
      setUserData(user);
      setFrom(user.name);
    };
    fetchInfo();
  }, []);

  useEffect(() => {
    const getNurses = async () => {
      setLoading(true);
      try {
        const response = await axios.get('https://medi-mind-s2fr.onrender.com/nurse/get-allNurses');
        setNurses(response.data);
      } catch (error) {
        Swal.fire({
          icon: 'error',
          title: 'Oops...',
          text: error.message,
        });
      } finally {
        setLoading(false);
      }
    };

    getNurses();
  }, []);

  const handleAddMessage = async (e) => {
    e.preventDefault();
    if (!message.trim()) {
      Swal.fire({
        icon: 'warning',
        title: 'Warning',
        text: 'Message cannot be empty',
      });
      return;
    }

    try {
      const newMessage = {
        from,
        message,
      };

      
      await axios.post('https://medi-mind-s2fr.onrender.com/doctor/add-message', {
        email,
        message,
        from,
      });

      
      setMessages((prevMessages) => [...prevMessages, newMessage]);
      setMessage(''); 

      
      scrollToBottom();

      Swal.fire({
        icon: 'success',
        title: 'Success',
        text: 'Message Sent',
      });
    } catch (error) {
      Swal.fire({
        icon: 'error',
        title: 'Oops...',
        text: error.message,
      });
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]); 

  return (
    <section className="bg-slate-300 flex justify-center items-center">
      <div className="h-[80%] w-[80%] bg-white shadow-xl p-2 flex">
        <DoctorSidebar userName={userData.name} profiePic={profiePic} />
        <div className="w-[70%] ms-24 p-4 flex flex-col justify-between">
          
          <div className="flex-1 overflow-auto bg-gray-100 p-4 rounded-lg">
            {messages.length > 0 ? (
              messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex mb-2 ${msg.from === from ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`p-3 rounded-lg max-w-xs ${
                      msg.from === from ? 'bg-blue-500 text-white' : 'bg-gray-200 text-black'
                    }`}
                  >
                    <p className="text-sm font-semibold">{msg.from}</p>
                    <p>{msg.message}</p>
                  </div>
                </div>
              ))
            ) : (
              <p>No messages yet</p>
            )}
            <div ref={messagesEndRef} />
          </div>

          {loading && (
            <div className="flex justify-center items-center">
              <div className="loader border-t-4 border-blue-500 rounded-full w-8 h-8 animate-spin"></div>
            </div>
          )}

          
          <form className="flex flex-col mt-4" onSubmit={handleAddMessage}>
            <div>
              <p>Select Nurse:</p>
              <select
                id="nurses"
                onChange={(e) => setEmail(e.target.value)}
                className="flex h-10 w-full rounded-md border border-gray-300 bg-transparent px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-400 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {nurses.map((value, index) => (
                  <option key={index} value={value.email}>
                    {value.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="mt-2">
              <p>Message:</p>
              <textarea
                className="flex w-full rounded-md border border-gray-300 bg-transparent px-3 py-2 text-sm placeholder:text-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-400 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
                placeholder="Enter your message"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
              ></textarea>
            </div>
            <button
              type="submit"
              className="bg-blue-500 text-white p-2 mt-4 rounded-md"
            >
              Send Message
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}

export default DoctorReview;
