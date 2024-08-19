"use client";
import React, { useEffect, useState } from 'react';

const App = () => {
  const [action, setAction] = useState('');
  const [videoSrc, setVideoSrc] = useState('');

  useEffect(() => {
    // Fetch the action from the Flask server every second
    const fetchAction = () => {
      console.log('Fetching action...');
      fetch('http://localhost:5000/get_action')
        .then(response => response.json())
        .then(data => {
          console.log('Action:', data.action);
          setAction(data.action);
        })
        .catch(err => console.error('Error fetching action:', err));
    };

    const intervalId = setInterval(fetchAction, 1000);

    // Set the video source for streaming
    setVideoSrc('http://localhost:5000/video');

    return () => clearInterval(intervalId); // Cleanup on unmount
  }, []);

  return (
    <div style={{ textAlign: 'center' }}>
      <h1>Gesture Detection</h1>
      <div>
        <h2>Detected Action: {action}</h2>
      </div>
      <div>
        <video
          src={videoSrc}
          autoPlay
          controls
          style={{ width: '80%', height: 'auto', border: '1px solid black' }}
        />
      </div>
    </div>
  );
};

export default App;
