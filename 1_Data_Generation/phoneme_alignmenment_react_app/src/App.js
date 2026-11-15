import React, { Component }  from 'react';
import { useState, useRef, useEffect } from 'react'
import Notes from './components/midi_note'

function App() {
  const queryParams = new URLSearchParams(window.location.search);
  const artist_name = queryParams.get('artist_name');
  const song_name = queryParams.get('song_name');
  const bpm = queryParams.get('bpm');
  const stanza = queryParams.get('stanza');

  const [error, setError] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [items, setItems] = useState([]);

  // Note: the empty deps array [] means
  // this useEffect will run once
  // similar to componentDidMount()
  useEffect(() => {
    fetch(`http://127.0.0.1:5000/api?artist_name=${artist_name}&song_name=${song_name}&bpm=${bpm}&stanza=${stanza}`)
      .then(res => res.json())
      .then(
        (result) => {
          console.log("result", result)
          setItems(result);
          console.log("items", items)
          setIsLoaded(true);
        },
        // Note: it's important to handle errors here
        // instead of a catch() block so that we don't swallow
        // exceptions from actual bugs in components.
        (error) => {
          setIsLoaded(true);
          setError(error);
        }
      )
  }, [])
 

  if (error) {
    return <div>Error: {error.message}</div>;
  } else if (!isLoaded) {
    return <div>Loading...</div>;
  } else {
    return (
      <div>
        <Notes note_data={items['notes']}/>
      </div>
  );
  }
}

export default App
