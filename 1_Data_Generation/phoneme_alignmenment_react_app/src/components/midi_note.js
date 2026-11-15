import React  from 'react';
import { useState, useRef, useEffect } from 'react'
import findMinMax from '../lib/min_max'
import scalePitch from '../lib/scalePitch'
import AudioControls from './AudioControls'
import useEventListener from '@use-it/event-listener'
import { useHistory } from "react-router-dom";

const Notes = ({ note_data }) => {
  const [notes, setNotes] = useState(note_data)
  const [dragging, setDragging] = useState(false);
  const [active, setActive] = useState(false);

  const queryParams = new URLSearchParams(window.location.search);
  const artist_name = queryParams.get('artist_name');
  const song_name = queryParams.get('song_name');
  const bpm = queryParams.get('bpm');
  const stanza = queryParams.get('stanza');

  var [min, max] = findMinMax(notes);
  var yScale = 800
  var xScale = 5
  let history = useHistory();

  const dragItem = useRef();
  const dragNode = useRef();

// Audio Stuff
// -------------------------------------
  const [trackProgress, setTrackProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  var wav_path = ''
  if ( stanza==='verse1') {
    wav_path = `http://127.0.0.1:8887/${artist_name}_${song_name}/verse1_vocals.wav`
  } else if ( stanza==='prechorus') {
    wav_path = `http://127.0.0.1:8887/${artist_name}_${song_name}/prechorus_vocals.wav`
  } else if ( stanza==='chorus') {
    wav_path = `http://127.0.0.1:8887/${artist_name}_${song_name}/chorus_vocals.wav`
  } else if ( stanza==='post_chorus') {
    wav_path = `http://127.0.0.1:8887/${artist_name}_${song_name}/post_vocals.wav`
  } else if ( stanza==='verse2') {
    wav_path = `http://127.0.0.1:8887/${artist_name}_${song_name}/verse2_vocals.wav`
  }
  const audioRef = useRef(new Audio(wav_path));

  const intervalRef = useRef();
  const isReady = useRef(false);

  // Destructure for conciseness
  const { duration } = audioRef.current;
  const wav_width = ((Number(bpm)/60) * 4 * 6) * duration * xScale

  useEffect(() => {
    if (isPlaying) {
      audioRef.current.play();
      startTimer();
      console.log({trackProgress}, {duration})
    } else {
      audioRef.current.pause();
    }
  }, [isPlaying]);

  useEffect(() => {
    // Pause and clean up on unmount
    return () => {
      audioRef.current.pause();
      clearInterval(intervalRef.current);
    }
  }, []);

  const startTimer = () => {
    // Clear any timers already running
    clearInterval(intervalRef.current);
  
    intervalRef.current = setInterval(() => {
      setTrackProgress(audioRef.current.currentTime);
    }, [10]);
  }

  const ESCAPE_KEYS = ['32', ' '];
  function handler({ key }) {
    if (ESCAPE_KEYS.includes(String(key))) {
    console.log("Spacebar pressed!");
      if (isPlaying) {
        setIsPlaying(false);
      } else {
        setIsPlaying(true);
      }
    }
  }

  useEventListener('keydown', handler);

// -------------------------------------

  const handleClick = (e, params) => {
    if ( e.shiftKey ) {
      console.log("shift clicking...", params.note.id)
      setNotes(oldNotes => {
        let newNotes = JSON.parse(JSON.stringify(oldNotes));
        if ( oldNotes[params.note.id].syllable === "" ) {
          newNotes[params.note.id].syllable = "-"
        }
        else if ( oldNotes[params.note.id].syllable === "-" ) {
          newNotes[params.note.id].syllable = ""
        }
        else {
          console.log("not allowed. Syllable must be blank")
        }
        return newNotes
      })
    }
    else {
      console.log("clicking...", params.note.id)
    }
  }

  const handleDragStart = (e, params) => {
    console.log('drag starting...', params.note.id)
    dragItem.current = params;
    dragNode.current = e.target;
    dragNode.current.addEventListener('dragend', handleDragEnd)
    setTimeout(() => {
      setDragging(true)
    }, 0)
  }

  const handleDrop = (e, params) => {
    e.preventDefault();
    e.stopPropagation();
    const currentNote = dragItem.current;
    console.log('Dropping...', params)
    if (e.target !== dragNode.current) {
      setNotes(oldNotes => {
        let newNotes = JSON.parse(JSON.stringify(oldNotes));
        let last_ind = newNotes.length - params.note.id + currentNote.note.id
        let backwards = false
        if ( params.note.id < currentNote.note.id ){
          backwards = true
        }
        if ( backwards ) {
          let allow_backwards = true
          for (let i = params.note.id, len=currentNote.note.id; i < len; i++) {
            if (oldNotes[i].syllable !== "" ){
              allow_backwards=false
            }
          }
          
          if ( allow_backwards ) {
            let end_ind = newNotes.length - currentNote.note.id + params.note.id
            for (let i = params.note.id, len=end_ind; i < len; i++) {
              let old_ind = i - params.note.id + currentNote.note.id
              newNotes[i].syllable = oldNotes[old_ind].syllable
            }
            for (let i = end_ind, len=newNotes.length; i < len; i++) {
              newNotes[i].syllable = ""
            }
          }
          else {
            console.log("No. Backwards can only go to empty space..")
          }
        }
        else if ( oldNotes[last_ind].syllable === "" ){  
          console.log("Allowed", oldNotes[last_ind].syllable)
          for (let i = params.note.id, len=newNotes.length; i < len; i++) {
            let old_ind = i - params.note.id + currentNote.note.id
            newNotes[i].syllable = oldNotes[old_ind].syllable
          }
          for (let i = currentNote.note.id, len=params.note.id; i < len; i++) {
            newNotes[i].syllable = ""
          }
        }
        else {
          console.log("NOT Allowed", oldNotes[last_ind].syllable)
        }
        return newNotes
      })
    }
  }

  const handleDragOver = (e, params) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.target !== dragNode.current) {
      console.log('Dragging over...', params)
    }
  };

  const handleDragEnd = () => {
    console.log('ending drag...')
    dragNode.current.removeEventListener('dragend', handleDragEnd)
    setDragging(false);
    dragItem.current = null;
    dragNode.current = null;
  }

  const getStyles = (params) => {
    const currentNote = dragItem.current;
    console.log(currentNote.note.id)
    if (currentNote.note.id === params.note.id) {
    	return 'white'
    }
    return 'blue'
  }

  useEffect(() => {
    let activate = true 
    for (let i = 0, len=notes.length; i < len; i++) {
      if ( notes[i].syllable === "" ){
        activate=false
      }
    }
    if ( activate ) {
      setActive(true)
    }
    else {
      setActive(false)
    }
  }, [notes]
  );

  const SubmitClick = async () => {
    console.log("submitting...", notes)
    const response = await fetch( `http://127.0.0.1:5000/postapi?artist_name=${artist_name}&song_name=${song_name}&bpm=${bpm}&stanza=${stanza}`, {
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      }, 
      method: 'POST',
      body: JSON.stringify(notes)
      })

      if (response.ok) {
        console.log("it worked")
        let url = `pronunciation_form?artist_name=${artist_name}&song_name=${song_name}&bpm=${bpm}`
        history.push(url)  // redirect
      }
    };

  return (
    <div className='container'>
      {notes.map((note) => (
        <div
          key={note.id}
          draggable
          style={{
            display: "flex",
            position: 'absolute',
            width: note.duration * xScale,
            height: 50,
            bottom: scalePitch(note.pitch, min, max, yScale),
            left: note.start * xScale,
            color: 'orange',
            cursor: 'grab',
            justifyContent: "center",
            alignItems: "center",
            borderRadius:15,  // rounded edges
            backgroundColor: dragging ? getStyles({note}) : 'blue'
          }}
          onDragStart={(e) => {handleDragStart(e, {note})}}
          onClick={(e) => {handleClick(e, {note})}}
          onDrop={(e) => handleDrop(e, {note})}
          onDragOver={e => handleDragOver(e, {note})}
        >
          {note.syllable}
        </div>
      ))}
        <button
          onClick={SubmitClick}
//          disabled={{submitCheck}}
          disabled={!active}
          style={{
            height: 50,
          }}
        >Submit</button>
    	<AudioControls
          isPlaying={isPlaying}
          onPlayPauseClick={setIsPlaying}
        />
        <div
          style={{
            display: "flex",
            position: 'relative',
            width:  10,
            height: 200,
            left: duration ? (( trackProgress / duration ) * wav_width) : 0,
            backgroundColor: 'red',
            zIndex: '100'
          }}>
        </div>
    </div>
  )
}

export default Notes
