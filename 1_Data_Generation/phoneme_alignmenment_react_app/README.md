# Vocal MIDI–Syllable Alignment Editor

A React-based, interactive editor for **aligning phonetic lyric syllables with vocal MIDI notes**. Built as part of the Maximus / Music Generative AI research workflow, this tool visualizes the timing and pitch of a vocal performance, allows manual correction of syllable-to-note assignments, plays a corresponding vocal recording, and sends annotated note data back to a local service.

**Scope:** This is a human-in-the-loop **dataset annotation interface**, not a standalone MIDI generator or vocal synthesizer. The React frontend is included; the API and WAV-file server on which it depends are not.

## Motivation and workflow

A symbolic-vocal model can use music and language jointly only when vocal note events and their corresponding phonetic syllables are aligned. A MIDI track supplies onset, duration, and pitch, while phonetic lyric data supplies the sounds associated with those note events. Incorrect initial matches can be corrected by an annotator.

The intended workflow is:

1. Convert vocal MIDI and phonetic lyric material into an indexed list of note/syllable records (the accompanying Python notebook demonstrates this preparation).
2. Load one song section into the editor using artist, song, BPM, and stanza URL parameters.
3. Inspect a **time-versus-pitch** arrangement of notes, each labeled with its current phonetic syllable.
4. Move labels among note positions, or mark an unassigned note with a placeholder.
5. Review the sequence against a playable vocal recording with a visible playback-position indicator.
6. Submit the edited annotations for downstream processing.

## User-facing functionality

### Note visualization

`src/components/midi_note.js` renders every note as a blue, rounded horizontal bar containing its phonetic label. The bars are positioned using their note-event fields:

- **Start time:** the left offset is `start * 5` pixels.
- **Duration:** the bar width is `duration * 5` pixels.
- **Pitch:** the bottom offset is calculated from a minimum/maximum pitch scan (`src/lib/min_max.js`) and `src/lib/scalePitch.js`.
- **Note ID:** used as both the React item key and an index in editing operations, so IDs should be sequential and zero-based.

This is a simplified pitch/timeline diagram, not traditional staff notation. The current pitch helper divides the offset above the lowest pitch by `max` rather than by the complete pitch range; its vertical scaling should be treated as experimental.

### Manual syllable editing

Two interactions are implemented:

- **Shift-click an empty note:** toggle its `syllable` field between the empty string `""` and the placeholder `"-"`. Occupied phonetic labels are not overwritten by Shift-click.
- **Drag one labeled note onto another note:** attempt to shift a sequence of syllable fields along the note list. The forward/backward branches guard against some overwrites and clear vacated slots. This shifts **labels**, not pitches, note starts, or durations.

The drag-and-drop algorithm works with array indices and makes assumptions about note ordering and available blank positions. Edge cases and out-of-range conditions are not comprehensively validated.

### Audio playback and synchronization

The editor uses the browser's native `Audio` object and a separate `AudioControls.jsx` play/pause button. The **spacebar** also toggles playback. A red vertical position marker advances according to the audio playback time and a tempo-dependent note-grid conversion. Its position is polled approximately every 10 ms.

Audio URLs follow this pattern:

`http://127.0.0.1:8887/<artist_name>_<song_name>/<section_file>.wav`

| Stanza parameter | Expected audio filename |
| --- | --- |
| `verse1` | `verse1_vocals.wav` |
| `prechorus` | `prechorus_vocals.wav` |
| `chorus` | `chorus_vocals.wav` |
| `post_chorus` | `post_vocals.wav` |
| `verse2` | `verse2_vocals.wav` |

The supplied MP3 examples are reference assets and are **not automatically loaded** by this playback logic. A separately served WAV file is required for the song and section.

### Submission

`Submit` becomes enabled when **every** note has a non-empty syllable field; `"-"` counts as an acceptable completed annotation. Pressing it POSTs the full edited note array as JSON. On an HTTP success response, the code attempts to navigate to a `pronunciation_form` URL. The corresponding form and route are **not implemented in the uploaded frontend**.

## Input format and backend contract

The application parses four query parameters:

| Parameter | Purpose |
| --- | --- |
| `artist_name` | Artist identifier used for the API query and audio directory |
| `song_name` | Song identifier used for the API query and audio directory |
| `bpm` | Tempo used for interpreting the note grid and audio timeline |
| `stanza` | Selected musical section, also controlling the WAV filename |

Example:

`http://localhost:3000/?artist_name=example_artist&song_name=example_song&bpm=138&stanza=verse1`

### Load

`src/App.js` performs a GET request on initial mount:

`GET http://127.0.0.1:5000/api?artist_name=...&song_name=...&bpm=...&stanza=...`

It expects a JSON object containing a `notes` array:

```json
{
  "notes": [
    {
      "id": 0,
      "start": 132,
      "duration": 11,
      "pitch": 25,
      "syllable": "SH IY Z"
    },
    {
      "id": 1,
      "start": 144,
      "duration": 11,
      "pitch": 27,
      "syllable": ""
    }
  ]
}
```

`start` and `duration` are quantized time-grid positions rather than seconds. The companion Python notebook constructs its grid with **24 steps per beat**, calculated as `(bpm / 60) * 4 * 6` positions per second. Its pitch values are offsets from MIDI note C2. Syllable text uses a sequence of phoneme symbols, such as ARPABET-style vowel and consonant codes.

### Save

`src/components/midi_note.js` sends the edited **array**, rather than a wrapping `{"notes": [...]}` object, to:

`POST http://127.0.0.1:5000/postapi?artist_name=...&song_name=...&bpm=...&stanza=...`

The request uses `Content-Type: application/json`. The backend implementation, storage layer, and the follow-on pronunciation form are not included here. A working integration must supply both API endpoints and permit appropriate cross-origin requests from the frontend.

## Companion Python preparation notebook

[`Syllables_break.ipynb`](Syllables_break.ipynb) explores how to prepare the data for this interface using `pretty_midi`, NumPy, pickle, Matplotlib, and phoneme parsing.

- **`midi_to_list(filename, bpm)`:** loads MIDI notes, orders them by onset and pitch, quantizes onset/end time to a BPM-derived grid, and expresses pitch relative to C2.
- **`midi_to_grid(filename, bpm)`:** converts notes into a NumPy piano-roll-like array with `1` marking occupied note positions and `2` marking note onsets, for visual inspection.
- **`syll_to_json(syll_file, section, midi_file, bpm, out_json_filename)`:** loads phonetic lyric data from a pickled section dictionary, removes selected stress digits, separates vowel-bearing portions of phoneme sequences, associates syllables with MIDI note events, and emits the JSON structure consumed by the app.

The notebook uses historical local sample paths and assumes the requisite pickled lyric material and MIDI files are present. It is an exploratory preprocessing notebook, **not** the running API service.

## Code layout

```text
phoneme_alignmenment_react_app/
├── README.md
├── package.json / package-lock.json
├── Syllables_break.ipynb       # MIDI + phonetic syllable preprocessing
├── public/index.html          # React HTML shell
└── src/
    ├── App.js                 # URL parameters, API loading, error/loading UI
    ├── index.js               # Entry point and React Router wrapper
    ├── components/
    │   ├── midi_note.js       # Note editor, label movement, audio, submit
    │   └── AudioControls.jsx  # Play/pause interface
    ├── lib/
    │   ├── min_max.js         # Minimum/maximum note pitch
    │   ├── scalePitch.js      # Visual pitch coordinate
    │   └── last_syllable.js   # Unused/incomplete experimental helper
    ├── data/
    │   ├── verse1.json        # Example note and syllable labels
    │   ├── verse1.mid         # Example vocal MIDI
    │   ├── verse1_acc.mp3     # Example accompaniment audio
    │   └── verse1_vocals.mp3  # Example vocal audio
    └── App.test.js            # Original scaffold test, not editor tests
```

Most editor styling lives inline in the note component; a few CSS, SVG, and public assets remain from the Create React App scaffold.

## Running locally

The project is built on **React 17**, **react-router-dom 5**, **react-scripts 4.0.3** (Create React App), and **@use-it/event-listener**.

```bash
git clone https://github.com/Ambrose-Plante/Music-generative-AI.git
cd Music-generative-AI/1_Data_Generation/phoneme_alignmenment_react_app
npm ci
npm start
```

Open `http://localhost:3000` with the required URL parameters.

**Important:** `npm start` only starts the React frontend. A functional editor additionally requires:

1. A local JSON API on **127.0.0.1:5000**, with compatible GET `/api` and POST `/postapi` endpoints.
2. A WAV static-file server on **127.0.0.1:8887**, with the expected artist/song folders and section filenames.

These services **are not included** in this directory. Without the API, the editor displays a loading or fetch error rather than note data. The bundled demonstration assets do not automatically substitute for the backend.

Other package scripts: `npm run build` creates a production bundle, and `npm test` invokes the scaffold Jest test runner. The existing `App.test.js` still asserts the old “learn react” link and has not been updated to test this editor.

