"use client"
import { useState, useRef, useEffect } from 'react';
import Plyr from 'plyr';
import Hls from 'hls.js';
import 'plyr/dist/plyr.css';

export default function VideoPlayer() {
  const videoRef = useRef(null);
  const [status, setStatus] = useState('disconnected');
  const [subtitles, setSubtitles] = useState([]);
  const [streamUrl, setStreamUrl] = useState('https://srv265.bilingly.cyou/aes/0/c541fbdab16d5976056985f470c8c2889c8d3761d3f44a2a0e504f57c68ebae4/sZHAQHPAHuwltvAGgc9VSw/1752858001/storage5/movies/0347149-howls-moving-castle-2004-1600380158/f2bb4134e29f1777182435839e20af5f.mp4/index.m3u8');
  const [VideoLanguage, setVideoLanguage] = useState("en");
  const wsRef = useRef(null);
  const playerRef = useRef(null);
  const hlsRef = useRef(null);
  const subtitleTrackRef = useRef(null);
  const countRef = useRef(0);

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (playerRef.current) playerRef.current.destroy();
      if (hlsRef.current) hlsRef.current.destroy();
    };
  }, []);

  const clearCues = () => {
    if (subtitleTrackRef.current) {
      const track = subtitleTrackRef.current.track;
      if (track && track.cues) {
        // Remove all cues
        Array.from(track.cues).forEach(cue => track.removeCue(cue));
      }
    }
  };

  const initializePlayer = () => {
    // Clean up existing player
    if (playerRef.current) {
      playerRef.current.destroy();
    }
    if (hlsRef.current) {
      hlsRef.current.destroy();
    }

    // Initialize HLS.js
    if (Hls.isSupported()) {
      hlsRef.current = new Hls();
      hlsRef.current.loadSource(streamUrl);
      hlsRef.current.attachMedia(videoRef.current);
      
      hlsRef.current.on(Hls.Events.MANIFEST_PARSED, () => {
        // Create player
        playerRef.current = new Plyr(videoRef.current, {
          captions: { active: true, update: true }
        });
        
        // Create text track for subtitles
        if (subtitleTrackRef.current) {
          videoRef.current.removeChild(subtitleTrackRef.current);
        }
        
        
        const track = document.createElement('track');
        track.kind = 'subtitles';
        track.label = 'Live Subtitles';
        track.srclang = 'en';
        track.default = true;
        videoRef.current.appendChild(track);
        subtitleTrackRef.current = track;
        
        // Try to play
        
        videoRef.current.play().catch(e => console.error('Play failed:', e));
      });
    } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
      videoRef.current.src = streamUrl;
      videoRef.current.addEventListener('loadedmetadata', () => {
        playerRef.current = new Plyr(videoRef.current);
        videoRef.current.play().catch(e => console.error('Play failed:', e));
      });
    }
  };

  const startPlayback = () => {
    if (wsRef.current) wsRef.current.close();
    setSubtitles([]);
    clearCues();
    countRef.current = 0;
    setStatus('connecting');
    wsRef.current = new WebSocket('ws://localhost:8011/ws');
    wsRef.current.onopen = () => {
      setStatus('connected');
      wsRef.current.send(streamUrl);
      wsRef.current.send(VideoLanguage);
    };
    wsRef.current.onmessage = (event) => {
      if(event.data === "start"){
        initializePlayer()
        return;
      }
      try {
        const data = JSON.parse(event.data);
        setSubtitles(prev => [...prev, data]);
        if (subtitleTrackRef.current) {
          const track = subtitleTrackRef.current.track;
          const cue = new VTTCue(data.start, data.end, data.text);
          track.addCue(cue);
        }
      } catch (e) {
        console.error('Error parsing subtitle:', e);
      }
    };
    wsRef.current.onerror = () => setStatus('error');
    wsRef.current.onclose = () => setStatus('disconnected');
  };

  const stopTranscription = () => {
    if (wsRef.current) {
      wsRef.current.close();
      setStatus('disconnected');
      setSubtitles([]);
      clearCues();
    }
  };

  const statusColors = {
    disconnected: 'bg-red-500',
    connecting: 'bg-yellow-500',
    connected: 'bg-green-500',
    error: 'bg-red-700'
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <h1 className="text-2xl font-bold mb-4">Video Player with Live Subtitles</h1>
      
      <div className={`p-2 rounded mb-4 ${statusColors[status]}`}>
        Status: {status.charAt(0).toUpperCase() + status.slice(1)}
      </div>
      
      <div className="mb-4">
        <label className="block mb-2">Stream URL:</label>
        <input
          type="text"
          value={streamUrl}
          onChange={(e) => setStreamUrl(e.target.value)}
          className="w-full p-2 bg-gray-700 rounded mb-2"
        />
        <label className="block mb-2">Video Language</label>
        <input
          type="text"
          value={VideoLanguage}
          onChange={(e) => setVideoLanguage(e.target.value)}
          className="w-full p-2 bg-gray-700 rounded mb-2"
        />
        <div className="text-sm text-gray-400">
          Note: Backend processes 50-second segments for better accuracy
        </div>
      </div>
      
      <div className="mb-4">
        <button
          onClick={startPlayback}
          className="px-4 py-2 bg-blue-600 rounded mr-2 disabled:opacity-50"
          disabled={status === 'connected'}
        >
          Start Playback & Transcription
        </button>
        <button
          onClick={stopTranscription}
          className="px-4 py-2 bg-red-600 rounded disabled:opacity-50"
          disabled={status !== 'connected'}
        >
          Stop Transcription
        </button>
      </div>
      
      <video ref={videoRef} className="w-full" controls />
      
      <div className="mt-6 bg-gray-800 p-4 rounded max-h-60 overflow-y-auto">
        <h2 className="text-xl font-bold mb-2">Subtitles Log</h2>
        {subtitles.length === 0 ? (
          <div className="text-gray-400">No subtitles received yet</div>
        ) : (
          subtitles.map((sub, i) => (
            <div key={i} className="mb-2 p-2 bg-gray-700 rounded">
              <div className="text-gray-400 text-sm">
                {formatTime(sub.start)} - {formatTime(sub.end)}
              </div>
              <div>{sub.text}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}