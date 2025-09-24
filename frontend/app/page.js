"use client";

import { useState, useRef, useEffect } from 'react';
import Plyr from 'plyr';
import Hls from 'hls.js';
import 'plyr/dist/plyr.css';

export default function VideoPlayer() {
  const videoRef = useRef(null);
  const [status, setStatus] = useState('disconnected');
  const [subtitles, setSubtitles] = useState([]);
  const [streamUrl, setStreamUrl] = useState(
    'https://srv265.bilingly.cyou/.../index.m3u8'
  );
  const [VideoLanguage, setVideoLanguage] = useState("en");

  const wsRef = useRef(null);
  const playerRef = useRef(null);
  const hlsRef = useRef(null);
  const subtitleTrackRef = useRef(null);

  const FirstCueArrived = useRef(false);

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (playerRef.current) playerRef.current.destroy();
      if (hlsRef.current) hlsRef.current.destroy();
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    function guardPlay(e) {
      if (!FirstCueArrived.current) {
        e.preventDefault();
        video.pause();
        console.warn('Playback blocked until first subtitle arrives');
      }
    }

    video.addEventListener('play', guardPlay);
    return () => video.removeEventListener('play', guardPlay);
  }, []);

  const clearCues = () => {
    if (subtitleTrackRef.current) {
      const track = subtitleTrackRef.current.track;
      if (track && track.cues) {
        Array.from(track.cues).forEach(cue => track.removeCue(cue));
      }
    }
  };

  const initializePlayer = async () => {
    // Destroy old player if exists
    if (playerRef.current) playerRef.current.destroy();
    if (hlsRef.current) hlsRef.current.destroy();

    // Setup HLS
    if (Hls.isSupported()) {
      hlsRef.current = new Hls();
      hlsRef.current.loadSource(streamUrl);
      hlsRef.current.attachMedia(videoRef.current);

      hlsRef.current.on(Hls.Events.MANIFEST_PARSED, () => {
        playerRef.current = new Plyr(videoRef.current, {
          captions: { active: true, update: true }
        });

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

      });
    } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
      videoRef.current.src = streamUrl;
      videoRef.current.addEventListener('loadedmetadata', () => {
        playerRef.current = new Plyr(videoRef.current);
      });
    }
  };

  const startPlayback = async () => {
    // 1) Initialize player & track
    await initializePlayer();

    setSubtitles([]);
    clearCues();
    FirstCueArrived.current = false;
    setStatus('connecting');

    wsRef.current = new WebSocket('ws://localhost:8011/ws');
    wsRef.current.onopen = () => {
      setStatus('connected');
      wsRef.current.send(streamUrl);
      wsRef.current.send(VideoLanguage);
    };

    wsRef.current.onmessage = (event) => {
      if (event.data === "start") return;
      const data = JSON.parse(event.data);

      setSubtitles(prev => [...prev, data]);

      if (subtitleTrackRef.current) {
        const track = subtitleTrackRef.current.track;
        const cue   = new VTTCue(data.start, data.end, data.text);
        track.addCue(cue);
      }

      if (!FirstCueArrived.current) {
        FirstCueArrived.current = true;

        videoRef.current.currentTime = data.start;
        videoRef.current.play().catch(e => console.error('Play failed:', e));
      }
    };

    wsRef.current.onerror = () => setStatus('error');
    wsRef.current.onclose = () => setStatus('disconnected');
  };

  const stopTranscription = () => {
    if (wsRef.current) wsRef.current.close();
    setStatus('disconnected');
    setSubtitles([]);
    clearCues();
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const statusColors = {
    disconnected: 'bg-red-500',
    connecting:   'bg-yellow-500',
    connected:    'bg-green-500',
    error:        'bg-red-700'
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
        <label className="block mb-2">Video Language:</label>
        <input
          type="text"
          value={VideoLanguage}
          onChange={(e) => setVideoLanguage(e.target.value)}
          className="w-full p-2 bg-gray-700 rounded"
        />
        <div className="text-sm text-gray-400">
          Note: Backend processes 50‑second segments for better accuracy
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
