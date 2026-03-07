
import { useState, useRef, useEffect } from "react";
import { HandleDrop, HandleDragEnter, HandleDragEnd, HandleDragOver, useDragDrop } from "./utils/HandleDrop.js";
import { videoPlayerResizeObserver } from "./utils/ResizeObserver.js";
import { convertFileSrc } from "@tauri-apps/api/core";
import VideoJS from "./components/VideoJS.jsx";
import PlayerUi from "./components/PlayerUi.jsx";
import { getCurrentWindow } from '@tauri-apps/api/window';
/**
 * Drop-in Video.js version of your Plyr + Hls.js player
 * ----------------------------------------------------
 * npm i video.js
 *
 * Feature parity:
 *  - HLS playback (via Video.js VHS; Safari uses native HLS)
 *  - Live subtitles injected over WebSocket as VTTCues
 *  - Block playback until first subtitle cue arrives
 *  - Start/Stop transcription controls
 *  - Cleanup of WebSocket + player on unmount
 *  - Same Tailwind UI, status badge, stream URL & language inputs
 */

export default function VideoPlayer() {
  const videoPlayerZone = useRef({
    element: null,
    contentRect: null
  });
  const TargetedZone = useRef(null);
  const videoRef = useRef(null); // <video> element
  const playerRef = useRef(null); // Video.js player instance
  const wsRef = useRef(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [status, setStatus] = useState("disconnected");
  const [subtitles, setSubtitles] = useState([]);
  const [streamUrl, setStreamUrl] = useState(
    "C:\\Users\\molip\\Downloads\\V for Vendetta (2006)\\V.For.Vendetta.2006.720p.BrRip.x264.YIFY.mp4"
  );
  const [VideoLanguage, setVideoLanguage] = useState("en");


  // handle resize detection
  
  const startPlayback = async () => {
    
    setSubtitles([]);
    setStatus("connecting");
    
    // Connect to your transcription WebSocket
    wsRef.current = new WebSocket("ws://localhost:8011/ws");
    
    wsRef.current.onopen = () => {
      setStatus("connected");
      try {
        wsRef.current?.send(streamUrl);
        wsRef.current?.send(VideoLanguage);
        wsRef.current?.send("ar")  
      } catch {}
    };
    
    wsRef.current.onmessage = (event) => {
      if (event.data === "start") return;
      let data;

      try { 
        
        data = JSON.parse(event.data); 
        
      } catch (e) { 
        
        console.error("Bad JSON:", e);
        return;
        
      }
      console.log("received subtitle : ",data)
      switch (data.route) {
        case "transcription":
          setSubtitles((prev) => [...prev, data.content]);
          break;
        case "translation":
          setSubtitles((prev) => {
            console.log("here is the prev" ,prev);
            const idx = prev.findIndex((l) => l.id === data.content.id);
            if (idx !== -1) {
              prev[idx].translated_text = data.content.translated_text;
              return [...prev];
            }
            return prev;
          });
          break;
      }
    };
    
    wsRef.current.onerror = () => setStatus("error");
    wsRef.current.onclose = () => setStatus("disconnected");
  };
  
  const stopTranscription = () => {
    try { wsRef.current && wsRef.current.close(); } catch {}
    wsRef.current = null;
    setStatus("disconnected");
    setSubtitles([]);
    clearCues();
  };
  
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };
  
  const statusColors = {
    disconnected: "bg-red-500",
    connecting: "bg-yellow-500",
    connected: "bg-green-500",
    error: "bg-red-700",
  };
  const [videoSrc, setVideoSrc] = useState("");
  const [videoData, setVideoData] = useState({
    duration: 0,
    length: "00:00",
    currentTime: 0
  });
  const [videoStatus, setVideoStatus] = useState({
    loaded: false,
    playing: false,
  });
  const [fullScreen, setFullScreen] = useState(false);
  
  videoPlayerResizeObserver(videoPlayerZone);
  
  // handle drag and drop tauri
  useDragDrop(videoPlayerZone, TargetedZone, setFileInfo);
  
  useEffect(() => {
    if(fileInfo && fileInfo.paths && fileInfo.paths.length > 0) {
      const video = convertFileSrc(fileInfo.paths[0]);
      setStreamUrl(fileInfo.paths[0]);
      console.log(video);
      setVideoSrc(video);
      setVideoStatus({...videoStatus, loaded: true, playing: true});
      console.log(video);

    }
  }, [fileInfo]);
  const videoJsOptions = {
    autoplay: true,
    controls: false,
    responsive: false,
    fluid: false,
    sources: [{
      src: videoSrc,
    }],
  };
  const PlayerUiRefs = useRef({});
  const setRefs = (key, el) => {
    PlayerUiRefs.current[key] = el;
  };
  
  const playButtonHandler = async () => {
    if(videoStatus.loaded){
      if(videoStatus.playing){
        setVideoStatus({...videoStatus, playing: false});
        playerRef.current.pause();
      } else {
        setVideoStatus({...videoStatus, playing: true});
        playerRef.current.play();
      }
    }
    return;
  }
  const fullScreenButtonHandler = async () => {
    if(videoStatus.loaded){
      setFullScreen(!fullScreen);
      const appWindow = getCurrentWindow();
      const isFullscreen = await appWindow.isFullscreen();
      console.log("Fullscreen state:", isFullscreen);
      appWindow.setFullscreen(!isFullscreen);
    }
    return videoStatus.loaded;
  }
  
  const volumeButtonHandler = async () => {
    if(videoStatus.loaded){
      if(playerRef.current.muted()){
        playerRef.current.muted(false);
      } else {
        playerRef.current.muted(true);
      }
    }
    return videoStatus.loaded;
  }

  useEffect(() => {
    const player = playerRef.current;
    return () => {
      if (player && !player.isDisposed()) {
        player.dispose();
        playerRef.current = null;
      }
    };
  }, [playerRef]);

  return (
    <div className="h-screen w-screen flex justify-between bg-gray-900 text-white">
      {/* Video.js element */}
      
      <div 
        ref={(el) => videoPlayerZone.current.element = el}
        onDrop={(e) => HandleDrop(e) }
        onDragLeave={(e) => HandleDragEnd(e, TargetedZone) }
        onDragEnter={(e) => HandleDragEnter(e, TargetedZone)}
        onDragOver={(e) => HandleDragOver(e)}
        className=" bg-black h-full relative"
        style={{
          width: fullScreen === false ? "80%" : "100%",
        }}
      >
        <PlayerUi
          setRefs={setRefs}
          videoData={videoData}
          videoStatus={videoStatus}
          fullScreen={fullScreen}
          setFullScreen={setFullScreen}
          playButtonHandler={playButtonHandler}
          fullScreenButtonHandler={fullScreenButtonHandler}
          volumeButtonHandler={volumeButtonHandler}
          subtitles={subtitles}
        />
        {videoStatus.loaded &&
          <VideoJS
            options={videoJsOptions}
            playerRef={playerRef}
            videoRef={videoRef}
            setVideoData={setVideoData}
          />
        }
      </div>
      {/* Optional: tweak caption appearance globally */}
      { fullScreen === false &&
      <div className="max-w-[20%]">
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
          </div>
        </div>

        <div className="mb-4">
          <button
            onClick={startPlayback}
            className="px-4 py-2 bg-blue-600 rounded mr-2 disabled:opacity-50"
            disabled={status === "connected"}
            >
            Start Playback &amp; Transcription
          </button>
          <button
            onClick={stopTranscription}
            className="px-4 py-2 bg-red-600 rounded disabled:opacity-50"
            disabled={status !== "connected"}
            >
            Stop Transcription
          </button>
        </div>


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
      }
    </div>
  );
}
