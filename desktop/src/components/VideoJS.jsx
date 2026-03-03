import React from 'react';
import videojs from 'video.js';
import 'video.js/dist/video-js.css';
import { secondsToHMS } from '../utils/secondToHMS';
import sleep from '../utils/sleep';

export const VideoJS = (props) => {
  const {options, videoRef, playerRef} = props;

  React.useEffect(() => {
    console.log("playerRef.current : ", playerRef.current);
    // Make sure Video.js player is only initialized once
    if (!playerRef.current) {
      // The Video.js player needs to be _inside_ the component el for React 18 Strict Mode. 
      const videoElement = document.createElement("video-js");

      videoElement.classList.add('vjs-big-play-centered');
      videoRef.current.appendChild(videoElement);

      playerRef.current = videojs(videoElement, options, () => {
        videojs.log('player is ready');
      });
      console.log("playerRef.current 2 : ", playerRef.current);
    // You could update an existing player in the `else` block here
    // on prop change, for example:
    } else {
      const player = playerRef.current;
      player.on("loadedmetadata", () => {
        const duration = player.duration();
      
        const {hours, minutes, seconds} = secondsToHMS(duration);
        console.log("current volume : ", player.volume());
        const length =
          `${hours > 0 ? Math.floor(hours) + ":" : ""}` +
          `${minutes < 10 ? "0" + Math.floor(minutes) : Math.floor(minutes)}:` +
          `${seconds < 10 ? "0" + Math.floor(seconds) : Math.floor(seconds)}`;
      
        props.setVideoData({
          duration,
          length
        });
      
      });
      window.addEventListener("keydown", (e) => {
        console.log(e.key);
        switch(e.key){
          case "ArrowLeft":
            player.currentTime(player.currentTime() - 5);
            break;
          case "ArrowRight":
            player.currentTime(player.currentTime() + 5);
            break;
          case "ArrowUp":
            player.volume(player.volume() + 0.05);
            break;
          case "ArrowDown":
            player.volume(player.volume() - 0.05);
            break;
          case "Escape":
            player.paused() ? player.play() : player.pause();
            break;
        }
      })
      player.on("timeupdate", async () => {
        await sleep(250); // this is for UX subtitle purpose wait for minimum 250 ms before changing subtitle
        props.setVideoData((prev) => {return {...prev , currentTime: player.currentTime()}});
      });
    }
  }, []);

  // Dispose the Video.js player when the functional component unmounts


  return (
    <div ref={videoRef} className='size-full absolute z-0' data-vjs-player>
    </div>
  );
}

export default VideoJS;