import { useEffect, useState } from "react";
import { secondsToHMS } from "../utils/secondToHMS";
export const PlayerUi = (props) => {
    const [isHovered, setIsHovered] = useState(1);
    
    const [currentTime, setCurrentTime] = useState("");
    const [currentSubtitle, setCurrentSubtitle] = useState({
        id: null,
        chunk: 0,
        start: 0,
        end: 0,
        text: ""
    });
    
    const imageSrcs = {
        fullScreen: '/player_ui_assets/full-screen.svg',
        exitFullScreen: '/player_ui_assets/full-screen-exit.svg',
        playButton: '/player_ui_assets/play_button.svg',
        pauseButton: '/player_ui_assets/pause_button.svg',
        volumeOn: '/player_ui_assets/volume_on.svg',
        volumeOff: '/player_ui_assets/volume_off.svg'
    }
    const [imageSrc, setImageSrc] = useState({
        fullScreen: imageSrcs.fullScreen,
        playButton: imageSrcs.playButton,
        volume: imageSrcs.volumeOn
    });
    const style = {
            transition: "opacity 0.3s",
            opacity: isHovered
    }
    const timelineTrackStyle = {
        width: `${props.videoData.currentTime / props.videoData.duration * 100}%`
    }
    const timelineThumbStyle = {
        left: `${props.videoData.currentTime / props.videoData.duration * 100}%`
    }
    useEffect(() => {
        if(!props.videoStatus.loaded) return;
        if(imageSrc.playButton === imageSrcs.playButton){
          setImageSrc({...imageSrc, playButton: imageSrcs.pauseButton});
          return;
        }
        setImageSrc({...imageSrc, playButton: imageSrcs.playButton});
        setIsHovered(1);
    },[props.videoStatus.loaded])
    useEffect(() => {
      const {hours, minutes, seconds} = secondsToHMS(props.videoData.currentTime);
      const durationSplited = secondsToHMS(props.videoData.duration);
      setCurrentTime(`${durationSplited.hours > 0 ? Math.floor(hours) + ":" : ""}${minutes < 10 ? "0" + Math.floor(minutes) : Math.floor(minutes)}:${seconds < 10 ? "0" + Math.floor(seconds) : Math.floor(seconds)}`);
      if(currentTime > currentSubtitle.start && currentTime < currentSubtitle.end) return;
      const index = props.subtitles.find((subtitle) => subtitle.start <= props.videoData.currentTime && subtitle.end >= props.videoData.currentTime)
      index ? setCurrentSubtitle(index) : setCurrentSubtitle({id: null,chunk: 0, start: 0, end: 0, text: "", translatedText: ""});
    }, [props.videoData.currentTime]);
    const playButtonUiHandler = async () => {
        await props.playButtonHandler();
        if(!props.videoStatus.loaded) return;
        if(imageSrc.playButton === imageSrcs.playButton){
            setImageSrc({...imageSrc, playButton: imageSrcs.pauseButton});
            return;
        }
        setImageSrc({...imageSrc, playButton: imageSrcs.playButton});
        setIsHovered(1);
    }
    
    const fullScreenButtonUiHandler = async () => {
        await props.fullScreenButtonHandler();
        console.log(props.videoStatus.loaded);
        if(!props.videoStatus.loaded) return;
        console.log(props.fullScreen);
        if(props.fullScreen){
            setImageSrc({...imageSrc, fullScreen: imageSrcs.fullScreen});
            return;
        }
        setImageSrc({...imageSrc, fullScreen: imageSrcs.exitFullScreen});
        console.log("full screen : ", props.fullScreen);
    }
    useEffect(() => {
      console.log(props.subtitles);
    }, [props.subtitles])
    const volumeButtonUiHandler = async () => {
        await props.volumeButtonHandler();
        if(!props.videoStatus.loaded) return;
        if(imageSrc.volume === imageSrcs.volumeOn){
            setImageSrc({...imageSrc, volume: imageSrcs.volumeOff});
            return;
        }
        setImageSrc({...imageSrc, volume: imageSrcs.volumeOn});
    }
    
  return (
    <div className="absolute size-full flex flex-col justify-between z-20" onMouseEnter={() => props.videoStatus.loaded && setIsHovered(1)} onMouseLeave={() => props.videoStatus.loaded && props.videoStatus.playing && setIsHovered(0)}>
      <div onClick={() => playButtonUiHandler()} ref={(el) => props.setRefs('videoContainer', el)} className="h-[90%] w-full">
      </div>
      <div className="h-[8vh] w-full z-20 bg-transparent px-6 flex flex-col justify-evenly text-black bottom-0"
        style={style}
      >
        <div className="w-full h-fit flex gap-6 justify-between items-center">
            <p className="text-white text-sm w-fit">{currentTime}</p>
            <button ref={(el) => props.setRefs('timeline', el)} className="flex-1 relative bg-neutral-600 rounded-full h-[1vh]">
                <div style={timelineTrackStyle} className="bg-red-700 h-[1vh] rounded-full">
                </div>
                <div style={timelineThumbStyle} className="absolute top-1/2 -translate-y-1/2">
                  <div className="max-h-10 max-w-10 size-[2vh]  bg-red-600 rounded-full absolute top-1/2 -translate-y-1/2"></div>
                </div>
            </button>
            <p className="text-white text-sm w-fit">{props.videoData.length}</p>
        </div>
        <div className="flex-1 flex items-center">
            <div className="flex gap-5">
                <button onClick={() => playButtonUiHandler()} ref={(el) => props.setRefs('playStop', el)} className="size-[3vh]"><img className="size-full" src={imageSrc.playButton}></img></button>
                <button onClick={() => volumeButtonUiHandler()} ref={(el) => props.setRefs('mute', el)} className="size-[3vh]"><img className="size-full" src={imageSrc.volume}></img></button>
            </div>
            <div className="flex-1 flex justify-end">
                <button onClick={() => fullScreenButtonUiHandler()} ref={(el) => props.setRefs('fullscreen', el)} className="size-[2.5vh]"><img className="size-full " src={imageSrc.fullScreen}></img></button>
            </div>
        </div>
      </div>
      {
        currentSubtitle && (
          <div className="absolute flex flex-col gap-1 bottom-[8vh] w-[80%] text-center h-fit left-1/2 -translate-x-1/2">
            <p className="text-white text-[1.8vw] bg-black/50 font-arial">{currentSubtitle.text}</p>
            {currentSubtitle.translated_text && <p className="text-white text-[1.5vw] bg-black/50 font-arial">{currentSubtitle.translated_text}</p>}
          </div>
        ) 
      }
    </div>
  )
}

export default PlayerUi