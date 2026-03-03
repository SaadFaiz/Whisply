import { getCurrentWebview } from "@tauri-apps/api/webview";
import { useEffect } from "react";
import axios from "axios";
const HandleDrop = async (e) => {
    try {
        e.preventDefault();
        e.stopPropagation();
    } catch (error) {
        console.error("Error handling drop event:", error);
    }
}
const HandleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    // Required to allow drop!
};
const HandleDragEnd = async (e, TargetedZone) => {
    e.preventDefault();
    e.stopPropagation();
    TargetedZone.current = null;
    console.log("Drag exit event", TargetedZone.current);
}
const HandleDragEnter = async (e, TargetedZone) => {
    e.preventDefault();
    e.stopPropagation();
    TargetedZone.current = e.currentTarget;
    console.log("Drag enter event", TargetedZone.current);
};
const inTargetedZone = (position, targetRect) => {
  if(!targetRect) return false;
  return (
    position.x >= targetRect.left &&
    position.x <= targetRect.right &&
    position.y >= targetRect.top &&
    position.y <= targetRect.bottom
  );
};
const useDragDrop = (videoPlayerZone, TargetedZone, setFileInfo) => {

    useEffect(() => {
        let unlisten;
        
        const dragDropSetup = async () => {
            const webview = getCurrentWebview();
            unlisten = await webview.onDragDropEvent((async e => {
                const { type, paths, position } = e.payload;
                if(type === "drop"){
                    setFileInfo(e.payload);
                    const screenSize = await webview.size()
                    console.log("Screen size:", screenSize);
                    position.x = (position.x / screenSize.width) * window.innerWidth; // scale position.x to current window size
                    position.y = (position.y / screenSize.height) * window.innerHeight; // scale position.y to current window size
                    console.log("Normalized position:", position);
                    console.log("video path :", paths);
                    const isTargeted = inTargetedZone(position, videoPlayerZone.current.contentRect); // Check if dropped in video player zone
                    const res = await axios.post("http://localhost:8011/path", {path : paths[0]} )
                    const path = res.data.path;
                    console.log("Received path from server:", path);
                    console.log("Screen width:", window.innerWidth);
                    if(isTargeted) {
                        TargetedZone.current = videoPlayerZone.current;
                        console.log("[DEBBUG] : dropped in video player zone ", true)
                    }else{
                        console.log("[DEBBUG] : dropped in video player zone ", false)
                    }
                }
            })) 
            
        };
        
        dragDropSetup();
        
        return () => {
            if (unlisten) {
                unlisten();
            }
        };
    }, [])
}
export { HandleDrop, HandleDragEnd, HandleDragEnter, HandleDragOver, useDragDrop };