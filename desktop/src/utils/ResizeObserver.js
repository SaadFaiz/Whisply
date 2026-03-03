import { useEffect } from "react";

const videoPlayerResizeObserver = (videoPlayerZone) => {
  useEffect(() => {

      const observer = new ResizeObserver((elements) => {
        if (videoPlayerZone.current) {
          videoPlayerZone.current.contentRect = elements[0].contentRect;
          console.log("[DEBBUG] : video player zone ", videoPlayerZone.current)
        } else {
          console.log("[DEBBUG] : No video player zone variable")
        }
        }
      );
      
      if (videoPlayerZone.current.element) {
        observer.observe(videoPlayerZone.current.element);
      }
      
      return () => observer.disconnect();
  }, []);
};

export { videoPlayerResizeObserver };