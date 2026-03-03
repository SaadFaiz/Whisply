import React from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';
const TitleBar = () => {
  const startWindowDrag = async (e) => {
    // Only allow dragging if the mouse event didn't originate from a button
    if (e.target.tagName !== 'BUTTON') {
      await appWindow.startDragging();
    }
  };
  return (

    <div
      data-tauri-drag-region
      className='w-full h-10 bg-gray-800 text-white flex items-center justify-between px-4'
      onMouseDown={ () => startWindowDrag() }
     >
        <h1>Title Bar Component</h1>
        <div className='space-x-2 w-fit h-full flex items-center'>
            <button className='w-8 h-8 flex items-center justify-center rounded hover:bg-gray-700' onClick={async () => await getCurrentWindow.minimize()}>-</button>
            <button className='w-8 h-8 flex items-center justify-center rounded hover:bg-gray-700' onClick={async () => await getCurrentWindow.togglemaximize()}>+</button>
            <button className='w-8 h-8 flex items-center justify-center rounded hover:bg-red-600' onClick={async () => await getCurrentWindow.close()}>x</button>
        </div>
    </div>
  )
}
export default TitleBar;
