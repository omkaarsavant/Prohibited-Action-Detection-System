"use client"

import { useEffect } from 'react';

const video_feed = () => {
    useEffect(() => {
        // You can add any additional functionality here if needed
    }, []);

    return (
        <div className='flex items-center justify-center text-center flex-col h-screen gap-16'>
            <h1>Face Recognition Video Stream</h1>
            <img
                src="http://localhost:5000/video_feed" // Adjust if needed
                alt="Video Stream"
                className='w-[800px] h-[700px]  '
            />
        </div>
    );
};

export default video_feed;
