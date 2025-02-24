"use client"

import { useEffect } from 'react';

const ActionDetection = () => {
    useEffect(() => {
        // You can add any additional functionality here if needed
    }, []);

    return (
        <div className='flex items-center justify-center text-center flex-col h-screen gap-16'>
            <h1>Face Recognition Video Stream</h1>
            <img
                src="http://localhost:5000/action-detection" // Adjust if needed
                alt="Video Stream"
                className='w-[800px] h-[700px]  '
            />
        </div>
    );
};

export default ActionDetection;