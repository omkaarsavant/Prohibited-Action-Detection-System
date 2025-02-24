"use client"

import { RegistrationForm } from '@/components/RegistrationForm';
import { useEffect } from 'react';

const RegistrationPage = () => {
    useEffect(() => {
        // You can add any additional functionality here if needed
    }, []);

    return (
        <div className='flex items-center justify-center text-center flex-col h-screen gap-16'>
            <RegistrationForm />
        </div>
    );
};

export default RegistrationPage;