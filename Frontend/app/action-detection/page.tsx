"use client";
import { useEffect, useState } from 'react';
import axios from 'axios';

const ActionDetection = () => {
    const [prisonerNumber, setPrisonerNumber] = useState('');
    const [prisonerDetails, setPrisonerDetails] = useState(null);

    const handleSearch = async () => {
        try {
            const response = await axios.get(`http://127.0.0.1:5000/user/${prisonerNumber}`);
            setPrisonerDetails(response.data);
        } catch (error) {
            console.error("Error fetching prisoner details:", error);
            setPrisonerDetails(null); // Reset details on error
        }
    };

    return (
        <div className="bg-gray-900 text-white">
            <div className="flex h-screen">
                {/* Sidebar */}
                <div className="w-1/4 bg-gray-800 p-4">
                    <div className="mb-4">
                        <h3 className="text-sm font-semibold mb-2">
                            Search Prisoner
                        </h3>
                        <input
                            className="w-full p-2 bg-gray-700 rounded"
                            placeholder="Enter prisoner ID"
                            type="text"
                            value={prisonerNumber}
                            onChange={(e) => setPrisonerNumber(e.target.value)}
                        />
                        <button
                            onClick={handleSearch}
                            className="mt-2 bg-blue-500 text-white py-2 px-4 rounded"
                        >
                            Search
                        </button>
                    </div>
                    <div id="prisoner-details" className="mt-4">
                        {prisonerDetails ? (
                            <div>
                                <h4 className="font-semibold">Name: {prisonerDetails.Name}</h4>
                                <p>Prisoner Number: {prisonerDetails.PrisonerNumber}</p>
                                <p>Age: {prisonerDetails.Age}</p>
                                <p>Height: {prisonerDetails.Height}</p>
                                <p>Weight: {prisonerDetails.Weight}</p>
                                <p>Gender: {prisonerDetails.Gender}</p>
                                <h5 className="mt-2">Profile Pictures:</h5>
                                <div className="flex flex-wrap">
                                    {prisonerDetails.ImagePaths.map((path, index) => (
                                        <img key={index} src={`http://127.0.0.1:5000/${path}`} alt={`Profile ${index + 1}`} className="w-24 h-24 object-cover rounded m-1" />
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <p>No prisoner details found.</p>
                        )}
                    </div>
                </div>
                {/* Main Content */}
                <div className="flex-1 bg-gray-900 p-4">
                    <div className="flex justify-between items-center mb-4">
                        <h1 className="text-xl font-semibold">
                            Surveillance Camera
                        </h1>
                        <div className="text-sm text-gray-400">
                            Detection Level:
                            <span className="text-white">
                                High
                            </span>
                        </div>
                    </div>
                    <div className="relative bg-gray-700 h-700">
                        <img alt="Video Stream" className="w-half h-half object-cover" src="http://localhost:5000/action-detection" />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ActionDetection;