"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import { supabase } from '@/lib/supabaseClient';

const VideoFeed = () => {
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  // Auth check
  useEffect(() => {
    const checkSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        router.push('/login');
      } else {
        setLoading(false);
      }
    };
    checkSession();
  }, []);

  // State for prisoner data
  const [prisonerNumber, setPrisonerNumber] = useState('');
  const [prisonerDetails, setPrisonerDetails] = useState(null);

  const handleSearch = async () => {
    try {
      const response = await axios.get(`http://127.0.0.1:5000/user/${prisonerNumber}`);
      setPrisonerDetails(response.data);
    } catch (error) {
      console.error("Error fetching prisoner details:", error);
      setPrisonerDetails(null);
    }
  };

  if (loading) {
    return <div className="text-white p-10 text-center">Loading...</div>;
  }

  return (
    <div className="bg-gray-900 text-white">
      <div className="flex h-screen">
        {/* Sidebar */}
        <div className="w-1/4 bg-gray-800 p-4">
          <br />
          <h1 className="text-xl font-semibold">Search Prisoner</h1>
          <br />
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
                    <img
                      key={index}
                      src={`http://127.0.0.1:5000/images/${path}`}
                      alt={`Profile ${index + 1}`}
                      className="w-24 h-24 object-cover rounded m-1"
                    />
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
              <br />
              Surveillance Camera
            </h1>
            <div className="flex justify-between items-center mb-4">
              <br />
              Detection Level: High
            </div>
          </div>
          <div className="relative bg-gray-700 flex justify-center items-center">
            <img
              alt="Video Stream"
              className="w-[80vw] h-[80vh]"
              src="http://localhost:5000/video_feed"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoFeed;
