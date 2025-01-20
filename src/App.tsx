import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

interface DetectionResults {
  faces_detected: number;
  phone_detected: boolean;
  dark_surroundings: boolean;
  suspicious_texture: boolean;
  smile_detected: boolean;
  blink_detected: boolean;
  eyebrow_movement: boolean;
  emotion: string | null;
  identity_verified: boolean;
  processed_frame: string;
  timestamp: string;
}

function App() {
  const webcamRef = useRef<Webcam>(null);
  const [results, setResults] = useState<DetectionResults | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAttendanceHalted, setIsAttendanceHalted] = useState(false);

  const captureFrame = async () => {
    if (webcamRef.current && !isProcessing) {
      setIsProcessing(true);
      const imageSrc = webcamRef.current.getScreenshot();
      
      if (imageSrc) {
        const base64Data = imageSrc.replace(/^data:image\/jpeg;base64,/, '');
        const blob = await fetch(`data:image/jpeg;base64,${base64Data}`).then(res => res.blob());
        
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
          const response = await axios.post('http://localhost:8000/api/process-frame', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });
          setResults(response.data);
          
          // Check if attendance should be halted
          const shouldHalt = response.data.phone_detected || 
                           response.data.dark_surroundings || 
                           response.data.suspicious_texture;
          setIsAttendanceHalted(shouldHalt);
        } catch (error) {
          console.error('Error processing frame:', error);
        }
      }
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    const interval = setInterval(captureFrame, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">Advanced Face Detection System</h1>
        
        {isAttendanceHalted && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
            <strong className="font-bold">Alert!</strong>
            <span className="block sm:inline"> Attendance processing has been halted due to suspicious activity.</span>
          </div>
        )}
        
        <div className="grid grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Live Camera</h2>
              <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="w-full rounded"
              />
            </div>
            
            {results?.processed_frame && (
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h2 className="text-xl font-semibold mb-4">Processed Frame</h2>
                <img 
                  src={`data:image/jpeg;base64,${results.processed_frame}`} 
                  alt="Processed frame"
                  className="w-full rounded"
                />
              </div>
            )}
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-6">Detection Results</h2>
            {results && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <ResultItem
                    label="Faces Detected"
                    value={results.faces_detected.toString()}
                  />
                  <ResultItem
                    label="Phone Detection"
                    value={results.phone_detected ? "Detected" : "Not Detected"}
                    status={!results.phone_detected}
                  />
                  <ResultItem
                    label="Dark Surroundings"
                    value={results.dark_surroundings ? "Detected" : "Not Detected"}
                    status={!results.dark_surroundings}
                  />
                  <ResultItem
                    label="Suspicious Texture"
                    value={results.suspicious_texture ? "Detected" : "Not Detected"}
                    status={!results.suspicious_texture}
                  />
                  <ResultItem
                    label="Smile Detection"
                    value={results.smile_detected ? "Detected" : "Not Detected"}
                    status={results.smile_detected}
                  />
                  <ResultItem
                    label="Blink Detection"
                    value={results.blink_detected ? "Detected" : "Not Detected"}
                    status={results.blink_detected}
                  />
                  <ResultItem
                    label="Eyebrow Movement"
                    value={results.eyebrow_movement ? "Detected" : "Not Detected"}
                    status={results.eyebrow_movement}
                  />
                  <ResultItem
                    label="Emotion"
                    value={results.emotion || "Unknown"}
                  />
                  <ResultItem
                    label="Identity Verification"
                    value={results.identity_verified ? "Verified" : "Not Verified"}
                    status={results.identity_verified}
                  />
                </div>
                
                <div className="mt-6 pt-4 border-t border-gray-200">
                  <p className="text-sm text-gray-500">
                    Last Updated: {new Date(results.timestamp).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface ResultItemProps {
  label: string;
  value: string;
  status?: boolean;
}

function ResultItem({ label, value, status }: ResultItemProps) {
  return (
    <div className="bg-gray-50 p-3 rounded">
      <p className="text-sm font-medium text-gray-600">{label}</p>
      <p className={`mt-1 font-semibold ${
        status === undefined
          ? 'text-gray-900'
          : status
            ? 'text-green-600'
            : 'text-red-600'
      }`}>
        {value}
      </p>
    </div>
  );
}

export default App;