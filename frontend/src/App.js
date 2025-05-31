import React, { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import GenreResults from './components/GenreResults';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState(null);

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/health');
      const data = await response.json();
      setBackendStatus(data);
    } catch (err) {
      setBackendStatus({ status: 'offline', error: 'Cannot connect to backend' });
    }
  };

  // Handler for file selection
  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
    setError(null);
  };

  // Handler for starting analysis with real API
  const handleAnalysis = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setResults(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('audio', selectedFile);

      console.log('Sending file for classification:', selectedFile.name);

      const response = await fetch('http://localhost:5000/api/classify', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server error: ${response.status}`);
      }

      console.log('Classification results:', data);
      setResults(data.results);

      // Show additional info if available
      if (data.total_genres_detected > 3) {
        console.log(`Total ${data.total_genres_detected} genres detected, showing top 3`);
      }

    } catch (err) {
      console.error('Classification error:', err);
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-indigo-50 font-serif">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-indigo-100">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
              </div>
              <h1 className="text-2xl font-bold text-gray-900">AudioSort</h1>
            </div>
            
            {/* Backend Status Indicator */}
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                backendStatus?.status === 'healthy' && backendStatus?.model_loaded 
                  ? 'bg-green-500' 
                  : backendStatus?.status === 'healthy' 
                  ? 'bg-yellow-500' 
                  : 'bg-red-500'
              }`}></div>
              <span className="text-sm text-gray-600">
                {backendStatus?.status === 'healthy' && backendStatus?.model_loaded 
                  ? 'Model Ready' 
                  : backendStatus?.status === 'healthy' 
                  ? 'Model Not Loaded' 
                  : 'Backend Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-16">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold text-gray-900 mb-6">
            Discover Your Music's Genre
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Upload any audio file and let our AI analyze its musical characteristics 
            to identify the genre with precision and confidence.
          </p>
        </div>

        {/* Backend Status Warning */}
        {backendStatus && !backendStatus.model_loaded && (
          <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              <p className="text-yellow-700 text-sm font-medium">
                {backendStatus.status === 'offline' 
                  ? 'Backend server is offline. Please start the backend server.'
                  : 'AI model is not loaded. Please train the model first.'}
              </p>
            </div>
          </div>
        )}

        <FileUpload 
          onFileSelect={handleFileSelect}
          isAnalyzing={isAnalyzing}
          selectedFile={selectedFile}
        />

        {selectedFile && !isAnalyzing && !results && backendStatus?.model_loaded && (
          <div className="text-center mb-16">
            <button
              onClick={handleAnalysis}
              className="bg-indigo-700 hover:bg-indigo-800 text-white font-bold py-4 px-12 rounded-lg transition-colors shadow-lg transform hover:scale-105"
            >
              Analyze Genre
            </button>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-700 text-sm font-medium">Error: {error}</p>
            </div>
          </div>
        )}

        <GenreResults 
          results={results}
          isAnalyzing={isAnalyzing}
          selectedFile={selectedFile}
        />

        {/* Features Section */}
        <div className="mt-32 grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Lightning Fast</h3>
            <p className="text-gray-600">Advanced AI processes your audio in seconds with high accuracy.</p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Multiple Formats</h3>
            <p className="text-gray-600">Supports MP3, WAV, FLAC, and other popular audio formats.</p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Privacy First</h3>
            <p className="text-gray-600">Your audio files are processed locally and never stored permanently.</p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-32">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="text-center text-gray-500">
            <p>&copy; 2025 AudioSort. AI-powered music genre classification.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;