import React, { useState } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Handler for file selection
  const handleFileSelect = (file) => {
    setSelectedFile(file);
    // TODO: Implement actual file analysis
    console.log('File selected:', file);
  };

  // Handler for starting analysis
  const handleAnalysis = () => {
    if (selectedFile) {
      setIsAnalyzing(true);
      // TODO: Implement actual analysis logic
      setTimeout(() => setIsAnalyzing(false), 3000); // Mock delay
    }
  };

  return (
    <div className="min-h-screen bg-indigo-50 font-serif">
      {/* Header with indigo tones */}
      <header className="bg-slate-50 border-b-2 border-indigo-200">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-indigo-700 rounded-sm flex items-center justify-center rotate-3">
                <div className="w-6 h-6 bg-indigo-100 rounded-full"></div>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-indigo-900 tracking-tight">
                  AudioSort
                </h1>
                <p className="text-indigo-700 text-sm font-medium">
                  Genre Classification Tool
                </p>
              </div>
            </div>
            <nav className="hidden md:flex space-x-8">
              <button className="text-indigo-800 hover:text-indigo-900 font-medium border-b-2 border-transparent hover:border-indigo-600 transition-all">
                Classify
              </button>
              <button className="text-indigo-800 hover:text-indigo-900 font-medium border-b-2 border-transparent hover:border-indigo-600 transition-all">
                Documentation
              </button>
              <button className="text-indigo-800 hover:text-indigo-900 font-medium border-b-2 border-transparent hover:border-indigo-600 transition-all">
                Support
              </button>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-16">
        {/* Hero with indigo accent */}
        <div className="text-center mb-20">
          <h2 className="text-5xl font-bold text-slate-800 mb-6 leading-tight tracking-tight">
            Intelligent Music 
            <span className="text-indigo-700"> Classification</span>
          </h2>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed font-normal">
            Upload your audio files and receive accurate genre predictions 
            powered by advanced machine learning algorithms.
          </p>
          <div className="mt-8 flex justify-center">
            <div className="w-24 h-1 bg-indigo-500 rounded-full"></div>
          </div>
        </div>

        {/* Upload with indigo accents */}
        <div className="bg-white border-2 border-slate-200 rounded-lg shadow-lg p-10 mb-16">
          <div className="max-w-md mx-auto">
            <div className="border-3 border-dashed border-slate-300 rounded-lg p-16 text-center hover:border-indigo-400 hover:bg-indigo-50/50 transition-all duration-300 cursor-pointer group">
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 bg-slate-100 border-2 border-slate-300 rounded-lg flex items-center justify-center mb-6 group-hover:border-indigo-400 group-hover:bg-indigo-100 transition-all">
                  {isAnalyzing ? (
                    <div className="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div>
                  ) : (
                    <svg className="w-10 h-10 text-slate-500 group-hover:text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  )}
                </div>
                <h3 className="text-xl font-semibold text-slate-700 mb-3">
                  {selectedFile ? selectedFile.name : 'Select Audio File'}
                </h3>
                <p className="text-slate-500 mb-6 font-medium">
                  {isAnalyzing ? 'Analyzing audio...' : 'Drag and drop or click to browse'}
                </p>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={(e) => {
                    if (e.target.files[0]) {
                      handleFileSelect(e.target.files[0]);
                    }
                  }}
                  className="hidden"
                  id="audio-upload"
                  disabled={isAnalyzing}
                />
                <label 
                  htmlFor="audio-upload"
                  className={`${
                    isAnalyzing 
                      ? 'bg-gray-400 cursor-not-allowed' 
                      : 'bg-indigo-600 hover:bg-indigo-700 cursor-pointer'
                  } text-white font-semibold py-3 px-8 rounded-lg transition-colors shadow-md`}
                >
                  {selectedFile ? 'Choose Different File' : 'Browse Files'}
                </label>
                <p className="text-xs text-slate-400 mt-4 font-medium">
                  Supports: MP3, WAV, FLAC, AAC • Maximum 15MB
                </p>
              </div>
            </div>
            
            {/* Analyze Button */}
            {selectedFile && !isAnalyzing && (
              <div className="text-center mt-6">
                <button
                  onClick={handleAnalysis}
                  className="bg-indigo-700 hover:bg-indigo-800 text-white font-bold py-3 px-8 rounded-lg transition-colors shadow-md"
                >
                  Analyze Genre
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Results with indigo progress bars */}
        <div className="bg-white border-2 border-slate-200 rounded-lg shadow-lg p-10">
          <div className="flex items-center justify-between mb-8">
            <h3 className="text-2xl font-bold text-slate-800">
              Classification Results
            </h3>
            <span className={`px-4 py-2 rounded-full text-sm font-medium ${
              isAnalyzing 
                ? 'bg-yellow-100 text-yellow-700' 
                : selectedFile 
                  ? 'bg-green-100 text-green-700' 
                  : 'bg-slate-100 text-slate-600'
            }`}>
              {isAnalyzing ? 'Analyzing...' : selectedFile ? 'Ready to Analyze' : 'No File Selected'}
            </span>
          </div>

          {!selectedFile && !isAnalyzing ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12 7-12 6z" />
                </svg>
              </div>
              <h4 className="text-lg font-semibold text-slate-700 mb-2">
                Upload an audio file to begin
              </h4>
              <p className="text-slate-500">
                Select an audio file above to see genre classification results
              </p>
            </div>
          ) : isAnalyzing ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <div className="w-8 h-8 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
              </div>
              <h4 className="text-lg font-semibold text-slate-700 mb-2">
                Analyzing your audio file...
              </h4>
              <p className="text-slate-500">
                This may take a few seconds
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {[
                { genre: 'Jazz Fusion', confidence: 88, description: 'Complex harmonies with contemporary elements' },
                { genre: 'Contemporary Jazz', confidence: 9, description: 'Modern jazz characteristics detected' },
                { genre: 'Blues', confidence: 3, description: 'Blues influences present in chord progressions' }
              ].map((result, index) => (
                <div key={index} className="border-l-4 border-indigo-500 bg-slate-50 p-6 rounded-r-lg">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex-1">
                      <h4 className="text-lg font-semibold text-slate-800 mb-1">
                        {result.genre}
                      </h4>
                      <p className="text-slate-600 text-sm font-medium">
                        {result.description}
                      </p>
                    </div>
                    <div className="text-right ml-6">
                      <span className="text-2xl font-bold text-slate-800">
                        {result.confidence}%
                      </span>
                      <p className="text-xs text-slate-500 font-medium">
                        Confidence
                      </p>
                    </div>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-indigo-500 to-indigo-600 h-3 rounded-full transition-all duration-1000 ease-out"
                      style={{width: `${result.confidence}%`}}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="mt-8 p-6 bg-indigo-50 border border-indigo-200 rounded-lg">
            <div className="flex items-start space-x-3">
              <div className="w-6 h-6 bg-indigo-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <h5 className="font-semibold text-indigo-900 mb-1">
                  Understanding Your Results
                </h5>
                <p className="text-indigo-800 text-sm leading-relaxed">
                  Our AI model analyzes tempo, harmony, rhythm patterns, and spectral features 
                  to determine the most likely genre. Higher confidence scores indicate stronger 
                  similarity to training data for that genre.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Features section */}
        <div className="mt-20 grid md:grid-cols-2 gap-12">
          <div className="bg-white border-2 border-slate-200 rounded-lg p-8 shadow-lg">
            <div className="w-14 h-14 bg-indigo-100 border-2 border-indigo-300 rounded-lg flex items-center justify-center mb-6">
              <svg className="w-7 h-7 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-3">
              Rapid Processing
            </h3>
            <p className="text-slate-600 leading-relaxed">
              Advanced neural networks provide genre classification 
              results in under 5 seconds for most audio files.
            </p>
          </div>

          <div className="bg-white border-2 border-slate-200 rounded-lg p-8 shadow-lg">
            <div className="w-14 h-14 bg-slate-100 border-2 border-slate-300 rounded-lg flex items-center justify-center mb-6">
              <svg className="w-7 h-7 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-3">
              Proven Accuracy
            </h3>
            <p className="text-slate-600 leading-relaxed">
              Trained on diverse datasets with over 92% accuracy 
              across major music genres and subgenres.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-slate-100 border-t-2 border-slate-200 mt-24 py-12">
        <div className="max-w-4xl mx-auto px-6">
          <div className="grid md:grid-cols-3 gap-8 mb-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-8 h-8 bg-indigo-700 rounded-sm rotate-3">
                  <div className="w-full h-full bg-indigo-100 rounded-full m-1"></div>
                </div>
                <span className="font-bold text-slate-800 text-lg">AudioSort</span>
              </div>
              <p className="text-slate-600 text-sm leading-relaxed">
                Professional music genre classification powered by machine learning.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-slate-800 mb-3">Resources</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><button className="hover:text-slate-800 transition-colors text-left">API Documentation</button></li>
                <li><button className="hover:text-slate-800 transition-colors text-left">Supported Formats</button></li>
                <li><button className="hover:text-slate-800 transition-colors text-left">Model Information</button></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold text-slate-800 mb-3">Support</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li><button className="hover:text-slate-800 transition-colors text-left">Help Center</button></li>
                <li><button className="hover:text-slate-800 transition-colors text-left">Contact Us</button></li>
                <li><button className="hover:text-slate-800 transition-colors text-left">Privacy Policy</button></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-slate-300 pt-8 text-center">
            <p className="text-slate-500 text-sm">
              © 2025 AudioSort by <span className="font-medium text-slate-700">shreyaupretyy</span>. 
              Built with care using React and Tailwind CSS.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;