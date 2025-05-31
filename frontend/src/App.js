import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import GenreResults from './components/GenreResults';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);

  // Mock data for demo purposes
  const mockResults = [
    { genre: 'Jazz Fusion', confidence: 88, description: 'Complex harmonies with contemporary elements' },
    { genre: 'Contemporary Jazz', confidence: 9, description: 'Modern jazz characteristics detected' },
    { genre: 'Blues', confidence: 3, description: 'Blues influences present in chord progressions' }
  ];

  // Handler for file selection
  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null); // Clear previous results
  };

  // Handler for starting analysis
  const handleAnalysis = () => {
    if (selectedFile) {
      setIsAnalyzing(true);
      setResults(null);
      
      // Simulate API call with mock data
      setTimeout(() => {
        setResults(mockResults);
        setIsAnalyzing(false);
      }, 3000);
    }
  };

  return (
    <div className="min-h-screen bg-indigo-50 font-serif">
      {/* Header */}
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
        {/* Hero Section */}
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

        {/* File Upload Component */}
        <FileUpload 
          onFileSelect={handleFileSelect}
          isAnalyzing={isAnalyzing}
          selectedFile={selectedFile}
        />

        {/* Analyze Button */}
        {selectedFile && !isAnalyzing && !results && (
          <div className="text-center mb-16">
            <button
              onClick={handleAnalysis}
              className="bg-indigo-700 hover:bg-indigo-800 text-white font-bold py-4 px-12 rounded-lg transition-colors shadow-lg transform hover:scale-105"
            >
              Analyze Genre
            </button>
          </div>
        )}

        {/* Results Component */}
        <GenreResults 
          results={results}
          isAnalyzing={isAnalyzing}
          selectedFile={selectedFile}
        />

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