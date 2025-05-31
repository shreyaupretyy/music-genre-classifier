import React from 'react';

const GenreResults = ({ results, isAnalyzing, selectedFile }) => {
  // Don't render anything if no file is selected
  if (!selectedFile) {
    return null;
  }

  // Show analyzing state
  if (isAnalyzing) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-16">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-200 border-t-indigo-600 mx-auto mb-4"></div>
          <h3 className="text-2xl font-bold text-gray-900 mb-2">Analyzing Your Music</h3>
          <p className="text-gray-600">
            Our AI is processing the audio file and extracting musical features...
          </p>
          <div className="mt-6 space-y-2">
            <div className="text-sm text-gray-500">
              ♪ Extracting MFCC features...
            </div>
            <div className="text-sm text-gray-500">
              ♫ Analyzing spectral characteristics...
            </div>
            <div className="text-sm text-gray-500">
              ♬ Detecting rhythm patterns...
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Show results if available
  if (results && results.length > 0) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-16">
        <div className="text-center mb-8">
          <h3 className="text-3xl font-bold text-gray-900 mb-2">Genre Analysis Results</h3>
          <p className="text-gray-600">
            Based on AI analysis of musical characteristics
          </p>
        </div>

        <div className="space-y-6">
          {results.map((result, index) => (
            <div
              key={index}
              className={`
                p-6 rounded-xl border-2 transition-all
                ${index === 0 
                  ? 'border-indigo-300 bg-indigo-50' 
                  : 'border-gray-200 bg-gray-50'
                }
              `}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  {index === 0 && (
                    <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  )}
                  <h4 className={`text-xl font-bold ${index === 0 ? 'text-indigo-900' : 'text-gray-900'}`}>
                    {result.genre}
                    {index === 0 && (
                      <span className="ml-2 text-sm font-normal text-indigo-600">Primary Match</span>
                    )}
                  </h4>
                </div>
                
                <div className="text-right">
                  <div className={`text-2xl font-bold ${index === 0 ? 'text-indigo-600' : 'text-gray-700'}`}>
                    {result.confidence}%
                  </div>
                  <div className="text-sm text-gray-500">confidence</div>
                </div>
              </div>

              {/* Confidence bar */}
              <div className="mb-3">
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all duration-1000 ${
                      index === 0 ? 'bg-indigo-600' : 'bg-gray-400'
                    }`}
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              <p className="text-gray-600 text-sm">
                {result.description}
              </p>

              {/* Genre characteristics (you can customize these) */}
              {index === 0 && (
                <div className="mt-4 pt-4 border-t border-indigo-200">
                  <div className="text-sm text-indigo-700">
                    <strong>Why this genre?</strong> Our AI detected musical patterns and characteristics 
                    commonly associated with {result.genre.toLowerCase()} music in your audio file.
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Additional info section */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <div className="text-center">
            <p className="text-sm text-gray-500 mb-2">
              Analysis completed using deep learning audio feature extraction
            </p>
            <div className="flex justify-center space-x-6 text-xs text-gray-400">
              <span>• MFCC Analysis</span>
              <span>• Spectral Features</span>
              <span>• Rhythm Patterns</span>
              <span>• Harmonic Content</span>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="mt-6 text-center">
          <button
            onClick={() => window.location.reload()}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
          >
            Analyze Another File
          </button>
        </div>
      </div>
    );
  }

  // Show ready state (file selected but not analyzed yet)
  return (
    <div className="bg-white rounded-2xl shadow-xl p-8 mb-16">
      <div className="text-center">
        <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
          </svg>
        </div>
        <h3 className="text-2xl font-bold text-gray-900 mb-2">Ready to Analyze</h3>
        <p className="text-gray-600">
          Click the "Analyze Genre" button to start AI-powered music classification
        </p>
        <div className="mt-4 text-sm text-gray-500">
          File: <span className="font-medium text-gray-700">{selectedFile.name}</span>
        </div>
      </div>
    </div>
  );
};

export default GenreResults;