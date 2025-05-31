 import React from 'react';

const GenreResults = ({ results, isAnalyzing, selectedFile }) => {
  const getConfidenceColor = (confidence) => {
    if (confidence >= 70) return 'from-green-500 to-green-600';
    if (confidence >= 40) return 'from-yellow-500 to-yellow-600';
    if (confidence >= 20) return 'from-orange-500 to-orange-600';
    return 'from-red-500 to-red-600';
  };

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 70) return 'High Confidence';
    if (confidence >= 40) return 'Medium Confidence';
    if (confidence >= 20) return 'Low Confidence';
    return 'Very Low Confidence';
  };

  const renderLoadingState = () => (
    <div className="text-center py-12">
      <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <div className="w-8 h-8 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
      </div>
      <h4 className="text-lg font-semibold text-slate-700 mb-2">
        Analyzing your audio file...
      </h4>
      <p className="text-slate-500 mb-4">
        This may take a few seconds
      </p>
      <div className="flex justify-center space-x-1">
        <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-100"></div>
        <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-200"></div>
      </div>
    </div>
  );

  const renderEmptyState = () => (
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
  );

  const renderReadyState = () => (
    <div className="text-center py-12">
      <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <h4 className="text-lg font-semibold text-slate-700 mb-2">
        Ready to analyze
      </h4>
      <p className="text-slate-500">
        Click "Analyze Genre" to process your audio file
      </p>
    </div>
  );

  const renderResults = () => (
    <div className="space-y-6">
      {results.map((result, index) => (
        <div key={index} className="border-l-4 border-indigo-500 bg-slate-50 p-6 rounded-r-lg">
          <div className="flex justify-between items-start mb-4">
            <div className="flex-1">
              <div className="flex items-center space-x-3 mb-2">
                <h4 className="text-lg font-semibold text-slate-800">
                  {result.genre}
                </h4>
                {index === 0 && (
                  <span className="px-2 py-1 text-xs bg-indigo-100 text-indigo-700 rounded-full font-medium">
                    Most Likely
                  </span>
                )}
              </div>
              <p className="text-slate-600 text-sm font-medium">
                {result.description}
              </p>
            </div>
            <div className="text-right ml-6">
              <span className="text-2xl font-bold text-slate-800">
                {result.confidence}%
              </span>
              <p className="text-xs text-slate-500 font-medium">
                {getConfidenceLabel(result.confidence)}
              </p>
            </div>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
            <div 
              className={`bg-gradient-to-r ${getConfidenceColor(result.confidence)} h-3 rounded-full transition-all duration-1000 ease-out`}
              style={{
                width: `${result.confidence}%`,
                animationDelay: `${index * 200}ms`
              }}
            ></div>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="bg-white border-2 border-slate-200 rounded-lg shadow-lg p-10">
      <div className="flex items-center justify-between mb-8">
        <h3 className="text-2xl font-bold text-slate-800">
          Classification Results
        </h3>
        <span className={`px-4 py-2 rounded-full text-sm font-medium ${
          isAnalyzing 
            ? 'bg-yellow-100 text-yellow-700' 
            : results && results.length > 0
              ? 'bg-green-100 text-green-700' 
              : selectedFile 
                ? 'bg-blue-100 text-blue-700' 
                : 'bg-slate-100 text-slate-600'
        }`}>
          {isAnalyzing 
            ? 'Analyzing...' 
            : results && results.length > 0
              ? 'Analysis Complete'
              : selectedFile 
                ? 'Ready to Analyze' 
                : 'No File Selected'
          }
        </span>
      </div>

      {isAnalyzing 
        ? renderLoadingState()
        : !selectedFile
          ? renderEmptyState()
          : !results || results.length === 0
            ? renderReadyState()
            : renderResults()
      }

      {results && results.length > 0 && (
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
      )}
    </div>
  );
};

export default GenreResults;