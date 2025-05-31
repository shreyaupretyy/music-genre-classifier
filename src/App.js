import React from 'react';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-gray-900">
              🎵 Music Genre Classifier
            </h1>
            <p className="text-gray-600">AI-Powered Music Analysis</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Discover Your Music's Genre
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Upload any audio file and let our AI classify its genre instantly.
          </p>
        </div>

        {/* Upload Area */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer bg-white shadow-sm">
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
              <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-700 mb-2">
              Drop your audio file here
            </h3>
            <p className="text-gray-500 mb-4">or click to browse</p>
            <button className="bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-6 rounded-lg transition-colors">
              Choose File
            </button>
          </div>
        </div>

        {/* Sample Results */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-shadow">
            <h3 className="text-lg font-semibold mb-3 flex items-center">
              <span className="mr-2">🎸</span> Rock
            </h3>
            <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
              <div className="bg-red-500 h-3 rounded-full" style={{width: '85%'}}></div>
            </div>
            <p className="text-sm text-gray-600">85% confidence</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-shadow">
            <h3 className="text-lg font-semibold mb-3 flex items-center">
              <span className="mr-2">🎹</span> Pop
            </h3>
            <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
              <div className="bg-blue-500 h-3 rounded-full" style={{width: '12%'}}></div>
            </div>
            <p className="text-sm text-gray-600">12% confidence</p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-shadow">
            <h3 className="text-lg font-semibold mb-3 flex items-center">
              <span className="mr-2">🎤</span> Hip Hop
            </h3>
            <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
              <div className="bg-green-500 h-3 rounded-full" style={{width: '3%'}}></div>
            </div>
            <p className="text-sm text-gray-600">3% confidence</p>
          </div>
        </div>

        {/* Features */}
        <div className="mt-16 grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">⚡</span>
            </div>
            <h3 className="text-lg font-semibold mb-2">Fast Analysis</h3>
            <p className="text-gray-600">Get results in seconds</p>
          </div>

          <div className="text-center">
            <div className="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🎯</span>
            </div>
            <h3 className="text-lg font-semibold mb-2">High Accuracy</h3>
            <p className="text-gray-600">Precise classification</p>
          </div>

          <div className="text-center">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🔒</span>
            </div>
            <h3 className="text-lg font-semibold mb-2">Privacy First</h3>
            <p className="text-gray-600">Your files stay secure</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;