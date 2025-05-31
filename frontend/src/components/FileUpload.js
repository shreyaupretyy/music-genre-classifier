 
import React, { useState, useRef, useCallback } from 'react';

const FileUpload = ({ onFileSelect, isAnalyzing, selectedFile }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  // Supported file types
  const supportedTypes = [
    'audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/flac', 
    'audio/m4a', 'audio/aac', 'audio/ogg'
  ];

  const validateFile = useCallback((file) => {
    // Check file type
    if (!supportedTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|flac|m4a|aac|ogg)$/i)) {
      return 'Please select a valid audio file (MP3, WAV, FLAC, M4A, AAC, OGG)';
    }

    // Check file size (15MB max)
    if (file.size > 15 * 1024 * 1024) {
      return 'File size must be less than 15MB';
    }

    return null;
  }, []);

  const handleFileSelection = useCallback((file) => {
    const error = validateFile(file);
    if (error) {
      setUploadError(error);
      return;
    }

    setUploadError(null);
    onFileSelect(file);
  }, [validateFile, onFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  }, [handleFileSelection]);

  const handleFileInputChange = useCallback((e) => {
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  }, [handleFileSelection]);

  const handleBrowseClick = useCallback(() => {
    if (!isAnalyzing) {
      fileInputRef.current?.click();
    }
  }, [isAnalyzing]);

  const clearFile = useCallback(() => {
    setUploadError(null);
    onFileSelect(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [onFileSelect]);

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="bg-white border-2 border-slate-200 rounded-lg shadow-lg p-10 mb-16">
      <div className="max-w-md mx-auto">
        <div
          className={`border-3 border-dashed rounded-lg p-16 text-center transition-all duration-300 cursor-pointer group ${
            isDragOver
              ? 'border-indigo-400 bg-indigo-50'
              : isAnalyzing
              ? 'border-slate-200 bg-slate-50 cursor-not-allowed'
              : 'border-slate-300 hover:border-indigo-400 hover:bg-indigo-50/50'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={!isAnalyzing ? handleBrowseClick : undefined}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileInputChange}
            className="hidden"
            disabled={isAnalyzing}
          />

          <div className="flex flex-col items-center">
            <div className={`w-20 h-20 border-2 rounded-lg flex items-center justify-center mb-6 transition-all ${
              isDragOver
                ? 'border-indigo-400 bg-indigo-100'
                : isAnalyzing
                ? 'border-slate-300 bg-slate-100'
                : 'border-slate-300 bg-slate-100 group-hover:border-indigo-400 group-hover:bg-indigo-100'
            }`}>
              {isAnalyzing ? (
                <div className="w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div>
              ) : selectedFile ? (
                <svg className="w-10 h-10 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className={`w-10 h-10 transition-colors ${
                  isDragOver ? 'text-indigo-600' : 'text-slate-500 group-hover:text-indigo-600'
                }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              )}
            </div>

            {selectedFile ? (
              <div className="text-center">
                <h3 className="text-xl font-semibold text-slate-700 mb-2">
                  {selectedFile.name}
                </h3>
                <p className="text-slate-500 mb-4">
                  {formatFileSize(selectedFile.size)}
                </p>
                {!isAnalyzing && (
                  <div className="space-x-3">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleBrowseClick();
                      }}
                      className="text-indigo-600 hover:text-indigo-700 font-medium text-sm border-b border-indigo-200 hover:border-indigo-300 transition-colors"
                    >
                      Choose Different File
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        clearFile();
                      }}
                      className="text-red-600 hover:text-red-700 font-medium text-sm border-b border-red-200 hover:border-red-300 transition-colors ml-4"
                    >
                      Remove File
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center">
                <h3 className="text-xl font-semibold text-slate-700 mb-3">
                  {isAnalyzing ? 'Processing audio...' : 'Select Audio File'}
                </h3>
                <p className="text-slate-500 mb-6 font-medium">
                  {isAnalyzing ? 'Please wait while we analyze your file' : 'Drag and drop or click to browse'}
                </p>
                {!isAnalyzing && (
                  <button className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors shadow-md">
                    Browse Files
                  </button>
                )}
              </div>
            )}

            {!isAnalyzing && (
              <p className="text-xs text-slate-400 mt-4 font-medium">
                Supports: MP3, WAV, FLAC, M4A, AAC, OGG • Maximum 15MB
              </p>
            )}
          </div>
        </div>

        {/* Error Message */}
        {uploadError && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-700 text-sm font-medium">{uploadError}</p>
            </div>
          </div>
        )}

        {/* File Info */}
        {selectedFile && !uploadError && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-green-700 text-sm font-medium">
                File ready for analysis
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;