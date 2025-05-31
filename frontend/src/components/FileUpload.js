import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const FileUpload = ({ onFileSelect, isAnalyzing, selectedFile }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      console.log('File selected:', file.name, 'Size:', file.size, 'Type:', file.type);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'audio/mpeg': ['.mp3'],
      'audio/wav': ['.wav'],
      'audio/flac': ['.flac'],
      'audio/mp4': ['.m4a'],
      'audio/aac': ['.aac'],
      'audio/ogg': ['.ogg']
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB limit
    disabled: isAnalyzing
  });

  // Handle file rejection errors
  const fileRejectionItems = fileRejections.map(({ file, errors }) => (
    <div key={file.path} className="text-red-600 text-sm mt-2">
      <p className="font-medium">{file.path}:</p>
      <ul className="list-disc list-inside ml-2">
        {errors.map(e => (
          <li key={e.code}>
            {e.code === 'file-too-large' 
              ? 'File is too large (max 50MB)' 
              : e.code === 'file-invalid-type'
              ? 'Invalid file type. Please use MP3, WAV, FLAC, M4A, AAC, or OGG files.'
              : e.message}
          </li>
        ))}
      </ul>
    </div>
  ));

  return (
    <div className="mb-16">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all
          ${isDragActive 
            ? 'border-indigo-500 bg-indigo-50' 
            : selectedFile
            ? 'border-green-500 bg-green-50'
            : 'border-gray-300 bg-white hover:border-indigo-400 hover:bg-indigo-50'
          }
          ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="space-y-4">
          {/* Icon */}
          <div className="mx-auto w-16 h-16 flex items-center justify-center">
            {isAnalyzing ? (
              <div className="animate-spin rounded-full h-16 w-16 border-4 border-indigo-200 border-t-indigo-600"></div>
            ) : selectedFile ? (
              <svg className="w-16 h-16 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
            ) : (
              <svg className="w-16 h-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            )}
          </div>
          
          {/* Text */}
          <div>
            {isAnalyzing ? (
              <div>
                <p className="text-xl font-bold text-indigo-600">Analyzing Audio...</p>
                <p className="text-sm text-gray-600 mt-2">
                  Our AI is extracting musical features and identifying the genre
                </p>
              </div>
            ) : selectedFile ? (
              <div>
                <p className="text-xl font-bold text-green-600">File Selected</p>
                <p className="text-lg text-gray-700 mt-2">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">
                  Size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
                <p className="text-sm text-indigo-600 mt-2">Ready to analyze!</p>
              </div>
            ) : isDragActive ? (
              <div>
                <p className="text-xl font-bold text-indigo-600">Drop your audio file here!</p>
                <p className="text-sm text-gray-600 mt-2">Release to upload</p>
              </div>
            ) : (
              <div>
                <p className="text-xl font-bold text-gray-700">Upload Audio File</p>
                <p className="text-gray-600 mt-2">
                  Drag & drop your audio file here, or click to browse
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Supports MP3, WAV, FLAC, M4A, AAC, OGG (max 50MB)
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* File rejection errors */}
      {fileRejectionItems.length > 0 && (
        <div className="mt-4">
          {fileRejectionItems}
        </div>
      )}
      
      {/* Selected file info */}
      {selectedFile && !isAnalyzing && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-900 mb-2">File Details:</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Name:</span>
              <p className="font-medium">{selectedFile.name}</p>
            </div>
            <div>
              <span className="text-gray-600">Size:</span>
              <p className="font-medium">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
            <div>
              <span className="text-gray-600">Type:</span>
              <p className="font-medium">{selectedFile.type || 'Unknown'}</p>
            </div>
            <div>
              <span className="text-gray-600">Last Modified:</span>
              <p className="font-medium">{new Date(selectedFile.lastModified).toLocaleDateString()}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;