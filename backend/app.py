from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load your trained model (you'll need to train this)
# model = tf.keras.models.load_model('models/genre_classifier.h5')

# Genre labels (adjust based on your model)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(audio_path):
    """Extract audio features using librosa"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Additional features
        chroma = np.mean(librosa.feature.chroma(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        
        # Combine all features
        features = np.hstack([mfcc_scaled, chroma, mel, contrast, tonnetz])
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/api/classify', methods=['POST'])
def classify_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            file.save(tmp_file.name)
            
            # Extract features
            features = extract_features(tmp_file.name)
            if features is None:
                return jsonify({'error': 'Could not extract features from audio'}), 400
            
            # TODO: Replace with actual model prediction
            # predictions = model.predict(features.reshape(1, -1))[0]
            
            # Mock predictions for now (replace with actual model)
            mock_predictions = np.random.rand(len(GENRES))
            mock_predictions = mock_predictions / np.sum(mock_predictions) * 100
            
            # Format results
            results = []
            for i, genre in enumerate(GENRES):
                confidence = float(mock_predictions[i])
                if confidence > 1:  # Only include genres with >1% confidence
                    results.append({
                        'genre': genre.title(),
                        'confidence': round(confidence),
                        'description': f'{genre.title()} characteristics detected in audio analysis'
                    })
            
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            return jsonify({'results': results[:3]})  # Return top 3 predictions
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)