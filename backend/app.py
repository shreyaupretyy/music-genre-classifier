from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import torch
import torch.nn as nn
from werkzeug.utils import secure_filename
import os
import tempfile
import pickle

app = Flask(__name__)
CORS(app)

# Global variables
model = None
label_encoder = None
scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenreClassifier(nn.Module):
    """PyTorch neural network for genre classification"""
    def __init__(self, input_size, num_classes, dropout_rate=0.3):
        super(GenreClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def load_model():
    """Load the trained PyTorch model"""
    global model, label_encoder, scaler
    
    try:
        if os.path.exists('models/genre_classifier.pth'):
            checkpoint = torch.load('models/genre_classifier.pth', map_location=device)
            
            # Create model
            input_size = checkpoint['input_size']
            num_classes = checkpoint['num_classes']
            model = GenreClassifier(input_size, num_classes).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load scaler and label encoder
            scaler = checkpoint['scaler']
            label_encoder = checkpoint['label_encoder']
            
            print("PyTorch model loaded successfully!")
            print(f"Model on device: {next(model.parameters()).device}")
            return True
        else:
            print("Model not found. Please train the model first.")
            return False
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def extract_features(audio_path):
    """Extract audio features using librosa (same as training)"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        # Spectral features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        chroma = librosa.feature.chroma(y=y, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_scaled = np.mean(mel.T, axis=0)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_scaled = np.mean(contrast.T, axis=0)
        
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_scaled = np.mean(tonnetz.T, axis=0)
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Combine features
        features = np.hstack([
            mfcc_scaled,
            chroma_scaled,
            mel_scaled,
            contrast_scaled,
            tonnetz_scaled,
            [tempo],
            [np.mean(spectral_centroids)],
            [np.std(spectral_centroids)],
            [np.mean(spectral_rolloff)],
            [np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth)],
            [np.std(spectral_bandwidth)],
            [np.mean(zero_crossing_rate)],
            [np.std(zero_crossing_rate)]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/api/classify', methods=['POST'])
def classify_audio():
    global model, label_encoder, scaler
    
    try:
        if model is None or label_encoder is None or scaler is None:
            return jsonify({'error': 'Model not loaded. Please ensure the model is trained and available.'}), 500
        
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
            
            # Normalize features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Get genre names
            genre_names = label_encoder.classes_
            
            # Format results
            results = []
            for i, genre in enumerate(genre_names):
                confidence = float(probabilities[i] * 100)
                if confidence > 1:  # Only include genres with >1% confidence
                    results.append({
                        'genre': genre.title(),
                        'confidence': round(confidence, 1),
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
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device)
    })

if __name__ == '__main__':
    print("Starting AudioSort Backend with PyTorch...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    
    # Load model on startup
    model_loaded = load_model()
    if not model_loaded:
        print("Warning: Running without trained model. Train the model first!")
    
    app.run(debug=True, port=5000)