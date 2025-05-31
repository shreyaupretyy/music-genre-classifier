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
CORS(app)  # Enable CORS for React frontend

# Global variables
model = None
label_encoder = None
scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenreClassifier(nn.Module):
    """PyTorch neural network for genre classification - must match training"""
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
            print("Loading PyTorch model...")
            checkpoint = torch.load('models/genre_classifier.pth', map_location=device)
            
            # Create model with same architecture as training
            input_size = checkpoint['input_size']
            num_classes = checkpoint['num_classes']
            model = GenreClassifier(input_size, num_classes).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            # Load scaler and label encoder
            scaler = checkpoint['scaler']
            label_encoder = checkpoint['label_encoder']
            
            print(f"✅ Model loaded successfully!")
            print(f"Input size: {input_size}")
            print(f"Number of classes: {num_classes}")
            print(f"Classes: {label_encoder.classes_}")
            print(f"Device: {device}")
            return True
            
        else:
            print("❌ Model file not found: models/genre_classifier.pth")
            print("Please train the model first by running: python train_model.py")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def extract_features(audio_path):
    """Extract audio features - must match training exactly"""
    y = None
    try:
        # Load audio file (same parameters as training)
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        if len(y) == 0:
            print(f"Warning: Empty audio file {audio_path}")
            return None
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Extract Chroma features
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        except AttributeError:
            chroma = librosa.feature.chromagram(y=y, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        # Extract Mel-spectrogram features
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_scaled = np.mean(mel.T, axis=0)
        
        # Extract Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_scaled = np.mean(contrast.T, axis=0)
        
        # Extract Tonnetz features
        try:
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        except:
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_scaled = np.mean(tonnetz.T, axis=0)
        
        # Extract rhythm features
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if np.isnan(tempo) or tempo == 0:
                tempo = 120.0
        except:
            tempo = 120.0
        
        # Extract spectral features
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        except Exception as e:
            print(f"Warning: Spectral feature extraction failed: {e}")
            spectral_centroids = np.array([1000.0])
            spectral_rolloff = np.array([2000.0])
            spectral_bandwidth = np.array([1000.0])
            zero_crossing_rate = np.array([0.1])
        
        # Combine all features (must match training order exactly)
        features = np.hstack([
            mfcc_scaled,
            chroma_scaled,
            mel_scaled,
            contrast_scaled,
            tonnetz_scaled,
            [tempo],
            [np.mean(spectral_centroids)],
            [np.std(spectral_centroids) if len(spectral_centroids) > 1 else 0],
            [np.mean(spectral_rolloff)],
            [np.std(spectral_rolloff) if len(spectral_rolloff) > 1 else 0],
            [np.mean(spectral_bandwidth)],
            [np.std(spectral_bandwidth) if len(spectral_bandwidth) > 1 else 0],
            [np.mean(zero_crossing_rate)],
            [np.std(zero_crossing_rate) if len(zero_crossing_rate) > 1 else 0]
        ])
        
        # Check for NaN values
        if np.any(np.isnan(features)):
            print(f"Warning: NaN values found in features")
            return None
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
    finally:
        # Clear variables to help with memory cleanup
        if y is not None:
            del y
        import gc
        gc.collect()

@app.route('/api/classify', methods=['POST'])
def classify_audio():
    global model, label_encoder, scaler
    
    try:
        # Check if model is loaded
        if model is None or label_encoder is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure the model is trained and available.',
                'details': 'Run: python train_model.py to train the model first.'
            }), 500
        
        # Check if file is provided
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported file format: {file_ext}',
                'supported': list(allowed_extensions)
            }), 400
        
        # Save uploaded file temporarily with better file handling
        filename = secure_filename(file.filename)
        temp_file_path = None
        
        try:
            # Create temporary file with a specific name pattern
            temp_fd, temp_file_path = tempfile.mkstemp(suffix=file_ext, prefix='audio_classify_')
            
            # Close the file descriptor to avoid file locking issues
            os.close(temp_fd)
            
            # Save the uploaded file
            file.save(temp_file_path)
            
            # Extract features
            print(f"Extracting features from {filename}...")
            features = extract_features(temp_file_path)
            
            if features is None:
                return jsonify({
                    'error': 'Could not extract features from audio file',
                    'details': 'The audio file may be corrupted or in an unsupported format.'
                }), 400
            
            # Normalize features using the same scaler from training
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Convert to tensor and move to device
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            
            # Make prediction
            print("Making prediction...")
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Get genre names and create results
            genre_names = label_encoder.classes_
            
            # Create results list
            results = []
            for i, genre in enumerate(genre_names):
                confidence = float(probabilities[i] * 100)
                if confidence > 0.5:  # Include genres with >0.5% confidence
                    results.append({
                        'genre': genre.title(),
                        'confidence': round(confidence, 1),
                        'description': f'{genre.title()} characteristics detected in audio analysis'
                    })
            
            # Sort by confidence (highest first)
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            print(f"Prediction completed. Top genre: {results[0]['genre'] if results else 'Unknown'}")
            
            # Return top 3 predictions
            return jsonify({
                'results': results[:3],
                'total_genres_detected': len(results),
                'model_info': {
                    'classes': genre_names.tolist(),
                    'device': str(device)
                }
            })
            
        except Exception as processing_error:
            print(f"Error during processing: {processing_error}")
            return jsonify({
                'error': f'Error processing audio file: {str(processing_error)}',
                'details': 'Please try again or check if the file is valid.'
            }), 500
            
        finally:
            # Clean up temp file with better error handling
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    # Force garbage collection to release any file handles
                    import gc
                    gc.collect()
                    
                    # Try to delete the file multiple times if needed
                    max_attempts = 5
                    for attempt in range(max_attempts):
                        try:
                            os.unlink(temp_file_path)
                            print(f"Temporary file deleted successfully")
                            break
                        except PermissionError as e:
                            if attempt < max_attempts - 1:
                                print(f"Attempt {attempt + 1}: Could not delete temp file, retrying...")
                                import time
                                time.sleep(0.1)  # Wait 100ms before retry
                            else:
                                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")
                                # Log the file for manual cleanup later
                                with open('temp_files_to_cleanup.log', 'a') as log_file:
                                    log_file.write(f"{temp_file_path}\n")
                        except Exception as cleanup_error:
                            print(f"Warning: Error during cleanup: {cleanup_error}")
                            break
                except Exception as final_cleanup_error:
                    print(f"Final cleanup error: {final_cleanup_error}")
            
    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({
            'error': f'Server error during classification: {str(e)}',
            'details': 'Please try again or contact support if the issue persists.'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'label_encoder_loaded': label_encoder is not None,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
        'classes': label_encoder.classes_.tolist() if label_encoder is not None else None
    })

@app.route('/api/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'PyTorch Neural Network',
        'classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_),
        'device': str(device),
        'parameters': sum(p.numel() for p in model.parameters())
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🎵 AudioSort Backend Starting...")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    
    # Load model on startup
    model_loaded = load_model()
    
    if model_loaded:
        print("✅ Ready to classify audio!")
    else:
        print("⚠️  Running without trained model.")
        print("   Train the model first: python train_model.py")
    
    print("=" * 50)
    print("Starting Flask server on http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, port=5000, host='0.0.0.0')