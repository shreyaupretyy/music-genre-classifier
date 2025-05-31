import numpy as np
import pandas as pd
import librosa
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

class AudioDataset(Dataset):
    """Custom dataset for audio features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

def extract_features(file_path):
    """Extract comprehensive audio features using librosa with proper error handling"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        
        # Check if audio is loaded properly
        if len(y) == 0:
            print(f"Warning: Empty audio file {file_path}")
            return None
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Extract Chroma features - fix the import issue
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        except AttributeError:
            # Fallback for older librosa versions
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
            # Get harmonic component
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        except:
            # Fallback: use original audio if harmonic separation fails
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_scaled = np.mean(tonnetz.T, axis=0)
        
        # Extract rhythm features
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if np.isnan(tempo) or tempo == 0:
                tempo = 120.0  # Default tempo if detection fails
        except:
            tempo = 120.0
        
        # Extract spectral features
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        except Exception as e:
            print(f"Warning: Spectral feature extraction failed for {file_path}: {e}")
            # Use default values
            spectral_centroids = np.array([1000.0])
            spectral_rolloff = np.array([2000.0])
            spectral_bandwidth = np.array([1000.0])
            zero_crossing_rate = np.array([0.1])
        
        # Combine all features
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
            print(f"Warning: NaN values found in features for {file_path}")
            return None
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation"):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, accuracy

def train_model():
    """Train the genre classification model using PyTorch"""
    data_path = "data/genres"
    
    if not os.path.exists(data_path):
        print(f"Data directory '{data_path}' not found!")
        print("Please create the directory structure and add audio files.")
        return None, None, None
    
    features = []
    labels = []
    
    print("Extracting features from audio files...")
    
    # Process each genre directory
    for genre in os.listdir(data_path):
        genre_path = os.path.join(data_path, genre)
        if os.path.isdir(genre_path):
            print(f"Processing {genre}...")
            file_count = 0
            
            audio_files = [f for f in os.listdir(genre_path) 
                          if f.endswith(('.mp3', '.wav', '.flac', '.m4a'))]
            
            for file in tqdm(audio_files, desc=f"  {genre}"):
                file_path = os.path.join(genre_path, file)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(genre)
                    file_count += 1
            
            print(f"  Processed {file_count} files from {genre}")
    
    if len(features) == 0:
        print("No audio files found! Please add audio files to the data/genres directories.")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Total files processed: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Genres found: {np.unique(y)}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    input_size = X.shape[1]
    num_classes = len(le.classes_)
    model = GenreClassifier(input_size, num_classes).to(device)
    
    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 150
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'scaler': scaler,
                'label_encoder': le
            }, 'models/best_model.pth')
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': le,
        'input_size': input_size,
        'num_classes': num_classes
    }, 'models/genre_classifier.pth')
    
    # Save scaler and label encoder separately for compatibility
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print(f"\nModel training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: models/genre_classifier.pth")
    print(f"Best model saved to: models/best_model.pth")
    print(f"Classes: {le.classes_}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    return model, le, scaler

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training history plot saved to: models/training_history.png")

if __name__ == "__main__":
    train_model()