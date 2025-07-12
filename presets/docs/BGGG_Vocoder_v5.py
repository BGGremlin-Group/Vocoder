#!/usr/bin/env python3
# BGGG Vocoder Version 5.0: Advanced professional-grade vocoder application
# Features (New in V5.0):
# - Few-shot voice cloning with Coqui-TTS (HifiGAN V2) and SpeechBrain ECAPA-VoxCeleb embeddings
# - Prosody extraction and transfer for expressive voice synthesis
# - Real-time voice cloning during live audio streaming
# - Aggressive noise filtering with WebRTC VAD
# - CPU- and GPU-optimized visualizations (Spectrogram, Mel Spectrogram, 3D Waterfall, Frequency Bar, 3D Particle Swarm)
# - Interactive visualization controls (zoom, pan, color themes)
# - Audio-reactive facial animation with dlib landmark detection and webcam/live selfie support
# - Voice style transfer between audio samples
# - Live latency monitoring and dynamic audio chunk size adjustment
# - Batch processing of multiple audio files with phoneme-aligned TTS
# - Full-featured PyQt5 GUI with splash screens, theming, and advanced settings
# - Integrated Web UI for remote configuration and control
# - Embeds metadata and visuals in exported audio/video recordings
# - Advanced training controls (epochs, learning rate, batch size, speaker ID management)
# Developer: BGGremlin Group
# Date: July 12, 2025
# License: MIT

import sys
import os
import time
import json
import queue
import logging
import subprocess
import tempfile
import shutil
import requests
import bz2
import pkg_resources
from pathlib import Path
from threading import Thread, Lock
import numpy as np
from scipy import signal
from scipy.signal import stft
from scipy.fft import fft
import librosa
import pyaudio
import torch
import cv2
import dlib
from pydub import AudioSegment
import moviepy.editor as mpy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm
from matplotlib.colors import to_rgb
import vispy.scene
from vispy import visuals, scene
from vispy.color import Color
from numba import jit
import soundfile as sf
import psutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QPushButton, QSlider, QComboBox, QLabel, QTextEdit, QProgressBar,
                             QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget, QDialog,
                             QLineEdit, QMessageBox, QColorDialog)
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.Qt import QDesktopServices
import trimesh
from g2p_en import G2p
from TTS.api import TTS
from speechbrain.pretrained import EncoderClassifier
import webrtcvad

# Constants
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
N_FFT = 2048
HOP_SIZE = 512
MAX_BUFFER_SIZE = SAMPLE_RATE * 60
LATENCY_THRESHOLD = 0.1

# Paths
PROJECT_DIR = Path.home() / "BGGG_Vocoder"
ASSETS_DIR = PROJECT_DIR / "assets"
CONFIG_DIR = PROJECT_DIR / "config"
RECORDINGS_DIR = PROJECT_DIR / "recordings"
VOICE_MODELS_DIR = PROJECT_DIR / "voice_models"
PRESETS_DIR = PROJECT_DIR / "presets"
THEMES_DIR = PROJECT_DIR / "themes"
DOCS_DIR = PROJECT_DIR / "docs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_DIR / "vocoder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "algorithm": "phase",
    "visualization_mode": "Spectrogram",
    "pitch_shift": 0,
    "modulation_index": 0,
    "modulation_rate": 1.0,
    "volume": 1.0,
    "vis_color": [0, 1, 0],
    "animate_face": False,
    "animation_mode": "Face Only",
    "use_cloned_voice": False,
    "use_webcam": False,
    "selfie_path": "",
    "vis_update_rate": 30,
    "particle_count": 1000,
    "particle_size": 2.0,
    "training_epochs": 10,
    "prosody_strength": 0.1,
    "theme": "dark",
    "speaker_id": "default",
    "text_input": "",
    "learning_rate": 1e-4,
    "batch_size": 16,
    "latency_threshold": 0.1,
    "max_buffer_size": SAMPLE_RATE * 10,
    "low_performance_mode": False
}

def check_disk_space(path, min_space_mb=500):
    try:
        stat = shutil.disk_usage(path)
        if stat.free / (1024 * 1024) < min_space_mb:
            raise OSError(f"Insufficient disk space at {path}: {stat.free / (1024 * 1024):.2f} MB available, {min_space_mb} MB required")
        test_file = path / f"test_{os.urandom(8).hex()}.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        logger.error(f"Disk check failed: {e}")
        raise

def setup_project_structure():
    try:
        for directory in [PROJECT_DIR, ASSETS_DIR, CONFIG_DIR, RECORDINGS_DIR, VOICE_MODELS_DIR, PRESETS_DIR, THEMES_DIR, DOCS_DIR]:
            directory.mkdir(exist_ok=True)
        if not (CONFIG_DIR / "config.json").exists():
            with open(CONFIG_DIR / "config.json", 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
        manual_path = DOCS_DIR / "user_manual.html"
        if not manual_path.exists():
            with open(manual_path, 'w') as f:
                f.write("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>BGGG Vocoder V5.0 User Manual</title>
                        <style>
                            body { font-family: Arial, sans-serif; margin: 40px; }
                            h1 { color: #333; }
                            h2 { color: #555; }
                            pre { background: #f4f4f4; padding: 10px; }
                            p { line-height: 1.6; }
                        </style>
                    </head>
                    <body>
                        <h1>BGGG Vocoder V5.0 User Manual</h1>
                        <h2>Introduction</h2>
                        <p>BGGG Vocoder is a versatile audio processing application supporting vocoding, voice cloning, and visualizations with a retro Winamp-style UI.</p>
                        <h2>Features</h2>
                        <ul>
                            <li>Real-time audio processing with phase, LPC, and channel vocoders.</li>
                            <li>Voice cloning using coqui-ai/TTS and speechbrain.</li>
                            <li>Multiple visualization modes: Spectrogram, 3D Waterfall, Frequency Bar, etc.</li>
                            <li>Facial, full-body (with OBJ support), and particle animations.</li>
                            <li>Batch processing and preset management.</li>
                            <li>Real-time text-to-speech with live input.</li>
                            <li>Optimized phoneme detection for noisy environments.</li>
                        </ul>
                        <h2>Installation</h2>
                        <p>Install dependencies using: <pre>pip install -r requirements.txt</pre></p>
                        <h2>Usage</h2>
                        <p>Select an algorithm, adjust settings, and start processing. Use the GitHub repository for support: <a href="https://github.com/username/BGGG_Vocoder">BGGG_Vocoder</a></p>
                        <h2>Controls</h2>
                        <p>Adjust sliders for pitch, modulation, and volume. Use tabs for advanced settings or enable simplified mode for retro UI.</p>
                    </body>
                    </html>
                """)
        web_ui_path = THEMES_DIR / "web_ui.html"
        if not web_ui_path.exists():
            with open(web_ui_path, 'w') as f:
                f.write("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>BGGG Vocoder Web UI</title>
                        <style>
                            body { font-family: Arial, sans-serif; background-color: #1A1A1A; color: #00FFFF; }
                            button { background-color: #444444; color: #00FFFF; border: 1px solid #00FFFF; padding: 5px; }
                            button:hover { background-color: #555555; }
                            input, select { background-color: #222222; color: #00FFFF; border: 1px solid #00FFFF; }
                        </style>
                        <script>
                            async function updateConfig() {
                                const config = {
                                    algorithm: document.getElementById('algorithm').value,
                                    visualization_mode: document.getElementById('vis_mode').value,
                                    volume: parseFloat(document.getElementById('volume').value),
                                    modulation_rate: parseFloat(document.getElementById('modulation_rate').value)
                                };
                                const response = await fetch('/update_config', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify(config)
                                });
                                const result = await response.json();
                                alert(result.message);
                            }
                            async function startProcessing() {
                                const response = await fetch('/start_processing', { method: 'POST' });
                                const result = await response.json();
                                alert(result.message);
                            }
                            async function stopProcessing() {
                                const response = await fetch('/stop_processing', { method: 'POST' });
                                const result = await response.json();
                                alert(result.message);
                            }
                        </script>
                    </head>
                    <body>
                        <h1>BGGG Vocoder Web UI</h1>
                        <div>
                            <label>Algorithm:</label>
                            <select id="algorithm">
                                <option value="phase">Phase</option>
                                <option value="lpc">LPC</option>
                                <option value="channel">Channel</option>
                            </select>
                        </div>
                        <div>
                            <label>Visualization Mode:</label>
                            <select id="vis_mode">
                                <option value="Spectrogram">Spectrogram</option>
                                <option value="3D Waterfall">3D Waterfall</option>
                                <option value="Frequency Bar">Frequency Bar</option>
                                <option value="Mel Spectrogram">Mel Spectrogram</option>
                                <option value="Waveform Envelope">Waveform Envelope</option>
                                <option value="Chroma Features">Chroma Features</option>
                                <option value="3D Particle Swarm">3D Particle Swarm</option>
                            </select>
                        </div>
                        <div>
                            <label>Volume:</label>
                            <input type="number" id="volume" min="0" max="2" step="0.1" value="{{ config.volume }}">
                        </div>
                        <div>
                            <label>Modulation Rate (Hz):</label>
                            <input type="number" id="modulation_rate" min="0.1" max="10" step="0.1" value="{{ config.modulation_rate }}">
                        </div>
                        <button onclick="updateConfig()">Update Config</button>
                        <button onclick="startProcessing()">Start Processing</button>
                        <button onclick="stopProcessing()">Stop Processing</button>
                    </body>
                    </html>
                """)
        for img in ["logo_screen.png", "credits_screen.png", "exit_screen.png", "static_visualizer.png"]:
            img_path = ASSETS_DIR / img
            if not img_path.exists():
                with open(img_path, 'w') as f:
                    f.write("")  # Placeholder for images
                logger.warning(f"Placeholder created for {img}. Replace with actual image.")
        logger.info("Project structure set up")
    except Exception as e:
        logger.error(f"Error setting up project structure: {e}")
        QMessageBox.critical(None, "Error", f"Failed to set up project structure: {e}")
        sys.exit(1)

def install_dependencies():
    dependencies = [
        ("pyaudio", "0.2.14"),
        ("numpy", "1.26.4"),
        ("scipy", "1.14.1"),
        ("pydub", "0.25.1"),
        ("PyQt5", "5.15.11"),
        ("requests", "2.32.3"),
        ("sounddevice", "0.5.0"),
        ("matplotlib", "3.9.2"),
        ("librosa", "0.10.2"),
        ("opencv-python", "4.10.0"),
        ("dlib", "19.24.6"),
        ("moviepy", "1.0.3"),
        ("torch", "2.4.1+cpu"),
        ("torchaudio", "2.4.1+cpu"),
        ("TTS", "0.22.0"),
        ("speechbrain", "1.0.1"),
        ("g2p_en", "2.1.0"),
        ("vispy", "0.14.3"),
        ("numba", "0.60.0"),
        ("soundfile", "0.12.1"),
        ("psutil", "6.0.0"),
        ("flask", "3.0.3"),
        ("trimesh", "4.2.3"),
        ("webrtcvad", "2.0.10")
    ]
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    for dep, version in dependencies:
        if dep not in installed or installed[dep] != version:
            logger.info(f"Installing {dep}=={version}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{dep}=={version}", "--no-cache-dir"], timeout=300)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.error(f"Failed to install {dep}: {e}")
                QMessageBox.critical(None, "Dependency Error",
                                     f"Failed to install {dep}=={version}. Please install manually using:\n"
                                     f"pip install {dep}=={version}\nThen restart the application.")
                sys.exit(1)
    predictor_path = ASSETS_DIR / "shape_predictor_68_face_landmarks.dat"
    fallback_path = ASSETS_DIR / "shape_predictor_68_face_landmarks.dat.fallback"
    if not predictor_path.exists():
        try:
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            logger.info("Downloading dlib shape predictor...")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            bz2_path = ASSETS_DIR / "shape_predictor_68_face_landmarks.dat.bz2"
            with open(bz2_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with bz2.BZ2File(bz2_path, 'rb') as f_in, open(predictor_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(bz2_path)
            logger.info("Downloaded dlib shape predictor")
        except Exception as e:
            logger.error(f"Failed to download dlib shape predictor: {e}")
            if fallback_path.exists():
                logger.info("Using fallback dlib shape predictor")
                shutil.copy(fallback_path, predictor_path)
            else:
                QMessageBox.critical(None, "Error",
                                     f"Failed to download dlib shape predictor: {e}\n"
                                     "No fallback available. Please download it manually from dlib.net and place it in assets directory.")
                sys.exit(1)

def handle_errors(method):
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {method.__name__}: {e}")
            if hasattr(args[0], 'update_status'):
                args[0].update_status(f"Error in {method.__name__}: {e}", error=True)
            raise
    return wrapper

@jit(nopython=True)
def amplitude_to_db(magnitude, ref=1.0):
    return 20 * np.log10(np.maximum(magnitude, 1e-10) / ref)

class VocoderProcessor:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tts = TTS(model_name="tts_models/en/ljspeech/hifigan_v2", progress_bar=False).to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            raise RuntimeError(f"Failed to initialize TTS model: {e}")
        self.vocoder = self.tts.vocoder
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ASSETS_DIR / "speechbrain")
        )
        try:
            self.g2p = G2p()
        except Exception as e:
            logger.error(f"Failed to initialize G2p: {e}")
            raise RuntimeError(f"Failed to initialize G2p: {e}")
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(ASSETS_DIR / "shape_predictor_68_face_landmarks.dat"))
        self.vad = webrtcvad.Vad(3)  # Aggressive mode for noise filtering
        self.stream = None
        self.recording = False
        self.audio_data = []
        self.spectral_data = []
        self.phoneme_data = []
        self.animation_frames = []
        self.selfie_image = None
        self.selfie_path = ""
        self.webcam = None
        self.webcam_index = 0
        self.use_webcam = False
        self.queue = queue.Queue(maxsize=100)
        self.spectral_queue = queue.Queue(maxsize=100)
        self.phoneme_queue = queue.Queue(maxsize=100)
        self.animation_queue = queue.Queue(maxsize=100)
        self.latency_queue = queue.Queue(maxsize=100)
        self.data_lock = Lock()
        self.config_lock = Lock()
        self.vis_canvas = None
        self.vis_view = None
        self.vis_particles = None
        self.obj_model = None
        self.running = True
        self.visualization_running = True
        self.algorithm = DEFAULT_CONFIG["algorithm"]
        self.visualization_mode = DEFAULT_CONFIG["visualization_mode"]
        self.pitch_shift = DEFAULT_CONFIG["pitch_shift"]
        self.modulation_index = DEFAULT_CONFIG["modulation_index"]
        self.modulation_rate = DEFAULT_CONFIG["modulation_rate"]
        self.volume = DEFAULT_CONFIG["volume"]
        self.vis_color = np.array(DEFAULT_CONFIG["vis_color"])
        self.animate_face = DEFAULT_CONFIG["animate_face"]
        self.animation_mode = DEFAULT_CONFIG["animation_mode"]
        self.use_cloned_voice = DEFAULT_CONFIG["use_cloned_voice"]
        self.use_webcam = DEFAULT_CONFIG["use_webcam"]
        self.vis_update_rate = DEFAULT_CONFIG["vis_update_rate"]
        self.particle_count = DEFAULT_CONFIG["particle_count"]
        self.particle_size = DEFAULT_CONFIG["particle_size"]
        self.training_epochs = DEFAULT_CONFIG["training_epochs"]
        self.prosody_strength = DEFAULT_CONFIG["prosody_strength"]
        self.speaker_id = DEFAULT_CONFIG["speaker_id"]
        self.text_input = DEFAULT_CONFIG["text_input"]
        self.learning_rate = DEFAULT_CONFIG["learning_rate"]
        self.batch_size = DEFAULT_CONFIG["batch_size"]
        self.latency_threshold = DEFAULT_CONFIG["latency_threshold"]
        self.max_buffer_size = DEFAULT_CONFIG["max_buffer_size"]
        self.low_performance_mode = DEFAULT_CONFIG["low_performance_mode"]
        self.input_device_index = None
        self.output_device_index = None
        self.voice_model_path = None
        self.chunk_size = CHUNK_SIZE

    def adjust_chunk_size(self, system_load):
        try:
            if system_load > 0.8:
                self.chunk_size = max(512, self.chunk_size // 2)
            elif system_load < 0.3:
                self.chunk_size = min(2048, self.chunk_size * 2)
            logger.info(f"Adjusted chunk size to {self.chunk_size}")
            self.queue = queue.Queue(maxsize=100)
            self.spectral_queue = queue.Queue(maxsize=100)
            self.animation_queue = queue.Queue(maxsize=100)
            self.phoneme_queue = queue.Queue(maxsize=100)
        except Exception as e:
            logger.error(f"Error adjusting chunk size: {e}")

    @handle_errors
    def start_stream(self):
        cpu_load = psutil.cpu_percent(interval=1) / 100
        self.adjust_chunk_size(cpu_load)
        stream_kwargs = {
            'format': pyaudio.paFloat32,
            'channels': 1,
            'rate': SAMPLE_RATE,
            'input': True,
            'output': True,
            'frames_per_buffer': self.chunk_size,
            'input_device_index': self.input_device_index,
            'output_device_index': self.output_device_index,
            'stream_callback': self.callback
        }
        if sys.platform == "win32":
            stream_kwargs['as_loopback'] = True
        self.stream = self.pa.open(**stream_kwargs)
        self.stream.start_stream()
        logger.info(f"Audio stream started with chunk size {self.chunk_size}")

    @handle_errors
    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.release_webcam()
        logger.info("Audio stream stopped")

    @handle_errors
    def init_webcam(self):
        self.webcam = cv2.VideoCapture(self.webcam_index)
        if not self.webcam.isOpened():
            raise ValueError(f"Failed to open webcam at index {self.webcam_index}")
        logger.info(f"Webcam initialized at index {self.webcam_index}")

    def release_webcam(self):
        if self.webcam:
            self.webcam.release()
            self.webcam = None
            logger.info("Webcam released")

    @handle_errors
    def load_selfie(self, filename):
        self.selfie_image = cv2.imread(filename)
        if self.selfie_image is None:
            raise ValueError(f"Failed to load selfie image: {filename}")
        self.selfie_image = cv2.cvtColor(self.selfie_image, cv2.COLOR_BGR2RGB)
        self.selfie_path = filename
        logger.info(f"Selfie loaded: {filename}")

    @jit(nopython=True)
    def normalize_audio(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        return audio

    @handle_errors
    def load_audio(self, filename):
        audio_segment = AudioSegment.from_file(filename)
        audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        return audio

    @handle_errors
    def apply_vocoder(self, audio):
        if self.algorithm == "phase":
            _, _, z = stft(audio, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP_SIZE)
            magnitude, phase = np.abs(z), np.angle(z)
            modified_phase = phase + np.random.uniform(-self.modulation_index, self.modulation_index, phase.shape)
            modified_z = magnitude * np.exp(1j * modified_phase)
            _, processed = signal.istft(modified_z, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP_SIZE)
            return processed[:len(audio)]
        elif self.algorithm == "lpc":
            coeffs = librosa.lpc(audio, order=16)
            residual = signal.lfilter(coeffs, [1], audio)
            processed = signal.lfilter([1], coeffs, residual)
            return processed[:len(audio)]
        elif self.algorithm == "channel":
            freqs, _, z = stft(audio, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP_SIZE)
            magnitude = np.abs(z)
            for i in range(magnitude.shape[0]):
                magnitude[i, :] *= np.random.uniform(0.5, 1.5, magnitude.shape[1])
            processed_z = magnitude * np.exp(1j * np.angle(z))
            _, processed = signal.istft(processed_z, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP_SIZE)
            return processed[:len(audio)]
        return audio

    @jit(nopython=True)
    def apply_pitch_shift(self, audio):
        if self.pitch_shift == 0:
            return audio
        n_steps = self.pitch_shift
        return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)

    @jit(nopython=True)
    def apply_modulation(self, audio):
        if self.modulation_index == 0:
            return audio
        t = np.arange(len(audio)) / SAMPLE_RATE
        carrier = np.sin(2 * np.pi * self.modulation_rate * t)
        return audio * (1 + self.modulation_index * carrier)

    @handle_errors
    def extract_prosody(self, audio):
        pitch = librosa.yin(audio, fmin=50, fmax=500, sr=SAMPLE_RATE)
        energy = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_SIZE)[0]
        return torch.tensor([pitch, energy], dtype=torch.float32).to(self.device)

    @handle_errors
    def train_voice(self, audio, model_path, speaker_id="default", lr=1e-4, batch_size=16):
        if len(audio) < SAMPLE_RATE * 10:
            raise ValueError("Audio sample too short for voice training. Record at least 10 seconds.")
        audio = torch.tensor(audio, dtype=torch.float32).to(self.device)
        embedding = self.speaker_encoder.encode_batch(audio.unsqueeze(0))[0]
        self.tts.model.train()
        optimizer = torch.optim.Adam(self.tts.model.parameters(), lr=lr)
        mel_spec = self.tts.mel_spectrogram(audio).unsqueeze(0)
        dataset = torch.utils.data.TensorDataset(mel_spec, mel_spec)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(self.training_epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                mel_input = batch[0].to(self.device)
                output = self.tts.model(mel_input, speaker_embedding=embedding)
                loss = torch.nn.MSELoss()(output['mel'], mel_input)
                prosody_loss = torch.nn.MSELoss()(output['prosody'], self.extract_prosody(audio))
                batch_loss = loss + self.prosody_strength * prosody_loss
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tts.model.parameters(), 1.0)
                optimizer.step()
                total_loss += batch_loss.item()
            logger.info(f"Voice training epoch {epoch + 1}/{self.training_epochs}, loss: {total_loss / len(dataloader):.4f}")
        torch.save({
            'model_state': self.tts.model.state_dict(),
            'speaker_embedding': embedding,
            'speaker_id': speaker_id,
            'lr': lr,
            'batch_size': batch_size
        }, model_path)
        self.voice_model_path = model_path
        logger.info(f"Voice model trained for speaker {speaker_id} and saved to {model_path}")

    @handle_errors
    def text_to_speech(self, text, speaker_id="default"):
        if not text.strip():
            raise ValueError("Text input is empty")
        if not self.voice_model_path or not os.path.exists(self.voice_model_path):
            raise FileNotFoundError("No trained voice model available")
        checkpoint = torch.load(self.voice_model_path, map_location=self.device)
        if 'model_state' not in checkpoint or 'speaker_embedding' not in checkpoint:
            raise ValueError("Invalid voice model file: missing required keys")
        try:
            self.tts.model.load_state_dict(checkpoint['model_state'])
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load voice model: {e}")
        self.tts.model.eval()
        phonemes = self.g2p(text)
        audio = self.tts.tts(text=text, speaker_embedding=checkpoint['speaker_embedding'])
        audio = np.array(audio, dtype=np.float32)
        if not np.all(np.isfinite(audio)):
            raise ValueError("Generated audio contains invalid values")
        phoneme_timings = [(i / len(phonemes)) * (len(audio) / SAMPLE_RATE) for i in range(len(phonemes))]
        audio = self.normalize_audio(audio)
        return audio, phonemes, phoneme_timings

    @handle_errors
    def apply_cloned_voice(self, audio, text, speaker_id):
        processed, phonemes, timings = self.text_to_speech(text, speaker_id)
        return processed, phonemes, timings

    @handle_errors
    def apply_style_transfer(self, audio, style_audio):
        if len(style_audio) < self.chunk_size:
            raise ValueError("Style audio too short")
        if len(audio) != len(style_audio):
            style_audio = librosa.util.fix_length(style_audio, len(audio))
        style_audio = librosa.resample(style_audio, orig_sr=SAMPLE_RATE, target_sr=SAMPLE_RATE, res_type='kaiser_best')
        style_embedding = self.speaker_encoder.encode_batch(torch.tensor(style_audio, dtype=torch.float32).to(self.device).unsqueeze(0))[0]
        content_mel = self.tts.mel_spectrogram(torch.tensor(audio, dtype=torch.float32).to(self.device)).unsqueeze(0)
        output = self.tts.model(content_mel, speaker_embedding=style_embedding)
        result = self.vocoder.inference(output['mel']).squeeze().cpu().numpy()
        if not np.all(np.isfinite(result)):
            raise ValueError("Style transfer produced invalid audio")
        return result

    @handle_errors
    def detect_phonemes(self, audio):
        if len(audio) < self.chunk_size:
            raise ValueError("Audio segment too short for phoneme detection")
        # Noise reduction with VAD
        audio_int16 = (audio * 32768).astype(np.int16)
        vad_frames = []
        frame_duration = self.chunk_size / SAMPLE_RATE * 1000  # ms
        for i in range(0, len(audio_int16), self.chunk_size):
            frame = audio_int16[i:i + self.chunk_size].tobytes()
            if len(frame) == self.chunk_size * 2:  # 16-bit samples
                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
                if is_speech:
                    vad_frames.append(audio[i:i + self.chunk_size])
        if not vad_frames:
            logger.warning("No speech detected after VAD")
            return [], []
        audio = np.concatenate(vad_frames)
        # Preemphasis for noise robustness
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        audio = self.normalize_audio(audio)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.speaker_encoder.classify_batch(audio_tensor.unsqueeze(0))
            phoneme_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            phonemes = []
            for id in phoneme_ids[0]:
                try:
                    phoneme = self.g2p.id_to_phoneme(id)
                    if phoneme:
                        phonemes.append(phoneme)
                except KeyError:
                    logger.warning(f"Invalid phoneme ID {id}, skipping")
            timings = [(i / len(phonemes)) * (len(audio) / SAMPLE_RATE) for i in range(len(phonemes))] if phonemes else []
        return phonemes, timings

    @handle_errors
    def save_recording(self, filename, export_video=False, metadata=None):
        check_disk_space(Path(filename).parent)
        if not self.audio_data:
            raise ValueError("No audio data to save")
        audio_array = np.concatenate(self.audio_data)
        phonemes = [p for p, _ in self.phoneme_data]
        timings = [t for _, t in self.phoneme_data]
        if self.use_cloned_voice:
            audio_array, phonemes, timings = self.apply_cloned_voice(audio_array, self.text_input, self.speaker_id)
        audio_array = self.normalize_audio(audio_array)
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=4,
            channels=1
        )
        audio_filename = filename if not export_video else filename.replace(".wav", "_audio.mp4")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            audio_segment.export(temp_path, format="wav")
            if export_video and self.animation_frames and self.animate_face and self.animation_mode == "Face Only":
                clip = mpy.ImageSequenceClip(self.animation_frames, fps=SAMPLE_RATE // self.chunk_size)
                with mpy.AudioFileClip(temp_path) as audio_clip:
                    clip = clip.set_audio(audio_clip)
                    clip.write_videofile(filename, codec="libx264", audio_codec="aac")
            else:
                audio_segment.export(audio_filename, format="mp4" if export_video else "wav", tags=metadata or {})
        os.remove(temp_path)
        logger.info(f"Recording saved to {filename}")
        return audio_array, phonemes, timings

    @handle_errors
    def batch_process(self, input_files, output_dir):
        check_disk_space(Path(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for i, filename in enumerate(input_files):
            try:
                audio = self.load_audio(filename)
                processed, phonemes, timings = self.apply_cloned_voice(audio, self.text_input, self.speaker_id)
                processed = self.apply_vocoder(processed)
                processed = self.apply_pitch_shift(processed)
                processed = self.apply_modulation(processed)
                processed = self.normalize_audio(processed * self.volume)
                output_path = os.path.join(output_dir, f"processed_{os.path.basename(filename)}")
                sf.write(output_path, processed, SAMPLE_RATE)
                results.append((output_path, phonemes, timings))
                logger.info(f"Batch processed {filename} to {output_path}")
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                results.append((filename, [], [], str(e)))
        return results

    @handle_errors
    def init_vispy(self):
        if self.vis_canvas is None:
            self.vis_canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=False)
            self.vis_view = self.vis_canvas.central_widget.add_view()
            self.vis_view.camera = 'panzoom'
            if self.visualization_mode == "3D Particle Swarm":
                self.vis_particles = visuals.ParticleSystem(
                    size=self.particle_size,
                    color=tuple(self.vis_color) + (1.0,),
                    edge_color=None
                )
                self.vis_view.add(self.vis_particles)
            logger.info("Vispy initialized for CPU-based visualizations")

    @handle_errors
    def animate_face(self, audio, phonemes, timings, frame_size=(480, 640)):
        if self.use_webcam and self.webcam:
            ret, frame = self.webcam.read()
            if not ret:
                logger.warning("Failed to read webcam frame")
                return self.selfie_image if self.selfie_image is not None else np.zeros(frame_size + (3,), dtype=np.uint8)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.selfie_image is None:
            logger.warning("No selfie image loaded")
            return np.zeros(frame_size + (3,), dtype=np.uint8)
        else:
            img = self.selfie_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector(gray)
        if not faces:
            logger.warning("No faces detected in image")
            return img
        face = faces[0]
        landmarks = self.predictor(gray, face)
        lip_points = [landmarks.part(i).x for i in range(48, 60)], [landmarks.part(i).y for i in range(48, 60)]
        current_time = len(audio) / SAMPLE_RATE
        phoneme_idx = min(len(timings) - 1, max(0, next((i for i, t in enumerate(timings) if t > current_time), 0)))
        phoneme = phonemes[phoneme_idx] if phonemes else ""
        lip_y = np.array(lip_points[1], dtype=np.int32)
        if phoneme in ['AA', 'AE', 'AH', 'AO']:
            lip_y[6:12] += 10
        elif phoneme in ['IY', 'IH', 'EH']:
            lip_y[6:12] += 5
        elif phoneme in ['M', 'B', 'P']:
            lip_y[6:12] -= 2
        points = np.array(list(zip(lip_points[0], lip_y)), np.int32)
        cv2.fillPoly(img, [points], (255, 192, 203))
        return img

    @handle_errors
    def animate_full_body(self, audio, phonemes, timings):
        rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_SIZE)[0]
        intensity = np.clip(np.mean(rms) * 10, 0, 1)
        pitch = librosa.yin(audio, fmin=50, fmax=500, sr=SAMPLE_RATE)
        pitch_intensity = np.clip(np.mean(pitch) / 500, 0, 1)
        
        if not self.vis_canvas:
            self.init_vispy()
        self.vis_canvas.show()
        self.vis_view.scene.clear()

        if not self.obj_model:
            obj_path, _ = QFileDialog.getOpenFileName(None, "Select OBJ Model", str(ASSETS_DIR), "OBJ Files (*.obj)")
            if obj_path:
                try:
                    self.obj_model = trimesh.load(obj_path)
                    logger.info(f"Loaded OBJ model from {obj_path}")
                except Exception as e:
                    logger.error(f"Failed to load OBJ model: {e}")
                    self.obj_model = trimesh.creation.box()
                    logger.warning("Using fallback box model")
            else:
                self.obj_model = trimesh.creation.box()
                logger.warning("No OBJ model selected, using fallback box")

        vertices = np.array(self.obj_model.vertices, dtype=np.float32)
        faces = np.array(self.obj_model.faces, dtype=np.uint32)
        mesh = visuals.Mesh(vertices=vertices, faces=faces, color=(0.5, 0.5, 0.5, 1.0))
        mesh.transform = scene.transforms.MatrixTransform()
        mesh.transform.rotate(intensity * 30, (0, 1, 0))
        mesh.transform.rotate(pitch_intensity * 15, (1, 0, 0))
        mesh.transform.scale((1 + intensity * 0.1, 1 + intensity * 0.1, 1 + intensity * 0.1))
        self.vis_view.add(mesh)
        self.vis_canvas.update()
        return None

    @handle_errors
    def animate_particles(self, audio):
        rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_SIZE)[0]
        intensity = np.clip(np.mean(rms) * 10, 0, 1)
        pitch = librosa.yin(audio, fmin=50, fmax=500, sr=SAMPLE_RATE)
        pitch_intensity = np.clip(np.mean(pitch) / 500, 0, 1)
        
        if not self.vis_canvas:
            self.init_vispy()
        self.vis_canvas.show()
        self.vis_view.scene.clear()
        
        particle_count = self.particle_count
        positions = np.random.uniform(-1, 1, (particle_count, 3)).astype(np.float32)
        velocities = np.random.uniform(-0.1, 0.1, (particle_count, 3)).astype(np.float32)
        positions += velocities + np.array([
            intensity * np.sin(pitch_intensity * positions[:, 0] + time.time()),
            intensity * np.cos(pitch_intensity * positions[:, 1] + time.time()),
            intensity * np.sin(pitch_intensity * positions[:, 2] + time.time())
        ]).T
        positions = np.clip(positions, -1, 1)
        
        self.vis_particles = visuals.ParticleSystem(
            size=self.particle_size,
            color=tuple(self.vis_color * (1 - pitch_intensity * 0.5)) + (1.0,)
        )
        self.vis_particles.set_data(pos=positions)
        self.vis_view.add(self.vis_particles)
        self.vis_canvas.update()
        return None

    def callback(self, in_data, frame_count, time_info, status):
        start_time = time.time()
        try:
            audio = np.frombuffer(in_data, dtype=np.float32)
            processed = self.apply_vocoder(audio)
            processed = self.apply_pitch_shift(processed)
            processed = self.apply_modulation(processed)
            phonemes, timings = [], []
            if self.use_cloned_voice:
                processed, phonemes, timings = self.apply_cloned_voice(processed, self.text_input, self.speaker_id)
            processed = self.normalize_audio(processed * self.volume)
            out_data = processed.astype(np.float32).tobytes()
            with self.data_lock:
                if self.recording:
                    total_samples = sum(len(chunk) for chunk in self.audio_data)
                    if total_samples + len(audio) < self.max_buffer_size:
                        self.audio_data.append(audio)
                        stft_matrix = stft(audio, n_fft=N_FFT, hop_length=HOP_SIZE, window='hann')
                        self.spectral_data.append(np.abs(stft_matrix))
                        phonemes, timings = self.detect_phonemes(audio)
                        self.phoneme_data.append((phonemes, timings))
                        if self.animate_face and (self.selfie_image or self.use_webcam):
                            if self.animation_mode == "Face Only":
                                frame = self.animate_face(processed, phonemes, timings)
                                if frame is not None:
                                    self.animation_frames.append(frame)
                            elif self.animation_mode == "Full Body":
                                self.animate_full_body(processed, phonemes, timings)
                            elif self.animation_mode == "Particles":
                                self.animate_particles(processed)
                try:
                    self.queue.put_nowait(processed)
                    self.spectral_queue.put_nowait(np.abs(stft_matrix))
                    self.phoneme_queue.put_nowait((phonemes, timings))
                    if self.animate_face and (self.selfie_image or self.use_webcam) and self.animation_mode == "Face Only":
                        frame = self.animate_face(processed, phonemes, timings)
                        if frame is not None:
                            self.animation_queue.put_nowait(frame)
                except queue.Full:
                    logger.warning("Queue full, dropping data")
            latency = time.time() - start_time
            self.latency_queue.put_nowait(latency)
            if latency > self.latency_threshold:
                logger.warning(f"High latency detected: {latency:.3f}s")
            return (out_data, pyaudio.paContinue)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            return (in_data, pyaudio.paContinue)

class BGGGVocoderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BGGG Vocoder V5.0")
        self.processor = VocoderProcessor()
        self.running = True
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_pan)
        self.zoom_level = 1.0
        self.pan_position = [0, 0]
        self.settings_history = []
        self.settings_index = -1
        self.current_screen = "logo"
        self.show_splash_screen()
        self.init_ui()
        self.start_visualization_thread()
        self.start_latency_monitor()
        self.save_settings_state()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_equalizer)
        self.timer.start(100)

    def show_splash_screen(self):
        self.splash_dialog = QDialog(self)
        self.splash_dialog.setWindowTitle("BGGG Vocoder V5.0")
        self.splash_dialog.setWindowFlags(Qt.FramelessWindowHint)
        layout = QVBoxLayout()
        self.image_label = QLabel()
        pixmap = QPixmap(str(ASSETS_DIR / f"{self.current_screen}_screen.png"))
        if pixmap.isNull():
            self.image_label.setText(f"{self.current_screen.capitalize()} Screen Missing")
            logger.warning(f"{self.current_screen}_screen.png not found in assets/")
        else:
            self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        layout.addWidget(self.image_label)
        self.splash_dialog.setLayout(layout)
        self.splash_dialog.resize(800, 600)
        self.splash_dialog.keyPressEvent = self.handle_splash_keypress
        self.splash_dialog.show()

    def handle_splash_keypress(self, event):
        try:
            if self.current_screen == "logo":
                self.current_screen = "credits"
                pixmap = QPixmap(str(ASSETS_DIR / "credits_screen.png"))
                if pixmap.isNull():
                    self.image_label.setText("Credits Screen Missing")
                    logger.warning("credits_screen.png not found in assets/")
                else:
                    self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
                self.splash_dialog.update()
            elif self.current_screen == "credits":
                self.splash_dialog.close()
                self.show()
        except Exception as e:
            logger.error(f"Error in splash screen keypress: {e}")

    def show_credits(self):
        self.current_screen = "credits"
        self.show_splash_screen()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        central_widget.setStyleSheet("""
            QWidget {
                background-color: #1A1A1A;
                background-image: url(assets/static_visualizer.png);
                background-repeat: no-repeat;
                background-position: center;
            }
        """)

        control_widget = QWidget()
        control_layout = QHBoxLayout()
        control_widget.setStyleSheet("""
            QWidget { background-color: #1A1A1A; border: 2px solid #333333; border-radius: 5px; }
            QPushButton { background-color: #444444; color: #00FFFF; border: 1px solid #00FFFF; border-radius: 5px; padding: 5px; }
            QPushButton:hover { background-color: #555555; }
        """)
        control_widget.setLayout(control_layout)

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.start_processing)
        control_layout.addWidget(self.play_button)

        self.stop_button = QPushButton("■ Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        control_layout.addWidget(self.stop_button)

        self.record_button = QPushButton("● Rec")
        self.record_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_button)

        self.save_button = QPushButton("💾 Save")
        self.save_button.clicked.connect(self.save_recording)
        control_layout.addWidget(self.save_button)

        main_layout.addWidget(control_widget)

        vis_widget = QWidget()
        vis_layout = QHBoxLayout()
        vis_widget.setStyleSheet("""
            QWidget { background-color: #1A1A1A; border: 2px solid #333333; border-radius: 5px; }
            QProgressBar { background-color: #222222; border: 1px solid #00FF00; height: 100px; }
            QProgressBar::chunk { background-color: #00FF00; width: 10px; }
        """)
        vis_widget.setLayout(vis_layout)

        self.equalizer_bars = []
        for _ in range(8):
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setOrientation(Qt.Vertical)
            vis_layout.addWidget(bar)
            self.equalizer_bars.append(bar)

        self.canvas = FigureCanvas(self.figure)
        vis_layout.addWidget(self.canvas)
        main_layout.addWidget(vis_widget)

        text_widget = QWidget()
        text_layout = QHBoxLayout()
        text_widget.setStyleSheet("""
            QWidget { background-color: #1A1A1A; border: 2px solid #333333; border-radius: 5px; }
            QLineEdit { background-color: #222222; color: #00FFFF; border: 1px solid #00FFFF; }
        """)
        text_widget.setLayout(text_layout)

        self.tts_input = QLineEdit()
        self.tts_input.setPlaceholderText("Enter text for live TTS...")
        self.tts_input.textChanged.connect(self.update_live_tts)
        text_layout.addWidget(self.tts_input)

        main_layout.addWidget(text_widget)

        status_widget = QWidget()
        status_layout = QHBoxLayout()
        status_widget.setStyleSheet("""
            QWidget { background-color: #1A1A1A; border: 2px solid #333333; border-radius: 5px; }
            QLabel { color: #00FFFF; font-family: 'Press Start 2P', monospace; }
        """)
        status_widget.setLayout(status_layout)

        self.status_label = QLabel("Status: Ready")
        status_layout.addWidget(self.status_label)

        self.latency_label = QLabel("Latency: 0 ms")
        status_layout.addWidget(self.latency_label)

        self.volume_label = QLabel("Volume: 0.0 dB")
        status_layout.addWidget(self.volume_label)

        main_layout.addWidget(status_widget)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget { background-color: #1A1A1A; }")
        main_layout.addWidget(self.tabs)

        main_widget = QWidget()
        main_layout_tab = QFormLayout()
        main_widget.setLayout(main_layout_tab)
        self.tabs.addTab(main_widget, "Main Settings")

        self.algorithm_label = QLabel("Vocoder Algorithm:")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Phase", "LPC", "Channel"])
        self.algorithm_combo.setCurrentText(self.processor.algorithm.capitalize())
        self.algorithm_combo.currentTextChanged.connect(self.update_algorithm)
        main_layout_tab.addRow(self.algorithm_label, self.algorithm_combo)

        self.vis_mode_label = QLabel("Visualization Mode:")
        self.vis_mode_combo = QComboBox()
        self.vis_mode_combo.addItems(["Spectrogram", "3D Waterfall", "Frequency Bar", "Mel Spectrogram", "Waveform Envelope", "Chroma Features", "3D Particle Swarm"])
        self.vis_mode_combo.setCurrentText(self.processor.visualization_mode)
        self.vis_mode_combo.currentTextChanged.connect(self.update_visualization_mode)
        main_layout_tab.addRow(self.vis_mode_label, self.vis_mode_combo)

        self.simplified_mode_check = QCheckBox("Simplified Mode")
        self.simplified_mode_check.setToolTip("Show only basic controls for retro UI")
        self.simplified_mode_check.stateChanged.connect(self.toggle_simplified_mode)
        main_layout_tab.addRow(self.simplified_mode_check)

        self.volume_control_label = QLabel("Volume:")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(200)
        self.volume_slider.setValue(int(self.processor.volume * 100))
        self.volume_slider.valueChanged.connect(self.update_volume)
        main_layout_tab.addRow(self.volume_control_label, self.volume_slider)

        self.audio_device_label = QLabel("Audio Input Device:")
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.addItems(self.get_audio_devices())
        self.audio_device_combo.currentIndexChanged.connect(self.update_audio_device)
        main_layout_tab.addRow(self.audio_device_label, self.audio_device_combo)

        self.audio_output_label = QLabel("Audio Output Device:")
        self.audio_output_combo = QComboBox()
        self.audio_output_combo.addItems(self.get_audio_devices(output=True))
        self.audio_output_combo.currentIndexChanged.connect(self.update_audio_output_device)
        main_layout_tab.addRow(self.audio_output_label, self.audio_output_combo)

        advanced_widget = QWidget()
        advanced_layout = QFormLayout()
        advanced_widget.setLayout(advanced_layout)
        self.tabs.addTab(advanced_widget, "Advanced Settings")

        self.pitch_shift_label = QLabel("Pitch Shift (semitones):")
        self.pitch_shift_slider = QSlider(Qt.Horizontal)
        self.pitch_shift_slider.setMinimum(-12)
        self.pitch_shift_slider.setMaximum(12)
        self.pitch_shift_slider.setValue(self.processor.pitch_shift)
        self.pitch_shift_slider.valueChanged.connect(self.update_pitch_shift)
        advanced_layout.addRow(self.pitch_shift_label, self.pitch_shift_slider)

        self.modulation_label = QLabel("Modulation Index:")
        self.modulation_slider = QSlider(Qt.Horizontal)
        self.modulation_slider.setMinimum(0)
        self.modulation_slider.setMaximum(100)
        self.modulation_slider.setValue(int(self.processor.modulation_index * 100))
        self.modulation_slider.valueChanged.connect(self.update_modulation)
        advanced_layout.addRow(self.modulation_label, self.modulation_slider)

        self.modulation_rate_label = QLabel("Modulation Rate (Hz):")
        self.modulation_rate_slider = QSlider(Qt.Horizontal)
        self.modulation_rate_slider.setMinimum(1)
        self.modulation_rate_slider.setMaximum(100)
        self.modulation_rate_slider.setValue(int(self.processor.modulation_rate * 10))
        self.modulation_rate_slider.valueChanged.connect(self.update_modulation_rate)
        advanced_layout.addRow(self.modulation_rate_label, self.modulation_rate_slider)

        self.vis_color_label = QLabel("Visualization Color:")
        self.vis_color_button = QPushButton("Choose Color")
        self.vis_color_button.clicked.connect(self.choose_vis_color)
        advanced_layout.addRow(self.vis_color_label, self.vis_color_button)

        self.vis_rate_label = QLabel("Visualization Update Rate (Hz):")
        self.vis_rate_spin = QSpinBox()
        self.vis_rate_spin.setRange(10, 60)
        self.vis_rate_spin.setValue(self.processor.vis_update_rate)
        self.vis_rate_spin.valueChanged.connect(self.update_vis_rate)
        advanced_layout.addRow(self.vis_rate_label, self.vis_rate_spin)

        self.particle_count_label = QLabel("Particle Count:")
        self.particle_count_slider = QSlider(Qt.Horizontal)
        self.particle_count_slider.setMinimum(100)
        self.particle_count_slider.setMaximum(5000)
        self.particle_count_slider.setValue(self.processor.particle_count)
        self.particle_count_slider.valueChanged.connect(self.update_particle_count)
        advanced_layout.addRow(self.particle_count_label, self.particle_count_slider)

        self.particle_size_label = QLabel("Particle Size:")
        self.particle_size_slider = QSlider(Qt.Horizontal)
        self.particle_size_slider.setMinimum(1)
        self.particle_size_slider.setMaximum(10)
        self.particle_size_slider.setValue(int(self.processor.particle_size * 5))
        self.particle_size_slider.valueChanged.connect(self.update_particle_size)
        advanced_layout.addRow(self.particle_size_label, self.particle_size_slider)

        self.lr_label = QLabel("Learning Rate:")
        self.lr_label.setToolTip("Learning rate for voice cloning training (1e-5 to 1e-3)")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-5, 1e-3)
        self.lr_spin.setValue(self.processor.learning_rate)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.valueChanged.connect(self.update_learning_rate)
        advanced_layout.addRow(self.lr_label, self.lr_spin)

        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_label.setToolTip("Batch size for voice cloning training (4–32)")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(4, 32)
        self.batch_size_spin.setValue(self.processor.batch_size)
        self.batch_size_spin.valueChanged.connect(self.update_batch_size)
        advanced_layout.addRow(self.batch_size_label, self.batch_size_spin)

        self.latency_threshold_label = QLabel("Latency Threshold (ms):")
        self.latency_threshold_label.setToolTip("Threshold for high-latency warnings (50–500 ms)")
        self.latency_threshold_spin = QSpinBox()
        self.latency_threshold_spin.setRange(50, 500)
        self.latency_threshold_spin.setValue(int(self.processor.latency_threshold * 1000))
        self.latency_threshold_spin.valueChanged.connect(self.update_latency_threshold)
        advanced_layout.addRow(self.latency_threshold_label, self.latency_threshold_spin)

        self.performance_mode_check = QCheckBox("Low Performance Mode")
        self.performance_mode_check.setToolTip("Enable to reduce visualization complexity on low-end systems")
        self.performance_mode_check.stateChanged.connect(self.toggle_performance_mode)
        advanced_layout.addRow(self.performance_mode_check)

        performance_widget = QWidget()
        performance_layout = QFormLayout()
        performance_widget.setLayout(performance_layout)
        self.tabs.addTab(performance_widget, "Performance")

        self.max_buffer_label = QLabel("Max Buffer Size (seconds):")
        self.max_buffer_label.setToolTip("Maximum audio buffer size (1–30 seconds)")
        self.max_buffer_spin = QSpinBox()
        self.max_buffer_spin.setRange(1, 30)
        self.max_buffer_spin.setValue(self.processor.max_buffer_size // SAMPLE_RATE)
        self.max_buffer_spin.valueChanged.connect(self.update_max_buffer)
        performance_layout.addRow(self.max_buffer_label, self.max_buffer_spin)

        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_widget.setLayout(buttons_layout)
        main_layout.addWidget(buttons_widget)

        self.upload_button = QPushButton("Upload Selfie")
        self.upload_button.clicked.connect(self.upload_selfie)
        buttons_layout.addWidget(self.upload_button)

        self.webcam_check = QCheckBox("Use Webcam")
        self.webcam_check.setChecked(self.processor.use_webcam)
        self.webcam_check.stateChanged.connect(self.toggle_webcam)
        buttons_layout.addWidget(self.webcam_check)

        self.webcam_label = QLabel("Webcam Device:")
        self.webcam_combo = QComboBox()
        self.webcam_combo.addItems(self.get_webcam_devices())
        self.webcam_combo.currentIndexChanged.connect(self.update_webcam_device)
        buttons_layout.addWidget(self.webcam_label)
        buttons_layout.addWidget(self.webcam_combo)

        self.animate_check = QCheckBox("Animate Face")
        self.animate_check.setChecked(self.processor.animate_face)
        self.animate_check.stateChanged.connect(self.toggle_animation)
        buttons_layout.addWidget(self.animate_check)

        self.animation_mode_label = QLabel("Animation Mode:")
        self.animation_mode_combo = QComboBox()
        self.animation_mode_combo.addItems(["Face Only", "Full Body", "Particles"])
        self.animation_mode_combo.setCurrentText(self.processor.animation_mode)
        self.animation_mode_combo.currentTextChanged.connect(self.update_animation_mode)
        buttons_layout.addWidget(self.animation_mode_label)
        buttons_layout.addWidget(self.animation_mode_combo)

        self.batch_button = QPushButton("Batch Process")
        self.batch_button.clicked.connect(self.batch_process)
        buttons_layout.addWidget(self.batch_button)

        self.vis_pause_button = QPushButton("Pause Visualization")
        self.vis_pause_button.setToolTip("Pause the visualization thread to save CPU")
        self.vis_pause_button.clicked.connect(self.toggle_vis_pause)
        buttons_layout.addWidget(self.vis_pause_button)

        self.credits_button = QPushButton("Credits")
        self.credits_button.clicked.connect(self.show_credits)
        buttons_layout.addWidget(self.credits_button)

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        buttons_layout.addWidget(self.help_button)

        self.online_help_button = QPushButton("Online Help")
        self.online_help_button.clicked.connect(self.show_online_help)
        buttons_layout.addWidget(self.online_help_button)

        self.web_ui_button = QPushButton("Launch Web UI")
        self.web_ui_button.clicked.connect(self.launch_web_ui)
        buttons_layout.addWidget(self.web_ui_button)

        zoom_widget = QWidget()
        zoom_layout = QHBoxLayout()
        zoom_widget.setLayout(zoom_layout)

        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(1, 10)
        self.zoom_spin.setValue(int(self.zoom_level))
        self.zoom_spin.valueChanged.connect(self.update_zoom)
        zoom_layout.addWidget(self.zoom_spin)

        self.zoom_label = QLabel(f"Zoom: {self.zoom_level:.1f}x")
        zoom_layout.addWidget(self.zoom_label)

        main_layout.addWidget(zoom_widget)

        self.setStyleSheet("""
            QMainWindow { background-color: #1A1A1A; }
            QLabel { font-family: 'Press Start 2P', monospace; font-size: 12px; }
            QSlider { background-color: #222222; }
        """)
        self.update_ui_from_config()

    def get_audio_devices(self, output=False):
        devices = []
        for i in range(self.processor.pa.get_device_count()):
            dev_info = self.processor.pa.get_device_info_by_index(i)
            if output and dev_info['maxOutputChannels'] > 0:
                devices.append(dev_info['name'])
            elif not output and dev_info['maxInputChannels'] > 0:
                devices.append(dev_info['name'])
        return devices

    def get_webcam_devices(self):
        devices = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append(f"Webcam {i}")
                cap.release()
        return devices if devices else ["No Webcams Detected"]

    def update_algorithm(self, value):
        try:
            self.processor.algorithm = value.lower()
            self.update_status(f"Vocoder algorithm set to {value}")
        except Exception as e:
            self.update_status(f"Error updating algorithm: {e}", error=True)
            logger.error(f"Error updating algorithm: {e}")

    def update_visualization_mode(self, value):
        try:
            self.processor.visualization_mode = value
            self.update_status(f"Visualization mode set to {value}")
            if self.processor.visualization_mode == "3D Particle Swarm" and not self.processor.vis_canvas:
                self.processor.init_vispy()
        except Exception as e:
            self.update_status(f"Error updating visualization mode: {e}", error=True)
            logger.error(f"Error updating visualization mode: {e}")

    def toggle_simplified_mode(self, state):
        try:
            self.tabs.setVisible(state == Qt.Unchecked)
            self.update_status("Toggled simplified mode")
        except Exception as e:
            self.update_status(f"Error toggling simplified mode: {e}", error=True)
            logger.error(f"Error toggling simplified mode: {e}")

    def update_volume(self, value):
        try:
            self.processor.volume = value / 100
            self.update_status(f"Volume set to {value}%")
        except Exception as e:
            self.update_status(f"Error updating volume: {e}", error=True)
            logger.error(f"Error updating volume: {e}")

    def update_pitch_shift(self, value):
        try:
            self.processor.pitch_shift = value
            self.update_status(f"Pitch shift set to {value} semitones")
        except Exception as e:
            self.update_status(f"Error updating pitch shift: {e}", error=True)
            logger.error(f"Error updating pitch shift: {e}")

    def update_modulation(self, value):
        try:
            self.processor.modulation_index = value / 100
            self.update_status(f"Modulation index set to {value}%")
        except Exception as e:
            self.update_status(f"Error updating modulation: {e}", error=True)
            logger.error(f"Error updating modulation: {e}")

    def update_modulation_rate(self, value):
        try:
            self.processor.modulation_rate = value / 10.0
            self.update_status(f"Modulation rate set to {value / 10.0:.1f} Hz")
        except Exception as e:
            self.update_status(f"Error updating modulation rate: {e}", error=True)
            logger.error(f"Error updating modulation rate: {e}")

    def choose_vis_color(self):
        try:
            color = QColorDialog.getColor(QtGui.QColor.fromRgbF(*self.processor.vis_color))
            if color.isValid():
                self.processor.vis_color = np.array([color.redF(), color.greenF(), color.blueF()])
                self.update_status(f"Visualization color set to RGB{self.processor.vis_color}")
        except Exception as e:
            self.update_status(f"Error choosing color: {e}", error=True)
            logger.error(f"Error choosing color: {e}")

    def update_vis_rate(self, value):
        try:
            self.processor.vis_update_rate = value
            self.update_status(f"Visualization update rate set to {value} Hz")
        except Exception as e:
            self.update_status(f"Error updating vis rate: {e}", error=True)
            logger.error(f"Error updating vis rate: {e}")

    def update_particle_count(self, value):
        try:
            self.processor.particle_count = value
            self.update_status(f"Particle count set to {value}")
        except Exception as e:
            self.update_status(f"Error updating particle count: {e}", error=True)
            logger.error(f"Error updating particle count: {e}")

    def update_particle_size(self, value):
        try:
            self.processor.particle_size = value / 5.0
            self.update_status(f"Particle size set to {value / 5.0:.1f}")
        except Exception as e:
            self.update_status(f"Error updating particle size: {e}", error=True)
            logger.error(f"Error updating particle size: {e}")

    def update_learning_rate(self, value):
        try:
            self.processor.learning_rate = value
            self.update_status(f"Learning rate set to {value}")
        except Exception as e:
            self.update_status(f"Error updating learning rate: {e}", error=True)
            logger.error(f"Error updating learning rate: {e}")

    def update_batch_size(self, value):
        try:
            self.processor.batch_size = value
            self.update_status(f"Batch size set to {value}")
        except Exception as e:
            self.update_status(f"Error updating batch size: {e}", error=True)
            logger.error(f"Error updating batch size: {e}")

    def update_latency_threshold(self, value):
        try:
            self.processor.latency_threshold = value / 1000
            self.update_status(f"Latency threshold set to {value} ms")
        except Exception as e:
            self.update_status(f"Error updating latency threshold: {e}", error=True)
            logger.error(f"Error updating latency threshold: {e}")

    def toggle_performance_mode(self, state):
        try:
            self.processor.low_performance_mode = bool(state)
            self.update_status(f"Low performance mode {'enabled' if state else 'disabled'}")
        except Exception as e:
            self.update_status(f"Error toggling performance mode: {e}", error=True)
            logger.error(f"Error toggling performance mode: {e}")

    def update_max_buffer(self, value):
        try:
            self.processor.max_buffer_size = value * SAMPLE_RATE
            self.update_status(f"Max buffer size set to {value} seconds")
        except Exception as e:
            self.update_status(f"Error updating max buffer: {e}", error=True)
            logger.error(f"Error updating max buffer: {e}")

    def update_audio_device(self, index):
        try:
            if index >= 0:
                self.processor.input_device_index = index
                self.update_status(f"Audio input device set to {self.audio_device_combo.currentText()}")
            else:
                self.processor.input_device_index = None
                self.update_status("Audio input device reset to default")
        except Exception as e:
            self.update_status(f"Error updating audio device: {e}", error=True)
            logger.error(f"Error updating audio device: {e}")

    def update_audio_output_device(self, index):
        try:
            if index >= 0:
                self.processor.output_device_index = index
                self.update_status(f"Audio output device set to {self.audio_output_combo.currentText()}")
            else:
                self.processor.output_device_index = None
                self.update_status("Audio output device reset to default")
        except Exception as e:
            self.update_status(f"Error updating audio output device: {e}", error=True)
            logger.error(f"Error updating audio output device: {e}")

    def update_webcam_device(self, index):
        try:
            self.processor.webcam_index = index
            if self.processor.use_webcam:
                self.processor.release_webcam()
                self.processor.init_webcam()
            self.update_status(f"Webcam device set to {self.webcam_combo.currentText()}")
        except Exception as e:
            self.update_status(f"Error updating webcam device: {e}", error=True)
            logger.error(f"Error updating webcam device: {e}")

    def toggle_webcam(self, state):
        try:
            self.processor.use_webcam = bool(state)
            if self.processor.use_webcam:
                self.processor.init_webcam()
            else:
                self.processor.release_webcam()
            self.update_status(f"Webcam {'enabled' if state else 'disabled'}")
        except Exception as e:
            self.update_status(f"Error toggling webcam: {e}", error=True)
            logger.error(f"Error toggling webcam: {e}")

    def toggle_animation(self, state):
        try:
            self.processor.animate_face = bool(state)
            self.update_status(f"Face animation {'enabled' if state else 'disabled'}")
        except Exception as e:
            self.update_status(f"Error toggling animation: {e}", error=True)
            logger.error(f"Error toggling animation: {e}")

    def update_animation_mode(self, value):
        try:
            self.processor.animation_mode = value
            self.update_status(f"Animation mode set to {value}")
        except Exception as e:
            self.update_status(f"Error updating animation mode: {e}", error=True)
            logger.error(f"Error updating animation mode: {e}")

    def upload_selfie(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(self, "Select Selfie Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            if filename:
                self.processor.load_selfie(filename)
                self.update_status(f"Selfie uploaded from {filename}")
        except Exception as e:
            self.update_status(f"Error uploading selfie: {e}", error=True)
            logger.error(f"Error uploading selfie: {e}")

    def start_processing(self):
        try:
            if not self.processor.stream:
                self.processor.start_stream()
            self.update_status("Processing started")
        except Exception as e:
            self.update_status(f"Error starting processing: {e}", error=True)
            logger.error(f"Error starting processing: {e}")

    def stop_processing(self):
        try:
            self.processor.stop_stream()
            self.update_status("Processing stopped")
        except Exception as e:
            self.update_status(f"Error stopping processing: {e}", error=True)
            logger.error(f"Error stopping processing: {e}")

    def toggle_recording(self):
        try:
            self.processor.recording = not self.processor.recording
            if self.processor.recording:
                self.processor.audio_data = []
                self.processor.spectral_data = []
                self.processor.phoneme_data = []
                self.processor.animation_frames = []
                self.update_status("Recording started")
            else:
                self.update_status("Recording stopped")
            logger.info(f"Recording {'started' if self.processor.recording else 'stopped'}")
        except Exception as e:
            self.update_status(f"Error toggling recording: {e}", error=True)
            logger.error(f"Error toggling recording: {e}")

    def save_recording(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Recording", str(RECORDINGS_DIR / "recording.wav"), "WAV Files (*.wav);;MP4 Files (*.mp4)")
            if filename:
                self.processor.save_recording(filename)
                self.update_status(f"Recording saved to {filename}")
        except Exception as e:
            self.update_status(f"Error saving recording: {e}", error=True)
            logger.error(f"Error saving recording: {e}")

    def batch_process(self):
        try:
            input_files, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", str(RECORDINGS_DIR), "Audio Files (*.wav *.mp3 *.flac)")
            if input_files:
                output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(RECORDINGS_DIR))
                if output_dir:
                    results = self.processor.batch_process(input_files, output_dir)
                    self.update_status(f"Batch processed {len(results)} files to {output_dir}")
                    for result in results:
                        logger.info(f"Batch result: {result}")
        except Exception as e:
            self.update_status(f"Error in batch processing: {e}", error=True)
            logger.error(f"Error in batch processing: {e}")

    def toggle_vis_pause(self):
        try:
            self.processor.visualization_running = not self.processor.visualization_running
            self.update_status(f"Visualization {'paused' if not self.processor.visualization_running else 'resumed'}")
            self.vis_pause_button.setText("Resume Visualization" if not self.processor.visualization_running else "Pause Visualization")
        except Exception as e:
            self.update_status(f"Error toggling visualization pause: {e}", error=True)
            logger.error(f"Error toggling visualization pause: {e}")

    def start_visualization_thread(self):
        self.vis_thread = Thread(target=self.update_visualization)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def update_visualization(self):
    try:
        while self.running and self.processor.visualization_running:
            if not self.processor.queue.empty() and not self.processor.spectral_queue.empty() and not self.processor.phoneme_queue.empty():
                with self.processor.data_lock:
                    if not self.processor.queue.empty():
                        audio = self.processor.queue.get()
                    else:
                        audio = np.zeros(self.processor.chunk_size, dtype=np.float32)
                    if not self.processor.spectral_queue.empty():
                        magnitude = self.processor.spectral_queue.get()
                    else:
                        magnitude = np.zeros((N_FFT // 2 + 1, self.processor.chunk_size // HOP_SIZE), dtype=np.float32)
                    if not self.processor.phoneme_queue.empty():
                        phonemes, timings = self.processor.phoneme_queue.get()
                    else:
                        phonemes, timings = [], []
                    if not self.processor.animation_queue.empty() and self.processor.animate_face and self.processor.animation_mode == "Face Only":
                        frame = self.processor.animation_queue.get()
                    else:
                        frame = None

                self.figure.clear()
                ax = self.figure.add_subplot(111, projection='3d' if self.processor.visualization_mode == "3D Waterfall" else None)
                freqs = np.linspace(0, SAMPLE_RATE / 2, magnitude.shape[0])
                times = np.linspace(0, len(audio) / SAMPLE_RATE, magnitude.shape[1])

                if self.processor.visualization_mode == "Spectrogram":
                    ax.imshow(amplitude_to_db(magnitude, ref=np.max(magnitude)), aspect='auto', origin='lower',
                              extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap='viridis')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Frequency (Hz)')
                    ax.set_title('Spectrogram')
                elif self.processor.visualization_mode == "3D Waterfall":
                    step = 2 if self.processor.low_performance_mode else 1
                    for i in range(0, magnitude.shape[1], step):
                        ax.plot(freqs, np.ones_like(freqs) * times[i],
                                amplitude_to_db(magnitude[:, i], ref=np.max(magnitude)),
                                c=tuple(self.processor.vis_color))
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Time (s)')
                    ax.set_zlabel('Amplitude (dB)')
                    ax.set_title('3D Waterfall')
                elif self.processor.visualization_mode == "Frequency Bar":
                    freq_magnitude = np.mean(magnitude, axis=1)
                    ax.bar(freqs, amplitude_to_db(freq_magnitude, ref=np.max(freq_magnitude)),
                           width=freqs[1] - freqs[0], color=tuple(self.processor.vis_color))
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Amplitude (dB)')
                    ax.set_title('Frequency Bar')
                elif self.processor.visualization_mode == "Mel Spectrogram":
                    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_SIZE)
                    ax.imshow(amplitude_to_db(mel_spec, ref=np.max(mel_spec)), aspect='auto', origin='lower',
                              extent=[times[0], times[-1], 0, 128], cmap='magma')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Mel Bins')
                    ax.set_title('Mel Spectrogram')
                elif self.processor.visualization_mode == "Waveform Envelope":
                    ax.plot(times, audio, color=tuple(self.processor.vis_color))
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title('Waveform Envelope')
                elif self.processor.visualization_mode == "Chroma Features":
                    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_SIZE)
                    ax.imshow(chroma, aspect='auto', origin='lower',
                              extent=[times[0], times[-1], 0, 12], cmap='coolwarm')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Chroma')
                    ax.set_title('Chroma Features')
                elif self.processor.visualization_mode == "3D Particle Swarm":
                    if not self.processor.vis_canvas:
                        self.processor.init_vispy()
                    self.processor.animate_particles(audio)
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, "Particle Swarm Active (Vispy Window)", ha='center', va='center')
                    ax.axis('off')

                if self.processor.animate_face and frame is not None and self.processor.animation_mode == "Face Only":
                    ax2 = self.figure.add_subplot(111, position=[0.7, 0.7, 0.25, 0.25])
                    ax2.imshow(frame)
                    ax2.axis('off')

                if phonemes:
                    ax.text(0.05, 0.95, f"Phoneme: {phonemes[-1] if phonemes else ''}", transform=ax.transAxes,
                            color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

                self.canvas.draw()
                time.sleep(1 / self.processor.vis_update_rate)
            else:
                time.sleep(0.01)  # Prevent tight loop when queues are empty
    except Exception as e:
        self.update_status(f"Error in visualization: {e}", error=True)
        logger.error(f"Error in visualization: {e}")

def start_latency_monitor(self):
    self.latency_timer = QTimer(self)
    self.latency_timer.timeout.connect(self.update_latency)
    self.latency_timer.start(1000)

def update_latency(self):
    try:
        if not self.processor.latency_queue.empty():
            latencies = []
            while not self.processor.latency_queue.empty():
                latencies.append(self.processor.latency_queue.get())
            if latencies:
                avg_latency = np.mean(latencies) * 1000  # Convert to ms
                self.latency_label.setText(f"Latency: {avg_latency:.1f} ms")
                if avg_latency > self.processor.latency_threshold * 1000:
                    self.latency_label.setStyleSheet("color: #FF0000;")
                else:
                    self.latency_label.setStyleSheet("color: #00FFFF;")
    except Exception as e:
        self.update_status(f"Error updating latency: {e}", error=True)
        logger.error(f"Error updating latency: {e}")

def update_equalizer(self):
    try:
        if not self.processor.spectral_queue.empty():
            with self.processor.data_lock:
                if not self.processor.spectral_queue.empty():
                    magnitude = self.processor.spectral_queue.get()
                else:
                    magnitude = np.zeros((N_FFT // 2 + 1, self.processor.chunk_size // HOP_SIZE))
            freq_bands = np.mean(magnitude, axis=1)
            freq_bands = amplitude_to_db(freq_bands, ref=np.max(freq_bands))
            freq_bands = np.clip((freq_bands + 100) / 100 * 100, 0, 100)  # Normalize to 0-100
            for i, bar in enumerate(self.equalizer_bars):
                if i < len(freq_bands):
                    bar.setValue(int(freq_bands[i]))
                else:
                    bar.setValue(0)
    except Exception as e:
        self.update_status(f"Error updating equalizer: {e}", error=True)
        logger.error(f"Error updating equalizer: {e}")

def update_live_tts(self, text):
    try:
        self.processor.text_input = text
        if self.processor.use_cloned_voice and text.strip():
            audio, phonemes, timings = self.processor.text_to_speech(text, self.processor.speaker_id)
            with self.processor.data_lock:
                self.processor.queue.put(audio)
                self.processor.phoneme_queue.put((phonemes, timings))
            self.update_status(f"Live TTS updated with text: {text[:20]}...")
    except Exception as e:
        self.update_status(f"Error in live TTS: {e}", error=True)
        logger.error(f"Error in live TTS: {e}")

def update_status(self, message, error=False):
    self.status_label.setText(f"Status: {message}")
    self.status_label.setStyleSheet("color: #FF0000;" if error else "color: #00FFFF;")

def save_settings_state(self):
    try:
        state = {
            "algorithm": self.processor.algorithm,
            "visualization_mode": self.processor.visualization_mode,
            "pitch_shift": self.processor.pitch_shift,
            "modulation_index": self.processor.modulation_index,
            "modulation_rate": self.processor.modulation_rate,
            "volume": self.processor.volume,
            "vis_color": self.processor.vis_color.tolist(),
            "animate_face": self.processor.animate_face,
            "animation_mode": self.processor.animation_mode,
            "use_cloned_voice": self.processor.use_cloned_voice,
            "use_webcam": self.processor.use_webcam,
            "selfie_path": self.processor.selfie_path,
            "vis_update_rate": self.processor.vis_update_rate,
            "particle_count": self.processor.particle_count,
            "particle_size": self.processor.particle_size,
            "training_epochs": self.processor.training_epochs,
            "prosody_strength": self.processor.prosody_strength,
            "speaker_id": self.processor.speaker_id,
            "text_input": self.processor.text_input,
            "learning_rate": self.processor.learning_rate,
            "batch_size": self.processor.batch_size,
            "latency_threshold": self.processor.latency_threshold,
            "max_buffer_size": self.processor.max_buffer_size,
            "low_performance_mode": self.processor.low_performance_mode
        }
        if len(self.settings_history) > self.settings_index + 1:
            self.settings_history = self.settings_history[:self.settings_index + 1]
        self.settings_history.append(state)
        self.settings_index += 1
        with open(CONFIG_DIR / "config.json", 'w') as f:
            json.dump(state, f, indent=4)
        logger.info("Settings state saved")
    except Exception as e:
        self.update_status(f"Error saving settings: {e}", error=True)
        logger.error(f"Error saving settings: {e}")

def undo_settings(self):
    try:
        if self.settings_index > 0:
            self.settings_index -= 1
            self.load_settings_state(self.settings_history[self.settings_index])
            self.update_status("Settings undone")
    except Exception as e:
        self.update_status(f"Error undoing settings: {e}", error=True)
        logger.error(f"Error undoing settings: {e}")

def redo_settings(self):
    try:
        if self.settings_index < len(self.settings_history) - 1:
            self.settings_index += 1
            self.load_settings_state(self.settings_history[self.settings_index])
            self.update_status("Settings redone")
    except Exception as e:
        self.update_status(f"Error redoing settings: {e}", error=True)
        logger.error(f"Error redoing settings: {e}")

def load_settings_state(self, state):
    try:
        self.processor.algorithm = state["algorithm"]
        self.processor.visualization_mode = state["visualization_mode"]
        self.processor.pitch_shift = state["pitch_shift"]
        self.processor.modulation_index = state["modulation_index"]
        self.processor.modulation_rate = state["modulation_rate"]
        self.processor.volume = state["volume"]
        self.processor.vis_color = np.array(state["vis_color"])
        self.processor.animate_face = state["animate_face"]
        self.processor.animation_mode = state["animation_mode"]
        self.processor.use_cloned_voice = state["use_cloned_voice"]
        self.processor.use_webcam = state["use_webcam"]
        self.processor.selfie_path = state["selfie_path"]
        self.processor.vis_update_rate = state["vis_update_rate"]
        self.processor.particle_count = state["particle_count"]
        self.processor.particle_size = state["particle_size"]
        self.processor.training_epochs = state["training_epochs"]
        self.processor.prosody_strength = state["prosody_strength"]
        self.processor.speaker_id = state["speaker_id"]
        self.processor.text_input = state["text_input"]
        self.processor.learning_rate = state["learning_rate"]
        self.processor.batch_size = state["batch_size"]
        self.processor.latency_threshold = state["latency_threshold"]
        self.processor.max_buffer_size = state["max_buffer_size"]
        self.processor.low_performance_mode = state["low_performance_mode"]
        self.update_ui_from_config()
        logger.info("Settings state loaded")
    except Exception as e:
        self.update_status(f"Error loading settings: {e}", error=True)
        logger.error(f"Error loading settings: {e}")

def update_ui_from_config(self):
    try:
        self.algorithm_combo.setCurrentText(self.processor.algorithm.capitalize())
        self.vis_mode_combo.setCurrentText(self.processor.visualization_mode)
        self.simplified_mode_check.setChecked(self.tabs.isHidden())
        self.volume_slider.setValue(int(self.processor.volume * 100))
        self.pitch_shift_slider.setValue(self.processor.pitch_shift)
        self.modulation_slider.setValue(int(self.processor.modulation_index * 100))
        self.modulation_rate_slider.setValue(int(self.processor.modulation_rate * 10))
        self.vis_rate_spin.setValue(self.processor.vis_update_rate)
        self.particle_count_slider.setValue(self.processor.particle_count)
        self.particle_size_slider.setValue(int(self.processor.particle_size * 5))
        self.lr_spin.setValue(self.processor.learning_rate)
        self.batch_size_spin.setValue(self.processor.batch_size)
        self.latency_threshold_spin.setValue(int(self.processor.latency_threshold * 1000))
        self.performance_mode_check.setChecked(self.processor.low_performance_mode)
        self.max_buffer_spin.setValue(self.processor.max_buffer_size // SAMPLE_RATE)
        self.webcam_check.setChecked(self.processor.use_webcam)
        self.animate_check.setChecked(self.processor.animate_face)
        self.animation_mode_combo.setCurrentText(self.processor.animation_mode)
        self.tts_input.setText(self.processor.text_input)
        if self.processor.input_device_index is not None:
            self.audio_device_combo.setCurrentIndex(self.processor.input_device_index)
        if self.processor.output_device_index is not None:
            self.audio_output_combo.setCurrentIndex(self.processor.output_device_index)
        self.webcam_combo.setCurrentIndex(self.processor.webcam_index)
    except Exception as e:
        self.update_status(f"Error updating UI from config: {e}", error=True)
        logger.error(f"Error updating UI from config: {e}")

def show_help(self):
    try:
        help_file = DOCS_DIR / "user_manual.html"
        if help_file.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(help_file)))
            self.update_status("Opened user manual")
        else:
            self.update_status("User manual not found", error=True)
            logger.error("User manual not found")
    except Exception as e:
        self.update_status(f"Error opening help: {e}", error=True)
        logger.error(f"Error opening help: {e}")

def show_online_help(self):
    try:
        QDesktopServices.openUrl(QUrl("https://github.com/username/BGGG_Vocoder"))
        self.update_status("Opened online help")
    except Exception as e:
        self.update_status(f"Error opening online help: {e}", error=True)
        logger.error(f"Error opening online help: {e}")

def launch_web_ui(self):
    try:
        from flask import Flask, render_template_string, request, jsonify
        app = Flask(__name__)

        @app.route('/')
        def index():
            with open(THEMES_DIR / "web_ui.html", 'r') as f:
                template = f.read()
            return render_template_string(template, config=self.processor.__dict__)

        @app.route('/update_config', methods=['POST'])
        def update_config():
            try:
                data = request.get_json()
                self.processor.algorithm = data.get('algorithm', self.processor.algorithm)
                self.processor.visualization_mode = data.get('visualization_mode', self.processor.visualization_mode)
                self.processor.volume = float(data.get('volume', self.processor.volume))
                self.processor.modulation_rate = float(data.get('modulation_rate', self.processor.modulation_rate))
                self.save_settings_state()
                return jsonify({"message": "Configuration updated"})
            except Exception as e:
                logger.error(f"Web UI config update failed: {e}")
                return jsonify({"message": f"Error: {e}"}), 500

        @app.route('/start_processing', methods=['POST'])
        def start_processing():
            try:
                self.start_processing()
                return jsonify({"message": "Processing started"})
            except Exception as e:
                logger.error(f"Web UI start processing failed: {e}")
                return jsonify({"message": f"Error: {e}"}), 500

        @app.route('/stop_processing', methods=['POST'])
        def stop_processing():
            try:
                self.stop_processing()
                return jsonify({"message": "Processing stopped"})
            except Exception as e:
                logger.error(f"Web UI stop processing failed: {e}")
                return jsonify({"message": f"Error: {e}"}), 500

        def run_flask():
            app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

        flask_thread = Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        QDesktopServices.openUrl(QUrl("http://localhost:5000"))
        self.update_status("Web UI launched at http://localhost:5000")
    except Exception as e:
        self.update_status(f"Error launching web UI: {e}", error=True)
        logger.error(f"Error launching web UI: {e}")

def on_scroll(self, event):
    try:
        if event.button == 'up':
            self.zoom_level = min(self.zoom_level * 1.1, 10)
        elif event.button == 'down':
            self.zoom_level = max(self.zoom_level / 1.1, 1)
        self.zoom_spin.setValue(int(self.zoom_level))
        self.zoom_label.setText(f"Zoom: {self.zoom_level:.1f}x")
        self.canvas.draw()
    except Exception as e:
        self.update_status(f"Error in scroll event: {e}", error=True)
        logger.error(f"Error in scroll event: {e}")

def on_pan(self, event):
    try:
        if event.button == 1:  # Left mouse button
            self.pan_position[0] += event.xdata if event.xdata else 0
            self.pan_position[1] += event.ydata if event.ydata else 0
            self.canvas.draw()
    except Exception as e:
        self.update_status(f"Error in pan event: {e}", error=True)
        logger.error(f"Error in pan event: {e}")

def closeEvent(self, event):
    try:
        # Show exit screen
        exit_dialog = QDialog(self)
        exit_dialog.setWindowTitle("BGGG Vocoder - Exiting")
        exit_dialog.setWindowFlags(Qt.FramelessWindowHint)
        layout = QVBoxLayout()
        exit_label = QLabel()
        pixmap = QPixmap(str(ASSETS_DIR / "exit_screen.png"))
        if pixmap.isNull():
            exit_label.setText("Exit Screen Missing")
            logger.warning("exit_screen.png not found in assets/")
        else:
            exit_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        layout.addWidget(exit_label)
        exit_dialog.setLayout(layout)
        exit_dialog.resize(800, 600)
        exit_dialog.show()
        QTimer.singleShot(3000, exit_dialog.close)  # Close after 3 seconds

        self.running = False
        self.processor.stop_stream()
        self.processor.running = False
        self.processor.release_webcam()
        if self.processor.vis_canvas:
            self.processor.vis_canvas.close()
        self.save_settings_state()
        event.accept()
        logger.info("Application closed")
    except Exception as e:
        self.update_status(f"Error closing application: {e}", error=True)
        logger.error(f"Error closing application: {e}")

def main():
    try:
        check_disk_space(PROJECT_DIR, min_space_mb=500)
        setup_project_structure()
        install_dependencies()
        app = QApplication(sys.argv)
        font_path = str(ASSETS_DIR / "PressStart2P-Regular.ttf")
        if os.path.exists(font_path):
            QFontDatabase().addApplicationFont(font_path)
        window = BGGGVocoderApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        QMessageBox.critical(None, "Startup Error", f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
