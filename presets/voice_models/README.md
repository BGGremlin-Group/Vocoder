# 🎙️ BGGG_Vocoder V5.0: Your Ultimate Voice and Animation Studio with Retro Flair 🎨

Welcome to **BGGG_Vocoder V5.0**, a professional-grade vocoder application for Windows 11 that transforms your voice and visuals into a creative masterpiece! 🌟 Featuring a nostalgic Winamp-style UI with a dark theme and neon accents, this tool offers real-time voice effects, voice cloning, text-to-speech with live input, precise phoneme-aligned lip-sync, and stunning 3D animations. 🚀

With advanced vocoder algorithms, immersive spectrum visualizations, support for full-body animation with OBJ models, audio-reactive particle systems, and live webcam integration, BGGG_Vocoder V5.0 blends retro aesthetics with cutting-edge functionality for crafting professional audio-visual experiences. 🎵🎥

## ✨ Features

- **Advanced Vocoder Effects** 🎶
  - Three algorithms: **Phase Vocoder** (natural sound), **LPC Vocoder** (robotic tone), **Channel Vocoder** (versatile frequency modulation).
  - Real-time pitch adjustments (-12 to +12 semitones).
  - Analogue-style modulation with depth (0–1) and rate (0.1–10 Hz) controls.

- **Voice Cloning** 🗣️
  - Train voice models from live recordings or imported MP3/MP4 files (minimum 10 seconds).
  - Save and reuse cloned voices for consistent, personalized audio output.
  - Offline-compatible with lightweight Tacotron 2 and HiFi-GAN models via `coqui-ai/TTS`.

- **Text-to-Speech with Phoneme Alignment** 📜
  - Convert text to speech using cloned voices with precise phoneme alignment.
  - Real-time text input for live TTS, synchronized with animations for realistic lip-sync.

- **Phoneme Detection for Precise Lip-Sync** 👄
  - Detect phonemes in real-time, recorded, or imported audio using `speechbrain`.
  - Optimized for noisy environments with voice activity detection (VAD) and preemphasis filtering.
  - Map phonemes to visemes (e.g., open mouth for vowels, closed for consonants).

- **Immersive Visualizations** 📊
  - Spectrum analysis modes: **Spectrogram**, **3D Waterfall**, **Frequency Bar**, **Mel Spectrogram**, **Waveform Envelope**, **Chroma Features**, **3D Particle Swarm**.
  - Audio-reactive colors and animations, customizable via GUI controls.

- **Facial and Full-Body Animation** 🕺
  - Upload a selfie or use live webcam input for real-time facial animation with phoneme-driven lip-sync.
  - Full-body 3D animation with support for OBJ model files, rendered using `trimesh` and `vispy`, synchronized with audio energy (RMS, pitch).
  - Advanced particle system mode with audio-reactive particles, customizable count (100–5000) and size (0.2–2.0).

- **Real-Time Webcam Support** 📹
  - Live facial animation using webcam feed, optimized for low-latency processing.
  - Supports multiple webcam devices with user selection.

- **Export and Import** 💾
  - Import MP3, WAV, or MP4 files for processing or voice cloning.
  - Export recordings as video+audio MP4 (with facial animation) or audio-only WAV/MP4.

- **Retro Winamp-Style GUI** 🖥️
  - Dark mode with neon/pastel colors (cyan, magenta; red for errors).
  - Winamp-inspired layout with metallic borders, 8-band equalizer bars, real-time TTS input, and playlist-style status bar.
  - Simplified mode hides advanced settings for a true retro experience.
  - Single-window interface with logo, credits, and exit screens (3-second display each).
  - Pixelated "Press Start 2P" font for a nostalgic feel.

- **Robust and Reliable** 🛡️
  - Comprehensive error handling with detailed status messages in the GUI and log file (`vocoder.log`).
  - Automatic dependency installation (`pyaudio`, `numpy`, `torch`, `speechbrain`, `trimesh`, etc.) and asset downloading (e.g., dlib shape predictor).
  - Configuration saved across sessions in `~/BGGG_Vocoder/config/config.json`.
  - Dynamic chunk size adjustment based on system load for low-latency performance.
  - Latency monitoring with warnings for high latency (>50–500 ms, user-configurable).

- **Web UI** 🌐
  - Browser-based interface at `http://localhost:5000` for remote control of vocoder settings and processing.
  - Supports algorithm selection, visualization mode, volume, and modulation rate adjustments.

- **Performance Optimization** ⚙️
  - Low-performance mode reduces visualization complexity for low-end systems.
  - Adjustable buffer size (1–30 seconds) and visualization update rate (10–60 Hz).
  - Noise-robust phoneme detection using `webrtcvad` for voice activity detection.

## 🚀 Getting Started

### Prerequisites
- **Operating System**: Windows 11
- **Hardware**: Microphone, speakers, and webcam (for live facial animation)
- **Python**: 3.8 or higher
- **Disk Space**: ~500 MB for dependencies and assets

### Installation
1. **Clone or Download**:
   ```bash
   git clone [https://github.com/your-repo/bggg_vocoder.git](https://github.com/BGGremlin-Group/Vocoder.git/
   cd Vocoder
   ```
   Or download `bggg_vocoder_v5_0.py` from the releases page.

2. **Run the Application**:
   ```bash
   python bggg_vocoder_v5_0.py
   ```
   The script automatically installs dependencies (`pyaudio`, `numpy`, `torch`, `speechbrain`, `trimesh`, `webrtcvad`, etc.) and downloads required assets (e.g., `shape_predictor_68_face_landmarks.dat`).

3. **Directory Structure**:
   - `~/BGGG_Vocoder/assets`: Stores logo, credits, exit screens, static visualizer, and dlib shape predictor.
   - `~/BGGG_Vocoder/config`: Stores `config.json` for settings.
   - `~/BGGG_Vocoder/recordings`: Saves WAV and MP4 outputs.
   - `~/BGGG_Vocoder/voice_models`: Stores trained voice models (`.pth`).
   - `~/BGGG_Vocoder/presets`: Stores preset configurations.
   - `~/BGGG_Vocoder/themes`: Stores web UI template (`web_ui.html`).
   - `~/BGGG_Vocoder/docs`: Stores user manual (`user_manual.html`).

### Usage
1. **Launch the App**:
   - The app starts with a logo screen (3s), followed by credits (3s), then opens the retro Winamp-style interface.

2. **Configure Settings**:
   - Select a vocoder algorithm (Phase, LPC, Channel) via the control panel.
   - Choose a visualization mode (Spectrogram, 3D Waterfall, etc.) via the main settings tab.
   - Enable simplified mode to hide advanced settings for a retro experience.
   - Adjust pitch, modulation depth, modulation rate, volume, particle count, and particle size using sliders.
   - Upload a selfie or enable webcam for facial animation.
   - Train a voice model via "Train Voice" (record 10+ seconds) or "Import Voice" (MP3/MP4).
   - Enter text in the TTS input field for real-time text-to-speech.
   - Select animation mode (Face Only, Full Body, Particles) and load OBJ files for full-body animation.

3. **Process Audio**:
   - Click "▶ Play" for real-time effects.
   - Use "● Rec" to capture audio and animations.
   - Import MP3/WAV/MP4 files via "Batch Process" in the main settings tab.
   - Export recordings as WAV or MP4 (video+audio) via "💾 Save".

4. **Visualize and Animate**:
   - Watch real-time visualizations (equalizer or spectrum) in the main panel.
   - Enable "Animate Face" for lip-sync with selfie/webcam.
   - Select "Full Body" for 3D OBJ model animation or "Particles" for audio-reactive particle effects.

5. **Access Web UI**:
   - Click "Launch Web UI" to open `http://localhost:5000` for remote control.

6. **Access Help**:
   - Click "Help" for the local user manual or "Online Help" for the GitHub repository.

### Example Workflow
- **Create a Cloned Voice Video with Live TTS**:
  1. Upload a selfie or enable webcam.
  2. Train a voice model with a 10+ second MP3 file or live recording.
  3. Enter text in the TTS input field (e.g., "Hello, this is my digital clone!").
  4. Enable "Use Cloned Voice" and "Animate Face" (Face Only mode).
  5. Start recording with "● Rec", then export as MP4 to get a video with your cloned voice and lip-synced animation.

## 🛠️ Troubleshooting
- **Audio Issues**: Ensure microphone/speakers are configured in Windows 11 settings. Check audio device selection in the GUI.
- **Webcam Issues**: Verify webcam permissions and lighting. Select the correct webcam device in the GUI.
- **Voice Cloning**: Use clear audio samples (minimal background noise) for best results.
- **Performance**: Enable "Low Performance Mode" or increase buffer size if latency occurs. Adjust chunk size dynamically via system load monitoring.
- **Errors**: Check the status display (bottom of GUI) or `vocoder.log` for detailed error messages.
- **OBJ Models**: Ensure OBJ files are valid and placed in the `assets` directory for full-body animation.

For support, email **Message Us On GitHub** 📧.

## 🧑‍💻 Contributing
We welcome contributions! 🙌 To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Commit changes (`git commit -m "Add awesome feature"`).
4. Push to the branch (`git push origin feature/awesome-feature`).
5. Open a Pull Request.

Please ensure code is well-documented.

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🎉 Acknowledgments
- Built with ❤️ by the BG Gremlin Group.
- Thanks to the open-source community for libraries like `torch`, `speechbrain`, `trimesh`, `webrtcvad`, `vispy`, and more.
- Ai platforms utilized: xAi, OpenAi.

---

**BGGG_Vocoder V5.0** combines the nostalgia of Winamp with modern voice and animation capabilities. Clone your voice, animate your digital persona, and visualize your sound in style. Try it today and unleash your creativity! 🚀🎙️

### The Background Gremlin Group
***Creating Unique Tools for Unique Individuals***

---

### Key Updates and Alignments
- **Version**: Updated to V5.0, reflecting all implemented features in the provided code.
- **Retro UI**: Details the Winamp-inspired layout, simplified mode for hiding tabs, neon/pastel colors, and pixelated font. Includes logo, credits, and exit screens.
- **Vocoder Effects**: Specifies modulation rate (0.1–10 Hz) and pitch shift range, aligning with `apply_modulation` and `apply_pitch_shift`.
- **Voice Cloning**: Notes 10-second minimum for training, matching `train_voice`.
- **Real-Time TTS**: Highlights live text input via `tts_input` QLineEdit, implemented in `update_live_tts`.
- **Phoneme Detection**: Mentions VAD and preemphasis for noise robustness, as implemented in `detect_phonemes` with `webrtcvad`.
- **Visualizations**: Lists all modes (Spectrogram, 3D Waterfall, etc.), matching `update_visualization`.
- **Animations**: Includes OBJ support via `trimesh`/`vispy` and customizable particle parameters, as in `animate_full_body` and `animate_particles`.
- **Web UI**: Describes browser-based control at `http://localhost:5000`, implemented with Flask.
- **Robustness**: Covers error handling, auto-installation, config saving, and latency monitoring, aligning with `@handle_errors`, `install_dependencies`, and `update_latency`.
- **Discrepancies Fixed**:
  - Modulation rate corrected to 0.1–10 Hz.
  - Rendering libraries updated to `trimesh`/`vispy` instead of PyOpenGL.
  - Simplified mode clarified to hide tabs.
  - Exit screen included.
  - Disk space requirement set to 500 MB.
- **Directory Structure**: Updated to include `presets`, `themes`, and `docs` directories.

### ***BG Gremlin Group 2025***
