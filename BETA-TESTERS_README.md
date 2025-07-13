# 📢 BGGG_Vocoder V5.0 Beta Testing Notes (Focus: Voice Cloning from MP3/MP4, Vocoder, Interface, Avatar Creation, TTS with Avatar)

Thank you for beta testing **BGGG_Vocoder V5.0**!

 🎉 This Windows 11 application delivers a retro Winamp-style UI with advanced audio processing, and we’re prioritizing your feedback on **voice cloning** (especially from MP3 and MP4 files), **vocoder effects**, **interface and integration**, **avatar creation**, and **TTS with avatar**.

 The PNG assets (`logo_screen.png`, `credits_screen.png`, `exit_screen.png`, `static_visualizer.png`) are uploaded to [https://github.com/BGGremlin-Group/Vocoder](https://github.com/BGGremlin-Group/Vocoder), and we need you to verify their integration. 

Please report issues, performance observations, and suggestions via [GitHub Issues](https://github.com/BGGremlin-Group/Vocoder/issues).


## 📋 Overview

BGGG_Vocoder V5.0 features:

- **Retro UI**: Dark theme (`#1A1A1A`), neon cyan text (`#00FFFF`), metallic borders, "Press Start 2P" font, and custom PNG splash screens.

- **Voice Cloning**: Train voice models from MP3, MP4, or WAV files using `TTS` (Tacotron 2 + HiFi-GAN) and `speechbrain` for personalized audio.

- **Vocoder Effects**: Phase, LPC, and Channel vocoders with pitch shift (-12 to +12 semitones) and modulation (0.1–10 Hz).

- **Avatar Creation**: Facial animation (selfie/webcam), full-body animation (OBJ models), and 3D particle effects.

- **TTS with Avatar**: Real-time TTS with phoneme-aligned lip-sync for facial avatars.
- **Integration**: Seamless UI, audio processing, visualization, and web UI (`http://localhost:5000`) interaction.

- **Other Features**: Noise-robust phoneme detection (`webrtcvad`), seven visualization modes, batch processing, and performance optimizations.


## 🛠️ Setup Instructions

### System Requirements
- **OS**: Windows 11
- **Hardware**: Microphone, speakers, webcam (for avatar creation), ~500 MB free disk space
- **Python**: 3.8 or higher
- **GPU (Optional)**: CUDA-capable GPU for faster voice cloning and processing (CPU fallback available)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/BGGremlin-Group/Vocoder.git
   cd Vocoder
   ```
2. **Run the Application**:
   ```bash
   python BGGG_Vocoder_v5.py
   ```
   - Automatically:
     - Creates directories: `~/BGGG_Vocoder/{assets,config,recordings,voice_models,presets,themes,docs}`

     - Installs dependencies (e.g., `pyaudio==0.2.14`, `torch==2.4.1+cpu`, `TTS==0.22.0`, `speechbrain==1.0.1`, `pydub==0.25.1` for MP3/MP4 support)

     - Downloads `shape_predictor_68_face_landmarks.dat` for facial animation

     - Loads PNG assets from `assets` (`logo_screen.png`, `credits_screen.png`, `exit_screen.png`, `static_visualizer.png`)

   - If dependency installation fails, manually run:
     ```bash
     pip install pyaudio==0.2.14 numpy==1.26.4 scipy==1.14.1 pydub==0.25.1 PyQt5==5.15.11 requests==2.32.3 sounddevice==0.5.0 matplotlib==3.9.2 librosa==0.10.2 opencv-python==4.10.0 dlib==19.24.6 moviepy==1.0.3 torch==2.4.1+cpu torchaudio==2.4.1+cpu TTS==0.22.0 speechbrain==1.0.1 g2p_en==2.1.0 vispy==0.14.3 numba==0.60.0 soundfile==0.12.1 psutil==6.0.0 flask==3.0.3 trimesh==4.2.3 webrtcvad==2.0.10
     ```

3. **Verify PNG Assets**:
   - Check `assets` for `logo_screen.png`, `credits_screen.png`, `exit_screen.png`, and `static_visualizer.png`.

   - Ensure each is **800x600 pixels**, valid PNG (RGB/RGBA), and non-corrupted (open in an image viewer).

   - Placeholders (“Missing [filename]”) appear if assets are missing or invalid.

### Testing Setup Notes
- **Disk Space**: Ensure ~500 MB free at `~/BGGG_Vocoder`.

- **Permissions**: Grant microphone and webcam access in Windows settings.

- **Font**: Uses "Press Start 2P" from `assets/PressStart2P-Regular.ttf` (falls back to system font).

- **Internet**: Required for initial dependency installation and dlib shape predictor download.

- **Audio Files**: Prepare MP3 and MP4 files for voice cloning (clear audio, 10+ seconds).


## 🚀 Usage Guide (Focused Features)

### 1. Interface and Integration
- **Launch**: Displays `logo_screen.png` (3s), then `credits_screen.png` (3s). Press any key to skip. On exit, `exit_screen.png` shows for 3s.


- **UI Layout**: Winamp-style with:
  - **Control Panel**: Play, Stop, Record, Save buttons.

  - **Equalizer**: 8 bars updating with audio input.
  - **Visualization Canvas**: Shows `static_visualizer.png` or dynamic visualizations.

  - **TTS Input**: Text field for live TTS.
  - **Status Bar**: Displays status, latency, volume (red for errors, cyan otherwise).

  - **Tabs**: Main Settings, Advanced Settings, Performance.

- **Web UI**: Access at `http://localhost:5000` to control algorithm, visualization mode, volume, and modulation rate.

- **Integration**: Settings sync between desktop and web UI. Audio, visualizations, and animations update in real-time.


### 2. Voice Cloning (Including MP3/MP4)
- **Training**:
  - **From Microphone**: Record 10+ seconds via `Main Settings > Record`.

  - **From File**: Use `Batch Process` to import MP3, MP4, or WAV files (processed via `pydub` to extract audio).

  - Train via `VocoderProcessor.train_voice` (saves to `~/BGGG_Vocoder/voice_models`).

  - Adjust `learning_rate` (1e-5 to 1e-3) and `batch_size` (4–32) in Advanced Settings.

- **Usage**: Enable “Use Cloned Voice” and select a model (`speaker_id`). Apply to TTS or audio processing.

- **Prosody**: Adjust `prosody_strength` (0.1 default) for pitch/energy alignment.

- **MP3/MP4 Handling**: Code uses `pydub` to convert MP3/MP4 to WAV internally for training (`train_voice` supports `AudioSegment.from_file`).


### 3. Vocoder Effects

- **Algorithms** (Main Settings > Vocoder Algorithm):
  - **Phase**: Natural, phase-based modulation.

  - **LPC**: Robotic, linear predictive coding.

  - **Channel**: Frequency-modulated, random gain per band.

- **Settings** (Advanced Settings):
  - **Pitch Shift**: -12 to +12 semitones.
  - **Modulation Index**: 0–1 (intensity).
  - **Modulation Rate**: 0.1–10 Hz (vibrato speed).
- **Application**: Real-time via `Play` or batch processing via `Batch Process`.


### 4. Avatar Creation
- **Facial Animation**:
  - Upload selfie (PNG/JPG) via `Upload Selfie` or enable webcam (`Use Webcam`).

  - Enable `Animate Face` and select `Animation Mode: Face Only`.

  - Uses `dlib` and `shape_predictor_68_face_landmarks.dat` for lip-sync.


- **Full-Body Animation**:
  - Select `Animation Mode: Full Body`.
  - Load OBJ file from `assets` (falls back to box model if invalid).
  - Animates based on audio RMS and pitch.

- **Particle Animation**:
  - Select `Animation Mode: Particles`.
  - Adjust `particle_count` (100–5000) and `particle_size` (0.2–2.0).
  - Audio-reactive particles in Vispy window.


### 5. TTS with Avatar
- **TTS Input**: Type text in `tts_input` field.
- **Voice**: Use default (`tts_models/en/ljspeech/hifigan_v2`) or cloned voice.
- **Lip-Sync**: Enable `Animate Face` and `Face Only`. Phonemes (`g2p_en`) drive lip movements.
- **Output**: Real-time audio with synced facial animation. Save as MP4 for video output.


## 🔍 Testing Focus Areas

### 1. Voice Cloning (Including MP3/MP4)
- **Training from Files**:

  - Import MP3 and MP4 files via `Batch Process`. Test with:
    - Clear audio (10–30s, speech).

    - Noisy audio (background music, chatter).

    - Short audio (<10s) to trigger error (`ValueError: Audio sample too short`).

    - Video-heavy MP4s (ensure audio extraction via `pydub`).

  - Verify models save to `~/BGGG_Vocoder/voice_models` and load correctly.


- **Quality**:
  - Use cloned voice for TTS and audio processing. Evaluate naturalness, clarity, and prosody.

  - Compare cloning from MP3, MP4, WAV, and microphone input.

  - Adjust `prosody_strength` (0.1–0.5) and test pitch/energy alignment.


- **Performance**:
  - Measure training time for MP3/MP4 (CPU vs. GPU).

  - Test with large files (e.g., 1-minute MP4) for memory usage.

  - Verify `pydub` handles various codecs (e.g., AAC, MP3).


### 2. Vocoder Effects

- **Algorithm Quality**:
  - Test Phase, LPC, and Channel vocoders with speech, music, and cloned voices from MP3/MP4.
  - Compare naturalness, robotic effects, and frequency modulation.

- **Settings**:
  - Push pitch (±12 semitones) and modulation (index 1, rate 10 Hz). Check for artifacts.
- **Real-Time vs. Batch**:
  - Test real-time processing (`Play`) for latency.
  - Batch process MP3/MP4 files and verify output quality.


### 3. Interface and Integration
- **PNG Assets**:
  - Verify `logo_screen.png`, `credits_screen.png`, `exit_screen.png` (3s each, skippable) and `static_visualizer.png` (visualization background).
  - Ensure 800x600 resolution and retro UI compatibility (dark background, neon accents).
  - Check `vocoder.log` for image loading errors.

- **UI Usability**:
  - Test navigation: buttons, tabs, sliders, dropdowns.
  - Enable `Simplified Mode` to hide tabs. Verify basic controls remain.
  - Confirm equalizer updates with audio and status bar shows latency/errors.

- **Web UI**:
  - Access `http://localhost:5000`. Test algorithm, visualization, volume, and modulation changes.
  - Verify settings sync with desktop UI.

- **Integration**:
  - Test seamless interaction between voice cloning, vocoder, TTS, and avatar animation.
  - Ensure MP3/MP4 file imports integrate with all features.


### 4. Avatar Creation
- **Facial Animation**:
  - Test with selfie and webcam in varied lighting. Verify face detection and lip-sync accuracy.
  - Check with multiple faces or no face detected.

- **Full-Body Animation**:
  - Load complex/invalid OBJ files. Verify fallback to box model.
  - Test animation responsiveness to audio from MP3/MP4.

- **Particle Animation**:
  - Test with `particle_count` (100, 1000, 5000) and `particle_size` (0.2, 1, 2).
  - Verify audio reactivity in Vispy window.


### 5. TTS with Avatar
- **TTS Quality**:
  - Input short/long texts (e.g., “Hello”, “This is a test for TTS”). Verify audio clarity.
  - Test with cloned voices from MP3/MP4 vs. default voice.

- **Lip-Sync**:
  - Enable `Animate Face` and `Face Only`. Check lip movements (vowels open, consonants close).
  - Test in noisy environments for phoneme detection (`webrtcvad`).

- **Integration**:
  - Verify TTS audio syncs with facial animation in real-time.
  - Save as MP4 and check video/audio alignment.


## 🔧 Known Limitations and Testing Focus

1. **Voice Cloning (MP3/MP4)**:
   - **Issue**: Requires 10+ seconds of clear audio. Noisy or short files degrade quality or raise errors.
   - **Test**: Use varied MP3/MP4 files (clear, noisy, short, video-heavy). Report model quality, training time, and errors.

2. **PNG Assets**:
   - **Issue**: Incorrect resolution, corrupted, or misnamed PNGs trigger placeholders.
   - **Test**: Confirm PNGs display correctly. Check `vocoder.log` for errors (e.g., `PIL.UnidentifiedImageError`).

3. **Vocoder Effects**:
   - **Issue**: Extreme pitch/modulation may cause artifacts.
   - **Test**: Test extremes (pitch ±12, modulation rate 10 Hz) with MP3/MP4 inputs.

4. **Avatar Creation**:
   - **Issue**: Facial animation fails in poor lighting or with no face. Complex OBJs slow performance.
   - **Test**: Use varied lighting, multiple faces, and large/invalid OBJs. Report detection and performance.

5. **TTS with Avatar**:
   - **Issue**: Phoneme detection may fail in noisy environments, affecting lip-sync.
   - **Test**: Test TTS in noisy settings with MP3/MP4 cloned voices. Verify phoneme accuracy.

6. **Interface/Integration**:
   - **Issue**: Web UI fails if port 5000 is in use. High CPU load lags UI.
   - **Test**: Launch Web UI with port conflicts. Test responsiveness with complex features.

7. **Performance**:
   - **Issue**: Cloning from large MP4s or 3D animations is CPU-intensive.
   - **Test**: Enable `Low Performance Mode`, adjust `vis_update_rate` (10–60 Hz), and report CPU/GPU usage.


## 🐞🔫 Troubleshooting

- **Voice Cloning (MP3/MP4)**:
  - Ensure MP3/MP4 files have clear audio (10+ seconds). Check `pydub` compatibility with codecs (install `ffmpeg` if needed):
    ```bash
    winget install ffmpeg
    ```
  - Verify models in `~/BGGG_Vocoder/voice_models`.
  - Adjust `learning_rate` and `batch_size`.

- **PNG Issues**:
  - Check `assets` for `logo_screen.png`, `credits_screen.png`, `exit_screen.png`, `static_visualizer.png` (800x600, valid PNG).
  - Resize if needed:
    ```bash
    convert assets/logo_screen.png -resize 800x600 assets/logo_screen.png
    ```

- **Vocoder Issues**:
  - Select correct audio devices.
  - Test lower pitch/modulation to avoid artifacts.

- **Avatar Creation**:
  - Ensure good lighting and single face for webcam/selfie.
  - Use simple OBJ files on low-end systems.
  - Verify `shape_predictor_68_face_landmarks.dat`.

- **TTS with Avatar**:
  - Test phoneme detection in quiet/noisy settings.
  - Check `g2p_en` and `webrtcvad` in `vocoder.log`.

- **Interface/Integration**:
  - Check port 5000:
    ```bash
    netstat -ano | findstr :5000
    ```
  - Kill conflicting processes or modify `launch_web_ui` for another port.
  - Enable `Low Performance Mode` for UI lag.

- **General**:
  - Check `vocoder.log` for errors.
  - Ensure 500 MB free at `~/BGGG_Vocoder`.
  - Manually install dependencies if needed.


## 📝 Feedback Guidelines

Provide detailed feedback on:
- **Voice Cloning (MP3/MP4)**: Training success, model quality, prosody, file format handling.
- **Vocoder Effects**: Audio quality, artifacts, real-time/batch performance.
- **Interface/Integration**: PNG display, UI usability, web UI sync, responsiveness.
- **Avatar Creation**: Facial detection, lip-sync, OBJ rendering, particle animation.
- **TTS with Avatar**: Audio clarity, phoneme accuracy, lip-sync quality.
- **Bugs**: Steps to reproduce, `vocoder.log`, screenshots/recordings.
- **Suggestions**: UI improvements, feature enhancements.

Submit via [GitHub Issues](https://github.com/BGGremlin-Group/Vocoder/issues) with:
- System specs (OS, CPU, GPU, RAM)
- Python version
- Steps to reproduce
- Screenshots/recordings


## 🎯 Test Scenarios

1. **Voice Cloning (MP3/MP4)**:
   - Import MP3/MP4 files (10s, 30s, noisy, video-heavy) via `Batch Process`. Train and test with TTS/processing.
   - Test short files (<10s) for error handling.
   - Compare cloning quality across MP3, MP4, WAV, and microphone.

2. **Vocoder Effects**:
   - Test Phase, LPC, and Channel with MP3/MP4 inputs (speech, music).
   - Push pitch (±12) and modulation (rate 10 Hz). Check real-time and batch outputs.

3. **Interface/Integration**:
   - Verify PNGs (`logo_screen.png`, etc.) display correctly.
   - Test UI navigation, simplified mode, and web UI sync (`http://localhost:5000`).
   - Check responsiveness with MP3/MP4 cloning and animations.

4. **Avatar Creation**:
   - Test facial animation with selfie/webcam (varied lighting, multiple faces).
   - Load complex/invalid OBJs for full-body animation.
   - Test particle animation with max/min `particle_count` and `particle_size`.

5. **TTS with Avatar**:
   - Input varied texts with cloned voices from MP3/MP4. Verify audio and lip-sync.
   - Test in noisy environments. Save as MP4 and check video/audio sync.

6. **Stress Testing**:
   - Process large MP4s, complex OBJs, and high `vis_update_rate`.
   - Test on low-end systems with `Low Performance Mode`.


## 🫡 Thank You! 

Your testing of voice cloning from MP3/MP4, vocoder effects, interface, avatar creation, and TTS with avatar will make BGGG_Vocoder V5.0 a powerful tool. Thank you for your time and effort! 🚀

**The Background Gremlin Group**  
*Creating Unique Tools for Unique Individuals*

---

### Updates from Previous Notes

- **Voice Cloning from MP3/MP4**: Added specific guidance for cloning from MP3/MP4 files, emphasizing `pydub` usage, testing with varied files, and verifying codec compatibility (e.g., `ffmpeg` for AAC/MP3).

- **Focus Areas**: Prioritized voice cloning (MP3/MP4), vocoder, interface/integration, avatar creation, and TTS with avatar, as requested.

- **Test Scenarios**: Included MP3/MP4-specific tests (clear, noisy, short, video-heavy files) and integration with other features.

- **Troubleshooting**: Added steps for MP3/MP4 issues (e.g., `ffmpeg` installation) and verified `pydub` handling.

- **PNG Assets**: Reconfirmed verification of uploaded PNGs (800x600, valid PNG, in `assets`).

- **Repository**: [https://github.com/BGGremlin-Group/Vocoder](https://github.com/BGGremlin-Group/Vocoder) contains assets
