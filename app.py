import os
import io
import time
import tempfile
import numpy as np
import tensorflow as tf
import shutil

import os, subprocess, tempfile, numpy as np, joblib
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import Model

import streamlit as st

globals()["tf"] = tf

# ---- Optional heavy deps for feature extraction ----
# We use OpenCV for frames and librosa for audio.
import cv2
import librosa

st.set_page_config(page_title="Video Similarity Demo", layout="wide")
st.title("🎞️ Video Encoder Demo — Upload, Embed, Compare")

@st.cache_resource(show_spinner=True)
def load_encoder(model_dir: str = "video_encoder"):
    # model_dir should be the folder you saved via video_encoder.save("video_encoder")
    model = K.models.load_model(model_dir, safe_mode=False)

    for lyr in model.layers:
        if isinstance(lyr, K.layers.Lambda):
            fn = getattr(lyr, "function", None)  # Keras stores the python callable here
            if fn is not None and "tf" not in fn.__globals__:
                fn.__globals__["tf"] = tf

    return model

try:
    encoder = load_encoder("model_save1.1.keras")
except Exception as e:
    st.error(
        "Could not load the encoder from folder.\n"
    )
    st.exception(e)
    st.stop()

def make_orthoprojector_2048_to_1024(seed=123):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(2048, 2048)).astype(np.float32)
    Q, _ = np.linalg.qr(G)              # Q: 2048x2048, orthonormal columns
    P = Q[:, :1024].astype(np.float32)  # 2048x1024
    print(P)
    return P

EMB_DIM = encoder.output_shape[-1]  # should be 256 per your build
RGB_DIM = 1024
AUD_DIM = 128

PROJ_PATH = "projection/proj_2048_to_1024.joblib"
if os.path.exists(PROJ_PATH):
    P_RGB = joblib.load(PROJ_PATH)
else:
    P_RGB = make_orthoprojector_2048_to_1024(seed=123)
    joblib.dump(P_RGB, PROJ_PATH)

_base = InceptionV3(weights="imagenet", include_top=False, input_shape=(299,299,3))
gap = tf.keras.layers.GlobalAveragePooling2D()(_base.output)
inception_rgb = Model(_base.input, gap)
inception_rgb.trainable = False

    
def read_frames_opencv(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(src_fps / fps)), 1)

    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise ValueError("No frames decoded — try lowering fps or check the video file.")
    return frames

def rgb_feature_inception(video_path, sample_fps=1, batch_size=32):
    frames = read_frames_opencv(video_path, fps=sample_fps)
    x = np.stack([cv2.resize(f, (299,299), interpolation=cv2.INTER_AREA).astype(np.float32) for f in frames], axis=0)
    x = preprocess_input(x)
    feats = []
    for i in range(0, len(x), batch_size):
        feats.append(inception_rgb(x[i:i+batch_size]).numpy())  # [b,2048]
    feats = np.concatenate(feats, axis=0)                       # [T,2048]
    return feats.mean(axis=0).astype(np.float32)  

def rgb_1024_from_inception(rgb_2048, P_RGB):
    x = rgb_2048.astype(np.float32)
    x = x / (np.linalg.norm(x) + 1e-12)
    x1024 = x @ P_RGB                           # [2048] x [2048x1024] -> [1024]
    x1024 = x1024 / (np.linalg.norm(x1024) + 1e-12)
    return x1024.astype(np.float32)

def extract_audio_wav(input_video_path, target_sr=16000):
    # Find an ffmpeg executable: system one if present, else imageio-ffmpeg’s bundled one
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise RuntimeError(
                "FFmpeg not found. On Streamlit Cloud either add 'ffmpeg' to packages.txt "
                "or keep 'imageio-ffmpeg' in requirements and ensure this fallback is used."
            ) from e

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        cmd = [ffmpeg_path, "-y", "-i", input_video_path, "-ac", "1", "-ar", str(target_sr), wav_path]
        # Capture output quietly; raise on failure
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0 or not os.path.exists(wav_path):
            raise RuntimeError(f"ffmpeg failed to extract audio: {result.stderr.decode(errors='ignore')[:500]}")

        y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        return y, sr
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

'''def extract_audio_wav(input_video_path, target_sr=16000):
    wav_path = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", input_video_path, "-ac", "1", "-ar", str(target_sr), wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    os.remove(wav_path)
    return y, sr'''

def logmel_feature(y, sr, n_mels=64, hop_length=160, n_fft=400):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=125, fmax=7500
    )
    logS = librosa.power_to_db(S, ref=np.max).T  # [T,64]
    return logS

def audio_feature_128(video_path):
    y, sr = extract_audio_wav(video_path, target_sr=16000)
    logm = logmel_feature(y, sr, n_mels=64)    # [T,64]
    mu, sd = logm.mean(axis=0), logm.std(axis=0)
    feat_128 = np.concatenate([mu, sd], axis=0).astype(np.float32)  # [128]
    # L2 normalize for stability (optional)
    feat_128 = feat_128 / (np.linalg.norm(feat_128) + 1e-12)
    return feat_128

def build_feature_vector_1152(video_path, sample_fps=1):
    rgb_2048 = rgb_feature_inception(video_path, sample_fps)
    rgb_1024 = rgb_1024_from_inception(rgb_2048, P_RGB)   # [1024]
    aud_128  = audio_feature_128(video_path)              # [128]
    feat_1152 = np.concatenate([rgb_1024, aud_128], axis=0)  # [1152]
    return feat_1152

def embed_x1152(x1152: np.ndarray) -> np.ndarray:
    x = tf.convert_to_tensor(x1152[None, :], tf.float32)
    with tf.device("/CPU:0"):
        v = encoder(x, training=False).numpy()[0]  # [EMB_DIM], already L2-normalized
    return v

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # encoder outputs are L2-normalized, but keep robust
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)




if "clips" not in st.session_state:
    st.session_state.clips = []  # list of dicts: {name, bytes, emb, x1152}

def add_clip(name: str, data_bytes: bytes, emb: np.ndarray, x1152: np.ndarray):
    st.session_state.clips.append({
        "name": name,
        "bytes": data_bytes,
        "emb": emb.astype(np.float32),
        "x1152": x1152.astype(np.float32),
    })

# ---------------------------
# Sidebar: upload + list
# ---------------------------
st.sidebar.header("Upload")
video_file = st.sidebar.file_uploader(
    "Upload a short clip (mp4/mov)", type=["mp4", "mov", "m4v", "avi"]
)

if video_file is not None:
    st.sidebar.video(video_file)
    if st.sidebar.button("➕ Add & Embed"):
        # Persist to a temp file for OpenCV/librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("Extracting features and embedding..."):
            x1152 = build_feature_vector_1152(tmp_path)
            emb = embed_x1152(x1152)

        add_clip(video_file.name, video_file.getvalue(), emb, x1152)
        os.remove(tmp_path)
        st.sidebar.success(f"Added: {video_file.name}")

st.sidebar.markdown("---")
st.sidebar.write(f"Total stored clips: **{len(st.session_state.clips)}**")

# ---------------------------
# Main: gallery + similarity explorer
# ---------------------------
if len(st.session_state.clips) == 0:
    st.info("Upload at least two clips to compare similarity.")
else:
    # Show a simple gallery
    cols = st.columns(5)
    for i, clip in enumerate(st.session_state.clips):
        with cols[i % 5]:
            st.video(io.BytesIO(clip["bytes"]))
            st.caption(clip["name"])

if len(st.session_state.clips) >= 2:
    st.markdown("## 🔎 Compare by cosine similarity")
    
    names = [c["name"] for c in st.session_state.clips]
    anchor_name = st.selectbox("Choose an anchor clip", names, index=0)
    anchor = next(c for c in st.session_state.clips if c["name"] == anchor_name)
    cols = st.columns(3)
    with cols[1]:
        # Rank the rest
        ranked = []
        for c in st.session_state.clips:
            if c["name"] == anchor_name:
                continue
            sim = cosine(anchor["emb"], c["emb"])
            ranked.append((sim, c))
        ranked.sort(key=lambda x: x[0], reverse=True)

        st.write(f"**Anchor:** {anchor_name}")
        st.progress(100)

        for sim, c in ranked:
            with st.expander(f"{c['name']} — cosine {sim:.3f}", expanded=False):
                st.video(io.BytesIO(c["bytes"]))

    # Optional: export embeddings
    st.markdown("### ⬇️ Export")
    if st.button("Download embeddings (.npz)"):
        npz_bytes = io.BytesIO()
        np.savez_compressed(
            npz_bytes,
            names=np.array([c["name"] for c in st.session_state.clips], dtype=object),
            embeddings=np.stack([c["emb"] for c in st.session_state.clips], axis=0),
            x1152=np.stack([c["x1152"] for c in st.session_state.clips], axis=0),
        )
        st.download_button(
            "Save embeddings.npz",
            data=npz_bytes.getvalue(),
            file_name="embeddings.npz",
            mime="application/octet-stream",
        )

st.caption(
    "Note: The included feature extractor is an approximation. For best results, "
    "replace `compute_1152_features_from_video` with your true YouTube-8M 1152-D pipeline."
)