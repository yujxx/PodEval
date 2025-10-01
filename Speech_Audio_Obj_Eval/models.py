import torch
import whisper
from demucs.pretrained import get_model
from pyannote.audio import Model, Pipeline, Inference
from casp.wrapper import CASPWrapper
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import gc

_demucs_model = None
_whisper_model = None
_vad_model = None
_vad_utils = None
_diarization_pipeline = None
_embedding_inference = None
_casp_model = None
_primary_onnx_session = None
_p808nnx_session = None

def get_demucs_model():
    global _demucs_model
    if _demucs_model is None:  # 第一次调用时才加载
        print("\tLoading Demucs model...")
        _demucs_model = get_model(name="htdemucs")
    return _demucs_model

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("\tLoading Whisper model...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def get_vad_model():
    global _vad_model, _vad_utils
    if _vad_model is None:
        print("\tLoading VAD model...")
        _vad_model, _vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    return _vad_model, _vad_utils

def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        print("\tLoading the speaker diarization model...")
        _diarization_pipeline = Pipeline.from_pretrained(
           "pyannote/speaker-diarization-3.0", 
            use_auth_token="hf_xxx" # Please replace with your own Hugging Face token
        )
        _diarization_pipeline.to(torch.device("cuda"))
    return _diarization_pipeline

def get_embedding_inference():
    global _embedding_inference
    if _embedding_inference is None:
        print("\tLoading the Speaker Embedding model...")
        model = Model.from_pretrained(
         "pyannote/embedding", 
            use_auth_token="hf_xxx" # Please replace with your own Hugging Face token
        )
        _embedding_inference = Inference(model, window="whole")
        _embedding_inference.to(torch.device("cuda"))
    return _embedding_inference

def get_casp_model(model_path):
    global _casp_model
    if _casp_model is None:
        print("\tLoading CASP model...")
        ckpt_path = hf_hub_download(repo_id="wonderfuluuuuuuuuuuu/DualDub", filename="podcast-10s.ckpt")
        pretrain_path = f"{model_path}/BEATs_iter3_plus_AS2M.pt"        
        _casp_model = CASPWrapper(d_model=768, ckpt_path=pretrain_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(ckpt_path, map_location=device)
        _casp_model.load_state_dict(checkpoint['state_dict'])
        _casp_model.to(device)
        _casp_model.eval()
    return _casp_model

def get_onnx_sessions(primary_model_path, p808_model_path):
    global _primary_onnx_session, _p808nnx_session
    if _primary_onnx_session is None or _p808nnx_session is None:
        _primary_onnx_session = ort.InferenceSession(primary_model_path)
        _p808nnx_session = ort.InferenceSession(p808_model_path)
    return _primary_onnx_session, _p808nnx_session

# Model release functions
def release_demucs_model():
    """Release Demucs model from memory"""
    global _demucs_model
    if _demucs_model is not None:
        print("\tReleasing Demucs model from memory...")
        del _demucs_model
        _demucs_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def release_whisper_model():
    """Release Whisper model from memory"""
    global _whisper_model
    if _whisper_model is not None:
        print("\tReleasing Whisper model from memory...")
        del _whisper_model
        _whisper_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def release_vad_model():
    """Release VAD model from memory"""
    global _vad_model, _vad_utils
    if _vad_model is not None:
        print("\tReleasing VAD model from memory...")
        del _vad_model, _vad_utils
        _vad_model = None
        _vad_utils = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def release_diarization_pipeline():
    """Release speaker diarization pipeline from memory"""
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        print("\tReleasing speaker diarization pipeline from memory...")
        del _diarization_pipeline
        _diarization_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def release_embedding_inference():
    """Release speaker embedding inference from memory"""
    global _embedding_inference
    if _embedding_inference is not None:
        print("\tReleasing speaker embedding inference from memory...")
        del _embedding_inference
        _embedding_inference = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def release_casp_model():
    """Release CASP model from memory"""
    global _casp_model
    if _casp_model is not None:
        print("\tReleasing CASP model from memory...")
        del _casp_model
        _casp_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def release_onnx_sessions():
    """Release ONNX sessions from memory"""
    global _primary_onnx_session, _p808nnx_session
    if _primary_onnx_session is not None or _p808nnx_session is not None:
        print("\tReleasing ONNX sessions from memory...")
        if _primary_onnx_session is not None:
            del _primary_onnx_session
            _primary_onnx_session = None
        if _p808nnx_session is not None:
            del _p808nnx_session
            _p808nnx_session = None
        gc.collect()

def release_all_models():
    """Release all models from memory"""
    print("Releasing all models from memory...")
    release_demucs_model()
    release_whisper_model()
    release_vad_model()
    release_diarization_pipeline()
    release_embedding_inference()
    release_casp_model()
    release_onnx_sessions()
    print("All models released from memory.")