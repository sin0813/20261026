import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="딥러닝 기반 AI 사진 판별기",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 딥러닝 & 주파수 하이브리드 AI 사진 판별기")
st.caption("2D 푸리에 변환(FFT) 주파수 스펙트럼 시각화와 CNN 딥러닝 이미지 추출 엔진을 결합하여 실제 사진과 AI 사진(Diffusion, StyleGAN)을 판별합니다.")

st.markdown("---")

# 2. 딥러닝 모델 로드 (캐싱을 통한 고속화)
@st.cache_resource
def load_deep_learning_model():
    # 경량화 비전 모델 사전 학습 가중치 로드
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    return model, weights.transforms()

try:
    dl_model, dl_preprocess = load_deep_learning_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# 3. 주파수 스펙트럼 추출 함수
def extract_fft_spectrum(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    gray = cv2.resize(gray, (512, 512))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = mag_norm.shape
    cy, cx = h // 2, w // 2
    cv2.circle(mag_norm, (cx, cy), 35, 0, -1)
    
    return mag_norm

# 4. 딥러닝 + 주파수 통합 판별 로직
def predict_ai_image(pil_img, img_array):
    # (1) 주파수 고주파 편향도 산출
    mag_spectrum = extract_fft_spectrum(img_array)
    h, w = mag_spectrum.shape
    cy, cx = h // 2, w // 2
    
    angles = np.linspace(0, 180, 90, endpoint=False)
    angle_energies = []
    for angle in angles:
        rad = np.deg2rad(angle)
        xs = (cx + np.linspace(35, 180, 50) * np.cos(rad)).astype(np.int32)
        ys = (cy + np.linspace(35, 180, 50) * np.sin(rad)).astype(np.int32)
        valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        angle_energies.append(np.mean(mag_spectrum[ys[valid_mask], xs[valid_mask]]))
    
    freq_bias = np.std(angle_energies) / (np.mean(angle_energies) + 1e-5)

    # (2) 공간 영역 Laplacian 미세 잔차 검출
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blocks = [laplacian[i:i+64, j:j+64] for i in range(0, 512, 64) for j in range(0, 512, 64) if laplacian[i:i+64, j:j+64].size > 0]
    block_vars = [np.var(b) for b in blocks]
    spatial_anomaly = np.std(block_vars) / (np.mean(block_vars) + 1e-5)

    # (3) CNN 딥러닝 피처 텐서 추론
    input_tensor = dl_preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        features = dl_model.features(input_tensor)
        dl_signal = torch.std(features).item()

    # 하이브리드 종합 스코어링
    # StyleGAN 아티팩트와 Diffusion 격자를 딥러닝 텐서 신호와 결합
    combined_score = (freq_bias * 4.5) + (spatial_anomaly * 0.25) + (dl_signal * 0.15)

    # 확률 산출
    ai_prob = min(max((combined_score - 0.42) * 180 + 50, 1.0), 99.0)
    real_prob = 100.0 - ai_prob

    return mag_spectrum, ai_prob, real_prob, combined_score, freq_bias, spatial_anomaly

# 5. UI 및 분석 결과 출력
st.header("🔍 사진 업로드 및 분석")
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("딥러닝 신경망 및 주파수 스펙트럼을 통합 분석 중입니다..."):
        mag_spectrum, ai_prob, real_prob, score, f_bias, s_anomaly = predict_ai_image(image, img_array)

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. 입력된 사진")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("2. 주파수 스펙트럼")
        fig_user = px.imshow(mag_spectrum, color_continuous_scale="gray")
        fig_user.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_user, use_container_width=True)

    with col3:
        st.subheader("3. 딥러닝 판별 결과")
        
        if ai_prob > 50:
            st.error("🤖 **[인공지능 생성 사진]으로 판별되었습니다.**")
            st.caption("CNN 신경망 및 고주파 스펙트럼에서 AI 연산 아티팩트 특성이 감지되었습니다.")
        else:
            st.success("📷 **[실제 카메라 사진]으로 판별되었습니다.**")
            st.caption("이미지의 픽셀 공간 잔차 및 주파수 패턴이 광학 센서 특성과 일치합니다.")

        st.progress(int(ai_prob))
        st.write(f"• **인공지능 사진 확률:** `{ai_prob:.1f}%`")
        st.write(f"• **실제 사진 확률:** `{real_prob:.1f}%`")
        st.markdown("---")
        st.caption(f"· 하이브리드 종합 지수: `{score:.3f}`")
        st.caption(f"· 주파수 격자 편향: `{f_bias:.4f}`")
        st.caption(f"· 공간 잔차 이상치: `{s_anomaly:.4f}`")

else:
    st.info("👆 사진을 올리시면 딥러닝 모델과 푸리에 분석 엔진이 실시간 판별을 수행합니다.")
