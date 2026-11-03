import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.express as px

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="인공지능 사진 구분",
    page_icon="🔍",
    layout="wide"
)

# 2. 메인 타이틀
st.title("인공지능 사진 구분")
st.caption("2D 푸리에 변환(FFT) + 공간 영역 잔차 편향성(Spatial Residual Bias) 멀티 스캐너 엔진입니다.")

st.markdown("---")

# 3. 원리 및 예시 설명
st.header("💡 사진이 구별되는 이유와 분석 원리")
st.write("Diffusion 알고리즘의 **고주파 격자 패턴**과 StyleGAN 알고리즘의 **배경-피사체 간 픽셀 노이즈 불균일성**을 수학적으로 자동 감지합니다.")

def generate_sample_spectrum(pattern_type="real"):
    size = 200
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    if pattern_type == "real":
        spectrum = np.exp(-R * 3) + np.random.normal(0, 0.04, (size, size))
    else:
        grid = (np.sin(X * 25) ** 2) * (np.cos(Y * 25) ** 2)
        spectrum = np.exp(-R * 2.5) + grid * 0.35 + np.random.normal(0, 0.02, (size, size))
    
    return np.clip(spectrum, 0, 1)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.subheader("📷 실제 카메라 사진")
    st.markdown("""
    - **원인:** 광학 렌즈의 빛 산란 및 CCD/CMOS 센서 고유의 **자연스러운 무작위 노이즈**
    - **특징:** 주파수 스펙트럼이 원형을 유지하며, 공간 영역 전반에 걸쳐 노이즈 분포가 자연스럽습니다.
    """)
    real_sample = generate_sample_spectrum("real")
    fig_real = px.imshow(real_sample, color_continuous_scale="gray", title="[예시] 실제 사진 스펙트럼")
    fig_real.update_layout(coloraxis_showscale=False, height=250, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_real, use_container_width=True)

with col_exp2:
    st.subheader("🤖 인공지능 생성 사진")
    st.markdown("""
    - **원인:** AI 연산의 **격자 잔상** 또는 피사체-배경 간의 **수학적 픽셀 비대칭 아티팩트**
    - **특징:** 주파수 스펙트럼 외곽 에너지 쏠림 또는 픽셀 공간 잔차의 구역별 왜곡 현상이 관찰됩니다.
    """)
    ai_sample = generate_sample_spectrum("ai")
    fig_ai = px.imshow(ai_sample, color_continuous_scale="gray", title="[예시] AI 사진 스펙트럼")
    fig_ai.update_layout(coloraxis_showscale=False, height=250, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_ai, use_container_width=True)

st.markdown("---")

# 4. 정밀 멀티 스캔 분석 알고리즘
def analyze_deep_detection(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    gray = cv2.resize(gray, (512, 512))

    # [지표 1] 주파수 영역 (FFT) 360도 스캔
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    h, w = mag_norm.shape
    cy, cx = h // 2, w // 2
    
    angles = np.linspace(0, 180, 90, endpoint=False)
    angle_energies = []
    for angle in angles:
        rad = np.deg2rad(angle)
        xs = (cx + np.linspace(35, 180, 50) * np.cos(rad)).astype(np.int32)
        ys = (cy + np.linspace(35, 180, 50) * np.sin(rad)).astype(np.int32)
        valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        angle_energies.append(np.mean(mag_norm[ys[valid_mask], xs[valid_mask]]))

    freq_bias_score = np.std(angle_energies) / (np.mean(angle_energies) + 1e-5)

    # [지표 2] 공간 잔차 비대칭성 (Spatial Residual Anomaly)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 중앙(얼굴) 영역과 외곽(배경) 영역의 노이즈 분산 비율
    center_region = laplacian[128:384, 128:384]
    border_mask = np.ones_like(laplacian, dtype=bool)
    border_mask[128:384, 128:384] = False
    
    center_var = np.var(center_region) + 1e-5
    border_var = np.var(laplacian[border_mask]) + 1e-5
    
    spatial_ratio = max(center_var / border_var, border_var / center_var)
    
    # [지표 3] 픽셀 고주파 구역별 불균일성
    blocks = [laplacian[i:i+64, j:j+64] for i in range(0, 512, 64) for j in range(0, 512, 64)]
    block_vars = [np.var(b) for b in blocks]
    block_std = np.std(block_vars) / (np.mean(block_vars) + 1e-5)

    # 종합 스코어 변환
    # StyleGAN 및 압축 AI 이미지 정밀 감지 기준 설정 (임계치 0.038)
    total_ai_score = (freq_bias_score * 5.0) + (spatial_ratio * 0.12) + (block_std * 0.2)

    ai_prob = min(max((total_ai_score - 0.48) * 220 + 50, 1.0), 99.0)
    real_prob = 100.0 - ai_prob

    display_map = mag_norm.astype(np.uint8).copy()
    cv2.circle(display_map, (cx, cy), 35, 0, -1)

    return display_map, ai_prob, real_prob, total_ai_score, freq_bias_score, spatial_ratio

# 5. UI 구성
st.header("🔍 사진 업로드 및 정밀 분석")
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    mag_spectrum, ai_prob, real_prob, total_score, f_score, s_ratio = analyze_deep_detection(img_array)

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. 입력된 사진")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("2. 주파수 스펙트럼 Map")
        fig_user = px.imshow(mag_spectrum, color_continuous_scale="gray")
        fig_user.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_user, use_container_width=True)

    with col3:
        st.subheader("3. 분석 및 판별 결과")
        
        if ai_prob > 50:
            st.error("🤖 **[인공지능 생성 사진]으로 구분되었습니다.**")
            st.caption("주파수 및 공간 잔차 비대칭성 분석 결과, AI 생성 고유의 픽셀 불균일성이 감지되었습니다.")
        else:
            st.success("📷 **[실제 카메라 사진]으로 구분되었습니다.**")
            st.caption("주파수 및 공간 영역의 노이즈가 자연스러운 광학 센서 특성을 나타냅니다.")

        st.progress(int(ai_prob))
        st.write(f"• **인공지능 사진 확률:** `{ai_prob:.1f}%`")
        st.write(f"• **실제 사진 확률:** `{real_prob:.1f}%`")
        st.markdown("---")
        st.caption(f"· 종합 분석 지수: `{total_score:.3f}`")
        st.caption(f"· 주파수 편향 지수: `{f_score:.4f}`")
        st.caption(f"· 공간 잔차 비율: `{s_ratio:.3f}`")

else:
    st.info("👆 사진을 올려주시면 주파수 스펙트럼과 공간 픽셀 비대칭성을 정밀 측정합니다.")
