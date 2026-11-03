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
st.caption("2D 푸리에 변환(FFT) + 공간 픽셀 잔차 분석(Spatial Residuals) 하이브리드 엔진으로 AI 이미지를 정밀 판별합니다.")

st.markdown("---")

# 3. 원리 설명
st.header("💡 사진이 구별되는 이유와 분석 원리")
st.write("Diffusion 계열의 **격자 잔상**과 StyleGAN 계열의 **비자연적 노이즈 평탄화** 현상을 두 가지 수학적 알고리즘으로 동시 검출합니다.")

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
    - **원인:** 광학 렌즈의 빛 산란 및 CCD/CMOS 센서 고유의 **무작위 불규칙 노이즈**
    - **특징:** 주파수 스펙트럼이 부드러운 원형을 이루며, 공간 노이즈의 국소 분산(Variance) 수치가 불규칙적입니다.
    """)
    real_sample = generate_sample_spectrum("real")
    fig_real = px.imshow(real_sample, color_continuous_scale="gray", title="[예시] 실제 사진 스펙트럼")
    fig_real.update_layout(coloraxis_showscale=False, height=250, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_real, use_container_width=True)

with col_exp2:
    st.subheader("🤖 인공지능 생성 사진")
    st.markdown("""
    - **원인:** Diffusion 연산의 **격자 패턴** 또는 GAN 연산의 **과도한 픽셀 평탄화/아티팩트**
    - **특징:** 주파수 방향성 편향이 심하거나, 픽셀 세부 잔차(Residual) 분석 시 생체 노이즈 특성이 결여되어 있습니다.
    """)
    ai_sample = generate_sample_spectrum("ai")
    fig_ai = px.imshow(ai_sample, color_continuous_scale="gray", title="[예시] AI 사진 스펙트럼")
    fig_ai.update_layout(coloraxis_showscale=False, height=250, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_ai, use_container_width=True)

st.markdown("---")

# 4. 하이브리드 감지 함수 (FFT + Spatial Residuals)
def analyze_hybrid_detection(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    gray = cv2.resize(gray, (512, 512))

    # [지표 1] 주파수 영역 (FFT) 360도 스캔 - Diffusion 격자 감지
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

    # [지표 2] 공간 영역 (Spatial Domain) Laplacian 잔차 분석 - StyleGAN 매끄러움/아티팩트 감지
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    spatial_var = np.var(laplacian)
    
    # 영역별 노이즈 균일성(Non-uniformity) 검출
    blocks = [laplacian[i:i+64, j:j+64] for i in range(0, 512, 64) for j in range(0, 512, 64)]
    block_vars = [np.var(b) for b in blocks]
    spatial_anomaly_score = np.std(block_vars) / (np.mean(block_vars) + 1e-5)

    # 통합 복합 스코어 연산
    # StyleGAN은 주파수 편향은 작으나 공간 노이즈 이상치(spatial_anomaly_score)가 높거나 spatial_var가 부자연스러움
    total_ai_score = (freq_bias_score * 8.0) + (spatial_anomaly_score * 0.45)

    # 기준선 (0.85 이상 시 AI 확률 증가)
    ai_prob = min(max((total_ai_score - 0.85) * 150 + 50, 1.0), 99.0)
    real_prob = 100.0 - ai_prob

    display_map = mag_norm.astype(np.uint8).copy()
    cv2.circle(display_map, (cx, cy), 35, 0, -1)

    return display_map, ai_prob, real_prob, total_ai_score, freq_bias_score, spatial_anomaly_score

# 5. UI 및 결과
st.header("🔍 사진 업로드 및 분석")
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    mag_spectrum, ai_prob, real_prob, total_score, f_score, s_score = analyze_hybrid_detection(img_array)

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
        st.subheader("3. 하이브리드 판별 결과")
        
        if ai_prob > 50:
            st.error("🤖 **[인공지능 생성 사진]으로 구분되었습니다.**")
            st.caption("주파수 격자 편향 또는 공간 픽셀 잔차(Laplacian) 이상치가 감지되었습니다.")
        else:
            st.success("📷 **[실제 카메라 사진]으로 구분되었습니다.**")
            st.caption("주파수 및 공간 영역 노이즈 분포가 자연스러운 광학 특성을 보입니다.")

        st.progress(int(ai_prob))
        st.write(f"• **인공지능 사진 확률:** `{ai_prob:.1f}%`")
        st.write(f"• **실제 사진 확률:** `{real_prob:.1f}%`")
        st.markdown("---")
        st.caption(f"· 통합 검출 지수: `{total_score:.3f}`")
        st.caption(f"· (1) 주파수 격자 편향 지수: `{f_score:.4f}`")
        st.caption(f"· (2) 픽셀 공간 잔차 지수: `{s_score:.4f}`")

else:
    st.info("👆 사진을 올리면 하이브리드 엔진이 주파수와 픽셀 노이즈 잔차를 동시에 측정합니다.")
