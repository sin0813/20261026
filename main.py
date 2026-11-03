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
st.caption("2D 푸리에 변환(FFT) 기반 방사형 주파수 스펙트럼 분석으로 AI 이미지를 판별합니다.")

st.markdown("---")

# 3. 원리 및 예시
st.header("💡 사진이 구별되는 이유와 주파수 모양 예시")
st.write("픽셀 데이터를 주파수 영역으로 변환하여 360도 방사형 스캔을 수행하면 AI 알고리즘 고유의 미세 격자 잔상을 감지할 수 있습니다.")

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
    - **주파수 특징:** 중심에서 외곽으로 **360도 전 방향이 균일하게 동그란 원형**을 이룹니다.
    """)
    real_sample = generate_sample_spectrum("real")
    fig_real = px.imshow(real_sample, color_continuous_scale="gray", title="[예시] 실제 사진의 동그란 주파수 스펙트럼")
    fig_real.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_real, use_container_width=True)

with col_exp2:
    st.subheader("🤖 인공지능 생성 사진")
    st.markdown("""
    - **원인:** AI 업스케일링 및 컨볼루션 연산 과정에서 생기는 **수학적 미세 격자 잔상**
    - **주파수 특징:** 외곽 고주파 대역에 **특정 각도로 치우친 십자가(+) 및 점 패턴**이 남습니다.
    """)
    ai_sample = generate_sample_spectrum("ai")
    fig_ai = px.imshow(ai_sample, color_continuous_scale="gray", title="[예시] AI 사진의 규칙적인 격자 주파수 스펙트럼")
    fig_ai.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_ai, use_container_width=True)

st.markdown("---")

# 4. 균형 잡힌 주파수 스캔 알고리즘
def analyze_balanced_spectrum(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    gray = cv2.resize(gray, (512, 512))

    # 2D 푸리에 변환 (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    h, w = mag_norm.shape
    cy, cx = h // 2, w // 2
    
    # 360도 스캔
    angles = np.linspace(0, 180, 90, endpoint=False)
    angle_energies = []
    
    for angle in angles:
        rad = np.deg2rad(angle)
        xs = (cx + np.linspace(40, 180, 50) * np.cos(rad)).astype(np.int32)
        ys = (cy + np.linspace(40, 1
