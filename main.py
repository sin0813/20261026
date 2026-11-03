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
st.caption("빅데이터 기반 주파수 스펙트럼(FFT) 분석을 통해 실제 카메라 사진과 AI 생성 이미지를 판별합니다.")

st.markdown("---")

# 3. 사진이 구별되는 이유 및 예시 그래프
st.header("💡 사진이 구별되는 이유와 주파수 모양 예시")
st.write("사람의 눈으로는 구별하기 힘든 AI 사진도, 픽셀 데이터(빅데이터)를 주파수 영역으로 변환하면 형태 차이가 뚜렷하게 나타납니다.")

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
    - **원인:** 빛이 동그란 렌즈를 통과해 센서(CMOS)에 찍힐 때 발생하는 **자연스러운 센서 노이즈**
    - **주파수 특징:** 중심(저주파)에서 외곽(고주파)으로 **부드러운 동그란 원형**으로 퍼집니다.
    """)
    real_sample = generate_sample_spectrum("real")
    fig_real = px.imshow(real_sample, color_continuous_scale="gray", title="[예시] 실제 사진의 동그란 주파수 스펙트럼")
    fig_real.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_real, use_container_width=True)

with col_exp2:
    st.subheader("🤖 인공지능 생성 사진")
    st.markdown("""
    - **원인:** AI 생성 알고리즘(Diffusion 등)이 픽셀을 연산하고 복원할 때 남는 **수학적 격자 잔상**
    - **주파수 특징:** 외곽 영역에 **십자가(+) 및 규칙적인 점/격자 패턴**이 나타납니다.
    """)
    ai_sample = generate_sample_spectrum("ai")
    fig_ai = px.imshow(ai_sample, color_continuous_scale="gray", title="[예시] AI 사진의 규칙적인 격자 주파수 스펙트럼")
    fig_ai.update_layout(coloraxis_showscale=False, height=260, margin=dict(l=10, r=10, b=10, t=40))
    st.plotly_chart(fig_ai, use_container_width=True)

st.markdown("---")

# 4. 범용 주파수 스펙트럼 분석 알고리즘
def analyze_general_spectrum(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 512x512 고정 해상도로 규격화
    gray = cv2.resize(gray, (512, 512))

    # 2D 푸리에 변환
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 피사체 폼(중심부 저주파) 마스킹 처리
    h, w = mag_norm.shape
    cy, cx = h // 2, w // 2
    cv2.circle(mag_norm, (cx, cy), 40, 0, -1)

    # 각도별 주파수 에너지 편향성(방향성 격자 특성) 연산
    # 실제 사진은 각도별 편차가 적고(동그람), AI 사진은 특정 각도로 에너지 쏠림 현상 발생
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    polar_energies = []
    
    for angle in angles:
        # 주요 대각선 및 수직/수평 각도 스캔
        dx = np.cos(angle)
        dy = np.sin(angle)
        sample_line = [mag_norm[int(cy + r * dy), int(cx + r * dx)] for r in range(50, 200)]
        polar_energies.append(np.mean(sample_line))

    # 방향성 편향도(표준편차/평균) 계산
    bias_score = np.std(polar_energies) / (np.mean(polar_energies) + 1e-5)

    # 범용 임계치 적용 확률 스케일링
    ai_prob = min(max((bias_score - 0.22) * 250 + 50, 3.0), 97.0)
    real_prob = 100.0 - ai_prob

    return mag_norm, ai_prob, real_prob, bias_score

# 5. 실시간 업로드 분석
st.header("🔍 직접 사진 업로드하여 분석하기")
uploaded_file = st.file_uploader("인터넷이나 외부에서 가져온 사진을 업로드해 보세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # 분석
    mag_spectrum, ai_prob, real_prob, bias_score = analyze_general_spectrum(img_array)

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. 입력된 사진")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("2. 실시간 추출 주파수")
        fig_user = px.imshow(mag_spectrum, color_continuous_scale="gray")
        fig_user.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_user, use_container_width=True)
        st.caption("▲ 상단 예시와 형태(동그라미 vs 격자/십자가)를 비교해 보세요.")

    with col3:
        st.subheader("3. 판별 결과")
        
        if ai_prob > 50:
            st.error("🤖 **[인공지능 생성 사진]으로 구분되었습니다.**")
            st.caption("고주파 영역에서 특정 방향으로 치우친 격자 연산 특성이 감지되었습니다.")
        else:
            st.success("📷 **[실제 카메라 사진]으로 구분되었습니다.**")
            st.caption("고주파 영역이 동그랗고 균일하게 퍼진 광학 센서 특성을 보입니다.")

        st.progress(int(ai_prob))
        st.write(f"• **인공지능 사진 확률:** `{ai_prob:.1f}%`")
        st.write(f"• **실제 사진 확률:** `{real_prob:.1f}%`")
        st.write(f"• **주파수 방향성 편향 수치:** `{bias_score:.3f}`")

else:
    st.info("👆 준비하신 아무 실제 사진이나 AI 사진을 올려서 실시간으로 주파수를 분석해 보세요.")
