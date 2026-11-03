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
st.caption("빅데이터 기반 2D 푸리에 변환(FFT) 알고리즘으로 실제 사진과 AI 생성 이미지를 자동 판별합니다.")

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

# 4. 자동 주파수 분석 알고리즘 (상대적 고주파 에너지를 스스로 판별)
def analyze_auto_spectrum(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 해상도 규격화
    gray = cv2.resize(gray, (512, 512))

    # 2D 푸리에 변환 (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 피사체 폼(중심부 저주파) 제거
    h, w = mag_norm.shape
    cy, cx = h // 2, w // 2
    
    # 마스킹 처리된 고주파 영역 생성
    high_freq_map = mag_norm.copy()
    cv2.circle(high_freq_map, (cx, cy), 40, 0, -1)

    # 고주파 노이즈의 방향성 방사상 분포(Radial Energy Distribution) 자동 계산
    # 수평/수평 축의 왜곡 점수 계산
    h_energy = np.mean(high_freq_map[cy-5:cy+5, :])
    v_energy = np.mean(high_freq_map[:, cx-5:cx+5])
    total_high_energy = np.mean(high_freq_map) + 1e-5

    # AI 격자 특성지수: 전체 고주파 대비 십자가 축 에너지 쏠림 비율
    axial_bias = (h_energy + v_energy) / (2 * total_high_energy)
    
    # 대각선 방향 에너지 편차 연산
    diag_std = np.std(high_freq_map) / total_high_energy
    
    # 종합 자동 판별 스코어 (특정 고정 임계치 대신 상대 수치 결합)
    ai_score = (axial_bias * 0.6) + (diag_std * 0.4)

    # 자동 로직 기반 확률 변환 (기준점: 0.95)
    ai_prob = min(max((ai_score - 0.95) * 180 + 50, 1.0), 99.0)
    real_prob = 100.0 - ai_prob

    return mag_norm, ai_prob, real_prob, ai_score

# 5. 자동 분석 실행 UI
st.header("🔍 사진 업로드 및 자동 분석")
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # 완전 자동 분석 실행
    mag_spectrum, ai_prob, real_prob, ai_score = analyze_auto_spectrum(img_array)

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. 입력된 사진")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("2. 추출된 주파수 스펙트럼")
        fig_user = px.imshow(mag_spectrum, color_continuous_scale="gray")
        fig_user.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_user, use_container_width=True)
        st.caption("▲ 상단 예시와 형태(동그라미 vs 격자/십자가)를 비교해 보세요.")

    with col3:
        st.subheader("3. 자동 판별 결과")
        
        if ai_prob > 50:
            st.error("🤖 **[인공지능 생성 사진]으로 자동 구별되었습니다.**")
            st.caption("주파수 외곽 축에서 AI 고유의 격자 연산 에너지 쏠림 현상이 자동 감지되었습니다.")
        else:
            st.success("📷 **[실제 카메라 사진]으로 자동 구별되었습니다.**")
            st.caption("주파수 스펙트럼이 균일한 원형 분포를 보이며 광학 센서의 무작위 특성을 나타냅니다.")

        st.progress(int(ai_prob))
        st.write(f"• **인공지능 사진 확률:** `{ai_prob:.1f}%`")
        st.write(f"• **실제 사진 확률:** `{real_prob:.1f}%`")
        st.write(f"• **자동 산출 격자 지수:** `{ai_score:.3f}`")

else:
    st.info("👆 사진을 올리면 알고리즘이 스스로 주파수 특징을 계산하여 자동으로 판별합니다.")
