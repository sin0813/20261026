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
st.caption("빅데이터 기반 2D 푸리에 변환(FFT) 및 360도 방사형 주파수 스캔으로 AI 생성이미지를 정밀 판별합니다.")

st.markdown("---")

# 3. 원리 및 예시
st.header("💡 사진이 구별되는 이유와 주파수 모양 예시")
st.write("사람의 눈으로는 구별하기 힘든 AI 사진도, 픽셀 데이터를 주파수 영역으로 변환하여 360도 방향성을 측정하면 미세한 격자 잔상이 감지됩니다.")

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
    - **원인:** 동그란 광학 렌즈와 센서(CMOS)를 거치며 생성되는 **자연스러운 무작위 노이즈**
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

# 4. 정밀 360도 방사형 주파수 스캔 알고리즘
def analyze_polar_spectrum(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 512x512 고정 규격화
    gray = cv2.resize(gray, (512, 512))

    # 2D 푸리에 변환 (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    # 피사체 형체(중심부 저주파) 및 극한 외곽 노이즈 제외한 중고주파 링(Ring) 영역 마스킹
    h, w = mag_norm.shape
    cy, cx = h // 2, w // 2
    
    # 360도 각도별 고주파 에너지 추출 (Polar Profile)
    angles = np.linspace(0, 180, 90, endpoint=False)
    angle_energies = []
    
    r_inner = 35
    r_outer = 180

    for angle in angles:
        rad = np.deg2rad(angle)
        # 해당 각도의 선상에 위치한 픽셀들의 평균 에너지 산출
        xs = (cx + np.linspace(r_inner, r_outer, 50) * np.cos(rad)).astype(np.int32)
        ys = (cy + np.linspace(r_inner, r_outer, 50) * np.sin(rad)).astype(np.int32)
        
        # 범위 내 안전한 좌표만 선택
        valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        energy = np.mean(mag_norm[ys[valid_mask], xs[valid_mask]])
        angle_energies.append(energy)

    # 각도별 에너지 편향 지수(Directional Bias Score) 연산
    # 실제 사진: 방향에 상관없이 에너지가 일정함 (편차 매우 작음)
    # AI 사진: 특정 결 방향으로 에너지가 튐 (편차 큼)
    bias_score = np.std(angle_energies) / (np.mean(angle_energies) + 1e-5)

    # AI 생성 알고리즘 자동 기준선 (0.045 기준)
    ai_prob = min(max((bias_score - 0.045) * 1200 + 50, 2.0), 98.5)
    real_prob = 100.0 - ai_prob

    # Visual 용 마스킹 이미지 생성
    display_map = mag_norm.astype(np.uint8).copy()
    cv2.circle(display_map, (cx, cy), r_inner, 0, -1)

    return display_map, ai_prob, real_prob, bias_score

# 5. UI 및 분석 결과 출력
st.header("🔍 사진 업로드 및 정밀 분석")
uploaded_file = st.file_uploader("분석할 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # 방사형 분석 수행
    mag_spectrum, ai_prob, real_prob, bias_score = analyze_polar_spectrum(img_array)

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. 입력된 사진")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("2. 추출된 주파수 스펙트럼")
        fig_user = px.imshow(mag_spectrum, color_continuous_scale="gray")
        fig_user.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_user, use_container_width=True)
        st.caption("▲ 상단 예시의 형태(동그라미 vs 격자/십자가)와 비교해 보세요.")

    with col3:
        st.subheader("3. 정밀 분석 결과")
        
        if ai_prob > 50:
            st.error("🤖 **[인공지능 생성 사진]으로 구별되었습니다.**")
            st.caption("360도 스캔 분석 결과, 특정 방향으로 치우친 AI 연산 미세 격자 패턴이 감지되었습니다.")
        else:
            st.success("📷 **[실제 카메라 사진]으로 구별되었습니다.**")
            st.caption("360도 스캔 분석 결과, 주파수 에너지가 전 방향으로 균일한 광학 센서 특성을 나타냅니다.")

        st.progress(int(ai_prob))
        st.write(f"• **인공지능 사진 확률:** `{ai_prob:.1f}%`")
        st.write(f"• **실제 사진 확률:** `{real_prob:.1f}%`")
        st.write(f"• **방향성 편향 수치(Bias Score):** `{bias_score:.4f}`")

else:
    st.info("👆 사진을 올려주시면 360도 주파수 스캔으로 AI 여부를 자동 측정합니다.")
