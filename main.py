import datetime
import requests
import pandas as pd
import pytz
import streamlit as st

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="일별 박스오피스 대시보드",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 일별 박스오피스 대시보드")

# -----------------------------------------------------------------------------
# 2. 한국 표준시(KST) 기준 선택 가능 날짜 설정 (최대 어제까지)
# -----------------------------------------------------------------------------
kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.datetime.now(kst)
yesterday_kst = (now_kst - datetime.timedelta(days=1)).date()

# 달력에서 조회할 날짜 선택 (기본값: 어제, 최대 선택 가능일: 어제)
selected_date = st.date_input(
    "조회할 날짜를 선택하세요 (오늘 날짜는 아직 집계 전입니다)",
    value=yesterday_kst,
    max_value=yesterday_kst,
    min_value=datetime.date(2004, 1, 1)  # KOBIS 제공 시작 연도 기준
)

# API 요청에 사용할 YYYYMMDD 및 표시용 YYYY-MM-DD
target_date_str = selected_date.strftime("%Y%m%d")
display_date_str = selected_date.strftime("%Y-%m-%d")

st.caption(f"📅 현재 조회 중인 날짜: **{display_date_str}**")

# -----------------------------------------------------------------------------
# 3. KOBIS API 인증키 가져오기 (secrets.toml)
# -----------------------------------------------------------------------------
if "KOBIS_KEY" not in st.secrets:
    st.error("🔑 API 인증키(KOBIS_KEY)를 찾을 수 없습니다.")
    st.info("""
    **해결 방법 (Streamlit Cloud):**
    1. Streamlit Cloud의 대시보드에서 앱 설정(App Settings)으로 이동합니다.
    2. **Secrets** 메뉴를 선택합니다.
    3. 아래와 같이 KOBIS_KEY를 설정하고 저장(Save)해주세요.
    
    ```toml
    KOBIS_KEY = "발급받은_KOBIS_인증키"
    ```
    """)
    st.stop()

api_key = st.secrets["KOBIS_KEY"]

# -----------------------------------------------------------------------------
# 4. KOBIS 박스오피스 API 데이터 요청
# -----------------------------------------------------------------------------
url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
params = {
    "key": api_key,
    "targetDt": target_date_str
}

try:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    st.error("🌐 네트워크 요청 중 오류가 발생했습니다.")
    st.warning("확인 사항: 인터넷 연결 상태나 KOBIS API 서버 점검 여부를 확인해 주세요.")
    st.caption(f"상세 오류: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 5. API 응답 데이터 예외 처리 (faultInfo 또는 데이터 부재 검사)
# -----------------------------------------------------------------------------
if "faultInfo" in data:
    st.error("⚠️ KOBIS API 오류가 발생했습니다.")
    st.warning("""
    **확인해야 할 사항:**
    1. `.streamlit/secrets.toml` 또는 Streamlit Cloud의 **Secrets**에 입력한 `KOBIS_KEY`가 올바른지 확인해 주세요.
    2. KOBIS 공식 홈페이지에서 인증키가 정상 발급/활성화되었는지 확인해 주세요.
    """)
    st.caption(f"KOBIS 메시지: {data['faultInfo'].get('message', '알 수 없는 오류')}")
    st.stop()

# dailyBoxOfficeList 추출
boxoffice_result = data.get("boxOfficeResult", {})
movie_list = boxoffice_result.get("dailyBoxOfficeList", [])

# 영화 목록이 비어서 들어올 경우 처리
if not movie_list:
    st.warning(f"📭 선택하신 날짜({display_date_str})는 아직 집계 전입니다.")
    st.info("""
    **안내 사항:**
    1. 영화관 집계 일정에 따라 해당 날짜의 데이터가 아직 반영되지 않았을 수 있습니다.
    2. 다른 날짜를 선택하여 조회를 시도해 주세요.
    """)
    st.stop()

# -----------------------------------------------------------------------------
# 6. 데이터 전처리
# -----------------------------------------------------------------------------
df = pd.DataFrame(movie_list)

# API 응답값(문자열)을 정수(int) 형으로 변환
numeric_cols = ['rank', 'rankInten', 'audiCnt', 'audiAcc', 'scrnCnt', 'showCnt']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# -----------------------------------------------------------------------------
# 7. 요구사항 반영: 순위 변동 기호 및 누적 100만 관객 트로피 처리 함수 적용
# -----------------------------------------------------------------------------

# 1) 순위 변동 화살표 텍스트 생성 (양수: 빨간 위 화살표, 음수: 파란 아래 화살표)
def format_rank_change(row):
    rank = row['rank']
    inten = row['rankInten']
    if inten > 0:
        return f"{rank} (🔺{inten})"
    elif inten < 0:
        return f"{rank} (🔹{abs(inten)})"
    else:
        return f"{rank} (-)"

# 2) 누적관객 100만 명 이상 영화 트로피(🏆) 추가
def format_movie_title(row):
    title = row['movieNm']
    acc = row['audiAcc']
    if acc >= 1000000:
        return f"🏆 {title}"
    return title

df['formatted_rank'] = df.apply(format_rank_change, axis=1)
df['formatted_movie_nm'] = df.apply(format_movie_title, axis=1)

# -----------------------------------------------------------------------------
# 8. 화면 구성 - 1위 영화 지표 카드 3개
# -----------------------------------------------------------------------------
top_movie = df.iloc[0]

st.subheader(f"🥇 1위 영화: {top_movie['formatted_movie_nm']}")

col1, col2, col3 = st.columns(3)

rank_inten = top_movie['rankInten']
if rank_inten > 0:
    rank_delta = f"▲ {rank_inten}"
elif rank_inten < 0:
    rank_delta = f"▼ {abs(rank_inten)}"
else:
    rank_delta = "변동 없음"

with col1:
    st.metric(
        label="당일 관객수",
        value=f"{top_movie['audiCnt']:,} 명",
        delta=f"순위 변동: {rank_delta}"
    )

with col2:
    st.metric(
        label="누적 관객수",
        value=f"{top_movie['audiAcc']:,} 명"
    )

with col3:
    st.metric(
        label="스크린수",
        value=f"{top_movie['scrnCnt']:,} 개"
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 9. 화면 구성 - 관객수 상위 5편 막대그래프
# -----------------------------------------------------------------------------
st.subheader("📊 관객수 상위 5개 영화")

top5_df = df.head(5)[['movieNm', 'audiCnt']].copy()
top5_df.columns = ['영화명', '당일 관객수']
top5_df = top5_df.set_index('영화명')

st.bar_chart(top5_df)

st.markdown("---")

# -----------------------------------------------------------------------------
# 10. 화면 구성 - 박스오피스 전체 순위 표 (DataFrame)
# -----------------------------------------------------------------------------
st.subheader("📋 선택 날짜 박스오피스 전체 순위")

# 변환된 순위 표시와 트로피 적용 영화명을 사용
display_df = df[['formatted_rank', 'formatted_movie_nm', 'openDt', 'audiCnt', 'audiAcc', 'scrnCnt']].copy()
display_df.columns = ['순위', '영화명', '개봉일', '당일 관객수', '누적 관객수', '스크린수']

st.dataframe(
    display_df,
    use_container_width=True,
    column_config={
        "당일 관객수": st.column_config.NumberColumn(
