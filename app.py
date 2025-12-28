import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# 페이지 설정
st.set_page_config(page_title="이미지 색소 정밀 분석기", layout="wide")

st.title("🎨 고정밀 그림 색소 분석기")

# 사이드바 설정
st.sidebar.header("분석 옵션")
# 1. 사용자가 색상 개수(K)를 직접 조절하게 함 (정확도 튜닝)
k_value = st.sidebar.slider("추출할 색상 개수", min_value=3, max_value=20, value=8)
# 2. 이미지 리사이징 크기 조절 (품질 vs 속도)
resize_quality = st.sidebar.select_slider(
    "분석 품질 (높을수록 느리지만 정확함)",
    options=[200, 400, 600, 800, 1000],
    value=600
)

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

def analyze_colors(image, k, resize_val):
    """이미지 색상 분석 함수 (품질 향상 버전)"""
    # 사용자가 설정한 크기로 리사이징
    img = image.resize((resize_val, resize_val))
    img_array = np.array(img)
    
    # 2차원 배열로 변환
    img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
    
    # K-Means 클러스터링
    clt = KMeans(n_clusters=k, n_init=10) # n_init을 명시하여 정확도 안정화
    clt.fit(img_array)
    
    # 비율 계산
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    
    # 빈도수 순으로 정렬
    zipped = sorted(zip(hist, clt.cluster_centers_), key=lambda x: x[0], reverse=True)
    hist, centers = zip(*zipped)
    
    return hist, centers

def plot_bar(hist, centers):
    """스펙트럼 바 차트 생성"""
    bar = np.zeros((100, 1000, 3), dtype="uint8") # 바 크기를 키움
    startX = 0
    
    for (percent, color) in zip(hist, centers):
        endX = startX + (percent * 1000)
        bar[:, int(startX):int(endX)] = color.astype("uint8")
        startX = endX
        
    return bar

if uploaded_file is not None:
    # 레이아웃을 2단으로 나눔
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_column_width=True)

    with st.spinner('정밀 분석 중...'):
        hist, centers = analyze_colors(image, k_value, resize_quality)
        bar = plot_bar(hist, centers)
        
        with col2:
            st.subheader("분석 결과")
            st.write(f"**총 {k_value}개의 주요 색소 추출됨**")
            st.image(bar, use_column_width=True, caption="색상 분포 스펙트럼")
            
            # 상세 분석 테이블
            st.write("### 색상 상세 데이터")
            for percent, color in zip(hist, centers):
                color_int = color.astype(int)
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color_int)
                
                # 색상 박스와 텍스트를 한 줄에 표시
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 30px; height: 30px; background-color: {hex_color}; border: 1px solid #ddd; margin-right: 10px;"></div>
                        <div style="font-family: monospace;">
                            <b>{hex_color}</b> : {percent*100:.2f}% (R:{color_int[0]} G:{color_int[1]} B:{color_int[2]})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
