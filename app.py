import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 페이지 설정
st.set_page_config(page_title="이미지 색상 분석기", layout="centered")

st.title("🎨 그림 속 색소 분석기")
st.write("이미지를 업로드하면 사용된 주요 색상과 스펙트럼을 분석해 드립니다.")

# 1. 입력: 그림 사진 넣기
uploaded_file = st.file_uploader("이미지를 선택하세요...", type=["jpg", "jpeg", "png"])

def analyze_colors(image, k=10):
    """이미지에서 주요 색상 k개를 추출하는 함수"""
    # 계산 속도를 위해 이미지 크기 줄이기
    img = image.resize((200, 200))
    img_array = np.array(img)
    
    # 2차원 배열로 변환 (픽셀 수 x 3(RGB))
    img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
    
    # K-Means 클러스터링으로 주요 색상 추출
    clt = KMeans(n_clusters=k)
    clt.fit(img_array)
    
    # 각 색상의 비율 계산
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    
    # 빈도수 순으로 정렬
    zipped = sorted(zip(hist, clt.cluster_centers_), key=lambda x: x[0], reverse=True)
    hist, centers = zip(*zipped)
    
    return hist, centers

def plot_colors(hist, centers):
    """색상 스펙트럼(바 차트)을 그리는 함수"""
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    
    for (percent, color) in zip(hist, centers):
        endX = startX + (percent * 300)
        # 스펙트럼 바에 색 채우기
        bar[:, int(startX):int(endX)] = color.astype("uint8")
        startX = endX
        
    return bar

# 2. 분석 및 3. 출력
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # 원본 이미지 표시
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    
    with st.spinner('색소를 분석하는 중입니다...'):
        # 색상 분석 실행 (주요 색상 10개 추출)
        hist, centers = analyze_colors(image, k=10)
        bar = plot_colors(hist, centers)
        
        st.success("분석 완료!")
        
        # 스펙트럼 출력
        st.subheader("📊 색상 스펙트럼")
        st.image(bar, caption='이미지 구성 색상 분포', use_column_width=True)
        
        # 상세 색상 정보 (옵션)
        st.write("### 주요 추출 색상 (RGB)")
        cols = st.columns(5)
        for i, (percent, color) in enumerate(zip(hist[:5], centers[:5])):
            color_int = color.astype(int)
            with cols[i]:
                # 색상 박스 표시 (HTML/CSS 활용)
                st.markdown(
                    f'<div style="background-color:rgb({color_int[0]},{color_int[1]},{color_int[2]});height:50px;border-radius:5px;"></div>',
                    unsafe_allow_html=True
                )
                st.caption(f"{percent*100:.1f}%")
