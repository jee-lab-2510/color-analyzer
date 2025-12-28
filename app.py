import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 페이지 설정 ---
st.set_page_config(page_title="우주 색소 에너지 분석기", layout="wide")

# --- 폰트 설정 (packages.txt 방식) ---
# 리눅스(Streamlit Cloud)에 설치된 나눔 폰트를 찾아서 설정합니다.
def setup_korean_font():
    # 1. 나눔 폰트가 설치되어 있는지 확인
    font_found = False
    for font in fm.fontManager.ttflist:
        if 'Nanum' in font.name:
            plt.rc('font', family=font.name)
            font_found = True
            break
            
    # 2. 설치된 폰트가 없으면(로컬 실행 등) 기본 폰트 시도
    if not font_found:
        # 윈도우/맥 등 로컬 환경을 위한 예비책
        if os.name == 'nt':  # Windows
            plt.rc('font', family='Malgun Gothic')
        elif os.name == 'posix':  # Mac/Linux
            plt.rc('font', family='AppleGothic')
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 폰트 설정 실행
setup_korean_font()

# --- 상수 정의 ---
H_PLANCK = 6.626e-34  
C_LIGHT = 3.00e8    
EV_PER_JOULE = 6.242e18 

st.title("✨ 우주 이미지 색소 & 에너지 분석기")
st.write("우주 사진에서 주요 색상을 추출하고, 해당 빛의 에너지를 분석합니다.")

# --- 사이드바 ---
st.sidebar.header("분석 옵션")
k_value = st.sidebar.slider("추출할 주요 색상 개수", 3, 20, 8)
resize_quality = st.sidebar.select_slider(
    "분석 품질 (높을수록 정밀)", options=[200, 400, 600, 800], value=600
)

uploaded_file = st.file_uploader("우주 이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

# --- 함수 정의 ---
def analyze_colors(image, k, resize_val):
    img = image.resize((resize_val, resize_val))
    img_array = np.array(img)
    img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
    
    clt = KMeans(n_clusters=k, n_init=10, random_state=42)
    clt.fit(img_array)
    
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    
    zipped = sorted(zip(hist, clt.cluster_centers_), key=lambda x: x[0], reverse=True)
    hist, centers = zip(*zipped)
    return hist, centers

def plot_bar(hist, centers):
    bar = np.zeros((100, 1000, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centers):
        endX = startX + (percent * 1000)
        bar[:, int(startX):int(endX)] = color.astype("uint8")
        startX = endX
    return bar

def rgb_to_wavelength(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    if r > g and r > b: 
        wavelength = 620 + (130 * (r/255))
    elif g > r and g > b: 
        wavelength = 495 + (125 * (g/255))
    elif b > r and b > g: 
        wavelength = 380 + (115 * (b/255))
    else:
        wavelength = 550 

    return max(380, min(750, wavelength))

def calculate_photon_energy(wavelength_nm):
    wavelength_m = wavelength_nm * 1e-9
    energy_joule = (H_PLANCK * C_LIGHT) / wavelength_m
    return energy_joule * EV_PER_JOULE

# --- 메인 실행 ---
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("원본 우주 이미지")
        st.image(image, use_column_width=True)

    with st.spinner('우주 에너지 분석 중...'):
        hist, centers = analyze_colors(image, k_value, resize_quality)
        bar = plot_bar(hist, centers)
        
        with col2:
            st.subheader("분석 결과")
            st.image(bar, use_column_width=True, caption="색상 분포 스펙트럼")
            
            st.subheader("🌠 빛의 파장 및 에너지")
            energy_values = []
            labels = []

            for i, (percent, color) in enumerate(zip(hist, centers)):
                color_int = color.astype(int)
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color_int)
                wavelength = rgb_to_wavelength(color_int)
                energy_ev = calculate_photon_energy(wavelength)
                
                energy_values.append(energy_ev)
                labels.append(f"색상 {i+1}") 
                
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 30px; height: 30px; background-color: {hex_color}; border: 1px solid #ddd; margin-right: 10px;"></div>
                        <div style="font-family: monospace;">
                            <b>{hex_color}</b> ({percent*100:.1f}%) <br>
                            파장: {wavelength:.1f} nm, 에너지: {energy_ev:.3f} eV
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # --- 그래프 그리기 ---
            st.subheader("⚡ 에너지 스펙트럼")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # 배경색 깔끔하게
            fig.patch.set_facecolor('#f0f2f6')
            ax.set_facecolor('#f0f2f6')

            for j in range(len(energy_values)):
                ax.barh(labels[j], energy_values[j], color=[c / 255. for c in centers[j]])
            
            ax.set_xlabel("에너지 (eV)")
            ax.set_ylabel("추출된 색상")
            ax.set_title("색상별 광자 에너지 분석")
            ax.invert_yaxis()
            
            st.pyplot(fig)
            
