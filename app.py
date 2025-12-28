import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="우주 색소 에너지 분석기", layout="wide")

# --- 2. 한글 폰트 설정 ---
def setup_korean_font():
    font_found = False
    for font in fm.fontManager.ttflist:
        if 'Nanum' in font.name:
            plt.rc('font', family=font.name)
            font_found = True
            break
    if not font_found:
        if os.name == 'nt':  # Windows
            plt.rc('font', family='Malgun Gothic')
        elif os.name == 'posix':  # Mac/Linux
            plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# --- 3. 과학 상수 및 계산 함수 ---
H_PLANCK = 6.626e-34
C_LIGHT = 3.00e8
EV_PER_JOULE = 6.242e18 

def rgb_to_wavelength(rgb):
    """RGB -> 파장(nm) 근사 변환"""
    r, g, b = rgb[0], rgb[1], rgb[2]
    if r > g and r > b: # Reddish
        wavelength = 620 + (130 * (r/255))
    elif g > r and g > b: # Greenish
        wavelength = 495 + (125 * (g/255))
    elif b > r and b > g: # Blueish
        wavelength = 380 + (115 * (b/255))
    else:
        wavelength = 550 
    return max(380, min(750, wavelength))

def calculate_photon_energy(wavelength_nm):
    """파장(nm) -> 에너지(eV)"""
    wavelength_m = wavelength_nm * 1e-9
    energy_joule = (H_PLANCK * C_LIGHT) / wavelength_m
    return energy_joule * EV_PER_JOULE

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
    
    return hist, clt.cluster_centers_

# --- 4. 메인 UI 및 로직 ---

st.title("🌌 우주 색소 & 에너지 분석기")
st.markdown("우주 사진의 색상을 분석하여 **에너지 분포**와 **구성 비율**을 시각화합니다.")

# 사이드바
st.sidebar.header("🔭 관측 옵션")
k_value = st.sidebar.slider("추출할 색상 개수", 3, 20, 8)
resize_quality = st.sidebar.select_slider(
    "분석 정밀도", options=[200, 400, 600, 800], value=600
)

uploaded_file = st.file_uploader("우주 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("📷 원본 이미지")
        st.image(image, use_column_width=True)

    with st.spinner('데이터 처리 중...'):
        # 1. 색상 분석
        hist, centers = analyze_colors(image, k_value, resize_quality)
        
        # 2. 데이터 구조화 (정렬을 위해 리스트로 변환)
        data_list = []
        for i, (percent, color) in enumerate(zip(hist, centers)):
            color_int = color.astype(int)
            wavelength = rgb_to_wavelength(color_int)
            energy = calculate_photon_energy(wavelength)
            
            data_list.append({
                "percent": percent,
                "color": color_int,
                "wavelength": wavelength,
                "energy": energy,
                "hex": '#{:02x}{:02x}{:02x}'.format(*color_int)
            })

        with col2:
            st.subheader("📊 분석 컨트롤 패널")
            # --- 정렬 버튼 추가 ---
            sort_option = st.radio(
                "그래프 정렬 기준을 선택하세요:",
                ("색상 분포(%) 많은 순", "에너지(eV) 높은 순"),
                horizontal=True
            )

            # 선택에 따른 데이터 정렬
            if sort_option == "에너지(eV) 높은 순":
                # 에너지가 높은 순서대로 정렬 (내림차순)
                sorted_data = sorted(data_list, key=lambda x: x['energy'], reverse=True)
                sort_label = "순위(에너지)"
            else:
                # 분포 비율이 높은 순서대로 정렬 (내림차순)
                sorted_data = sorted(data_list, key=lambda x: x['percent'], reverse=True)
                sort_label = "순위(분포)"

            # --- 시각화 데이터 준비 ---
            plot_energies = [d['energy'] for d in sorted_data]
            plot_percents = [d['percent'] for d in sorted_data]
            plot_colors = [d['color']/255 for d in sorted_data]
            plot_labels = [f"{sort_label} {i+1}" for i in range(len(sorted_data))]

            # 탭을 사용하여 그래프 분리
            tab1, tab2 = st.tabs(["⚡ 에너지 막대 그래프", "🥧 색상 분포 파이차트"])

            with tab1:
                # --- 막대 그래프 (에너지) ---
                fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
                fig_bar.patch.set_facecolor('#f0f2f6')
                ax_bar.set_facecolor('#f0f2f6')
                
                y_pos = np.arange(len(sorted_data))
                ax_bar.barh(y_pos, plot_energies, color=plot_colors, height=0.7)
                ax_bar.set_yticks(y_pos)
                ax_bar.set_yticklabels(plot_labels)
                ax_bar.invert_yaxis() # 상위 항목이 위로 오게
                
                ax_bar.set_xlabel("광자 에너지 (eV)")
                ax_bar.set_title(f"주요 색상별 에너지 ({sort_option})")
                
                st.pyplot(fig_bar)

            with tab2:
                # --- 파이 차트 (분포) - 리스트 대체 ---
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                fig_pie.patch.set_facecolor('#f0f2f6')
                
                # 파이 차트 그리기
                wedges, texts, autotexts = ax_pie.pie(
                    plot_percents, 
                    labels=plot_labels,
                    colors=plot_colors,
                    autopct='%1.1f%%', # 퍼센트 표시
                    startangle=90,
                    textprops=dict(color="black")
                )
                
                ax_pie.set_title("우주 이미지 색상 구성 비율")
                st.pyplot(fig_pie)
                
            # --- 간단한 요약 정보 표시 ---
            st.info(f"""
            **분석 요약:**
            * 가장 높은 에너지는 **{max(plot_energies):.2f} eV** 입니다.
            * 가장 많이 분포한 색상은 전체의 **{max(plot_percents)*100:.1f}%** 를 차지합니다.
            """)
