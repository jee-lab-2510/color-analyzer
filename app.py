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

def plot_bar(hist, centers):
    """색상 비율을 보여주는 가로 스펙트럼 바 생성"""
    bar = np.zeros((100, 1000, 3), dtype="uint8")
    startX = 0
    # 스펙트럼 바는 항상 분포율(%) 순서대로 그리는 것이 시각적으로 자연스럽습니다.
    # (zip으로 묶어서 정렬)
    zipped = sorted(zip(hist, centers), key=lambda x: x[0], reverse=True)
    
    for (percent, color) in zipped:
        endX = startX + (percent * 1000)
        bar[:, int(startX):int(endX)] = color.astype("uint8")
        startX = endX
    return bar

# --- 4. 메인 UI 및 로직 ---

st.title("🌌 우주 색소 & 에너지 분석기")
st.markdown("우주 사진의 색상을 분석하여 **에너지 분포**를 시각화하고 상세 데이터를 제공합니다.")

# 사이드바
st.sidebar.header("🔭 관측 옵션")
k_value = st.sidebar.slider("추출할 색상 개수", 3, 20, 10)
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

    with st.spinner('광자 에너지 계산 중...'):
        # 1. 색상 분석
        hist, centers = analyze_colors(image, k_value, resize_quality)
        
        # 2. 데이터 구조화
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
            # --- 정렬 버튼 ---
            sort_option = st.radio(
                "정렬 기준 선택:",
                ("색상 분포(%) 많은 순", "에너지(eV) 높은 순"),
                horizontal=True
            )

            # 데이터 정렬 로직
            if sort_option == "에너지(eV) 높은 순":
                sorted_data = sorted(data_list, key=lambda x: x['energy'], reverse=True)
                sort_label = "Rank"
            else:
                sorted_data = sorted(data_list, key=lambda x: x['percent'], reverse=True)
                sort_label = "Rank"

            # --- 탭 구성 ---
            tab1, tab2 = st.tabs(["⚡ 에너지 그래프", "🎨 색상 스펙트럼 & 상세"])

            with tab1:
                # [탭 1] 에너지 막대 그래프
                fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
                fig_bar.patch.set_facecolor('#f0f2f6')
                ax_bar.set_facecolor('#f0f2f6')
                
                plot_energies = [d['energy'] for d in sorted_data]
                plot_colors = [d['color']/255 for d in sorted_data]
                plot_labels = [f"{d['hex']}" for d in sorted_data] # 라벨을 색상코드로 변경

                y_pos = np.arange(len(sorted_data))
                ax_bar.barh(y_pos, plot_energies, color=plot_colors, height=0.6)
                ax_bar.set_yticks(y_pos)
                ax_bar.set_yticklabels(plot_labels)
                ax_bar.invert_yaxis() 
                
                ax_bar.set_xlabel("광자 에너지 (eV)")
                ax_bar.set_title(f"색상별 에너지 ({sort_option})")
                
                st.pyplot(fig_bar)
                
                # 간단 요약
                max_e = max(d['energy'] for d in data_list)
                min_e = min(d['energy'] for d in data_list)
                st.info(f"이 사진의 에너지 범위는 **{min_e:.2f} eV** ~ **{max_e:.2f} eV** 입니다.")

            with tab2:
                # [탭 2] 스펙트럼 바 + 상세 리스트
                st.write("**🌈 색상 분포 스펙트럼**")
                # 스펙트럼 바는 전체 분포를 보여주므로 항상 % 순으로 생성
                bar_image = plot_bar(hist, centers)
                st.image(bar_image, use_column_width=True)
                
                st.write(f"**📝 상세 데이터 ({sort_option})**")
                
                # 리스트 출력 (선택한 정렬 기준에 따름)
                for item in sorted_data:
                    st.markdown(
                        f"""
                        <div style="
                            display: flex; 
                            align-items: center; 
                            margin-bottom: 8px; 
                            padding: 10px; 
                            background-color: white; 
                            border-radius: 5px; 
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="
                                width: 40px; 
                                height: 40px; 
                                background-color: {item['hex']}; 
                                border: 1px solid #ddd; 
                                margin-right: 15px; 
                                border-radius: 4px;">
                            </div>
                            <div style="font-family: monospace; color: #333; width: 100%;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span style="font-weight: bold; font-size: 1.1em;">{item['hex']}</span>
                                    <span style="color: #666;">점유율: {item['percent']*100:.1f}%</span>
                                </div>
                                <div style="margin-top: 4px; font-size: 0.9em;">
                                    파장: {item['wavelength']:.1f} nm │ <span style="color: #d63031; font-weight: bold;">에너지: {item['energy']:.3f} eV</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
