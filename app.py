import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- 상수 정의 ---
H_PLANCK = 6.626e-34  # 플랑크 상수 (Joule * second)
C_LIGHT = 3.00e8    # 빛의 속도 (meters / second)
EV_PER_JOULE = 6.242e18 # 1 줄(Joule) 당 전자볼트(eV)

# --- 페이지 설정 ---
st.set_page_config(page_title="우주 색소 에너지 분석기", layout="wide")

st.title("✨ 우주 이미지 색소 & 에너지 분석기")
st.write("우주 사진에서 주요 색상을 추출하고, 해당 빛의 에너지를 분석합니다.")

# --- 사이드바 설정 ---
st.sidebar.header("분석 옵션")
k_value = st.sidebar.slider("추출할 주요 색상 개수", min_value=3, max_value=20, value=8)
resize_quality = st.sidebar.select_slider(
    "분석 품질 (높을수록 느리지만 정밀함)",
    options=[200, 400, 600, 800],
    value=600 # 우주 사진은 디테일이 많으므로 기본값을 높게 설정
)

uploaded_file = st.file_uploader("우주 이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

# --- 함수 정의 ---
def analyze_colors(image, k, resize_val):
    """이미지에서 주요 색상 k개를 추출하는 함수 (개선 버전)"""
    img = image.resize((resize_val, resize_val))
    img_array = np.array(img)
    img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
    
    clt = KMeans(n_clusters=k, n_init=10, random_state=42) # random_state 추가로 결과 일관성 유지
    clt.fit(img_array)
    
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    
    zipped = sorted(zip(hist, clt.cluster_centers_), key=lambda x: x[0], reverse=True)
    hist, centers = zip(*zipped)
    
    return hist, centers

def plot_bar(hist, centers):
    """색상 스펙트럼 바 차트 생성"""
    bar = np.zeros((100, 1000, 3), dtype="uint8")
    startX = 0
    for (percent, color) in zip(hist, centers):
        endX = startX + (percent * 1000)
        bar[:, int(startX):int(endX)] = color.astype("uint8")
        startX = endX
    return bar

def rgb_to_wavelength(rgb):
    """
    RGB 값을 가시광선 파장(nm)으로 근사적으로 매핑하는 함수.
    이것은 대략적인 근사치이며, 실제 스펙트럼 분석과는 다릅니다.
    """
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    # 간단한 RGB 가중치 기반 파장 근사 (매우 근사적임)
    # 가시광선 범위: 대략 380nm (보라) ~ 750nm (빨강)
    
    # 각 채널의 기여도를 고려한 가중 평균
    # 빨강은 긴 파장, 파랑은 짧은 파장
    wavelength = (r * 700) + (g * 550) + (b * 450)
    
    # 값 정규화 및 가시광선 범위로 스케일링 (380-750nm)
    # 이 부분은 RGB 색상 공간에서 파장으로의 매핑이 비선형적이고 복잡하기 때문에,
    # 정확한 물리적 파장으로 변환하는 것은 어렵고, 여기서는 대략적인 경향을 반영
    
    # 380nm(보라) - 750nm(빨강) 범위로 스케일링
    # R이 높으면 750nm에 가깝게, B가 높으면 380nm에 가깝게
    if r > g and r > b: # Mostly red
        wavelength = 620 + 130 * r # 620nm (주황) to 750nm (빨강)
    elif g > r and g > b: # Mostly green
        wavelength = 495 + 125 * g # 495nm (청록) to 620nm (주황)
    elif b > r and b > g: # Mostly blue
        wavelength = 380 + 115 * b # 380nm (보라) to 495nm (청록)
    else: # Mix of colors (e.g., white, gray) -> center of spectrum
        wavelength = 550 # Yellow/Green center
    
    # 범위 제한 (380nm ~ 750nm)
    wavelength = max(380, min(750, wavelength))
    
    return wavelength

def calculate_photon_energy(wavelength_nm):
    """
    파장(nm)을 광자 1개의 에너지(eV)로 계산하는 함수.
    E = hc / lambda (Joule)
    1 eV = 1.602e-19 Joule
    """
    wavelength_m = wavelength_nm * 1e-9 # nm를 m로 변환
    energy_joule = (H_PLANCK * C_LIGHT) / wavelength_m
    energy_ev = energy_joule * EV_PER_JOULE
    return energy_ev

# --- 메인 로직 ---
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
            st.write(f"**총 {k_value}개의 주요 색소 추출됨**")
            st.image(bar, use_column_width=True, caption="색상 분포 스펙트럼")
            
            st.subheader("🌠 빛의 파장 및 에너지")
            
            # 에너지 그래프를 그리기 위한 데이터 준비
            energy_values = []
            labels = []

            for i, (percent, color) in enumerate(zip(hist, centers)):
                color_int = color.astype(int)
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color_int)
                
                # RGB를 파장으로 변환
                wavelength = rgb_to_wavelength(color_int)
                
                # 파장을 에너지로 변환
                energy_ev = calculate_photon_energy(wavelength)
                
                energy_values.append(energy_ev)
                labels.append(f"C{i+1}") # Color 1, Color 2...
                
                # 상세 정보 표시
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 30px; height: 30px; background-color: {hex_color}; border: 1px solid #ddd; margin-right: 10px;"></div>
                        <div style="font-family: monospace;">
                            <b>{hex_color}</b> : {percent*100:.2f}% <br>
                            파장: {wavelength:.1f} nm <br>
                            에너지: {energy_ev:.3f} eV
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # --- 에너지 스펙트럼 그래프 ---
            st.subheader("⚡ 에너지 스펙트럼")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # 에너지 값과 해당 색상으로 바 차트 생성
            for j in range(len(energy_values)):
                ax.barh(labels[j], energy_values[j], color=[c / 255. for c in centers[j]])
            
            ax.set_xlabel("에너지 (eV)")
            ax.set_ylabel("주요 색상")
            ax.set_title("추출된 색상별 광자 에너지")
            ax.invert_yaxis() # 가장 높은 에너지가 위에 오도록
            st.pyplot(fig)
