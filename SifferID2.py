import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 1. INIT-RUTIN (Körs bara en gång)
# ==========================================
@st.cache_resource
def init_model():
    
    st.write("Laddar MNIST-data... (detta sker bara en gång)")
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # För demo-hastighet tränar vi på en delmängd, 
    # men du kan öka detta för bättre precision
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=2000, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)# Bara transform på testdatan, inte fit_transform

    #model = LogisticRegression(C=1.0,  tol=0.1) # Test accuracy: 0.7430
    #model = RandomForestClassifier(n_estimators=100, random_state=42) # Test accuracy: 0.9680   
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)# Test accuracy: 0.9710
    model.fit(X_train_scaled, y_train_val)
    
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, X_test, y_test, acc, scaler

# Kör init-rutinen
model, X_test, y_test, accuracy, scaler = init_model()

# ==========================================
# 2. BILDHANTERING & PREDIKTION
# ==========================================
def predict_digit(img_array):
    try:
        if img_array is None:
            return None, {}

        # Skapa Pillow-bild (RGBA -> L)
        raw_img = Image.fromarray(img_array.astype(np.uint8)).convert('L')

        # Invertera: MNIST bilder är vit siffra (255) på svart bakgrund (0)
        # st_canvas ritar svart siffra på vit bakgrund.
        if np.mean(np.array(raw_img)) > 128:
            clean_img = ImageOps.invert(raw_img)
        else:
            clean_img = raw_img.copy()

        # Skala ner till 28x28 (som MNIST)
        img_28 = clean_img.resize((28, 28), Image.Resampling.LANCZOS)

        # Förbered för prediktion
        final_input = np.array(img_28).reshape(1, 784)# På en kolumn, 784 rader
        
        final_input_scaled = scaler.transform(final_input)

        probs = model.predict_proba(final_input_scaled)[0]
        # Skapa en dict med sannolikheter för varje siffra
        label_output = {str(i): float(probs[i]) for i in range(10)}

        return img_28, label_output

    except Exception as e:
        st.error(f"Fel i bildbehandling: {e}")
        return None, {"error": str(e)}

# ==========================================
# 3. GRÄNSSNITT
# ==========================================
st.set_page_config(page_title="MNIST Labration", layout="centered")
st.title("Siffer-igenkänning ")
st.caption(f"Modellens noggrannhet: {accuracy:.2%}")

col1, col2, col3 = st.columns([2, 1, 1.5])

with col1:
    stroke_width = st.sidebar.slider("Pen storlek: ", 1, 25, 15)
    # --- NYTT: Väljare för att utforska MNIST ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Siffror från MNIST-databasen")
    digit_to_show = st.sidebar.selectbox("Välj en siffra att granska:", range(10))
    
    if st.sidebar.button(f"Visa exempel på {digit_to_show}"):
        # Hitta alla index för den valda siffran
        digit_indices = np.where(y_test.astype(str) == str(digit_to_show))[0]
        
        # Slumpa fram 5 index för att göra det lite roligare varje gång man klickar
        sample_indices = np.random.choice(digit_indices, 5, replace=False)
        
        st.sidebar.write(f"Exempel på {digit_to_show}:or:")
        
        # Visa bilderna i sidomenyn (i ett litet galleri)
        side_cols = st.sidebar.columns(5)
        for i, idx in enumerate(sample_indices):
            # Kom ihåg: X_test är en numpy-array, så ingen .iloc här
            img = X_test[idx].reshape(28, 28)
            side_cols[i].image(img, use_container_width=True)
    # --------------------------------------------
    st.markdown("### Rita en siffra (0-9)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width, 
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True # Gör att den uppdateras när man ritar
    )

# Vi kör prediktionen utanför kolumnerna så att datan finns redo
if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] < 255):
    processed_img, predictions = predict_digit(canvas_result.image_data)
    
    with col2:
        st.markdown("### 2. Input")
        if processed_img:
            st.image(processed_img, width=120, caption="AI-vy (28x28)")

    with col3:
        st.markdown("### 3. Analys")
        if predictions:
            best_guess = max(predictions, key=predictions.get)
            confidence = predictions[best_guess]
            
            st.metric("Gissning", f"Siffra {best_guess}", f"{confidence:.1%}")
            st.bar_chart(predictions)

    # --- NY SEKTION: GALLERI UNDER KOLUMNERNA ---
    # --- GALLERI-KODEN ---
    st.markdown("---")
    st.subheader(f"Så här ser 10 st '{best_guess}:or' ut i MNIST-databasen")

    # Eftersom y_test är en numpy-array (tack vare as_frame=False)
    # använder vi .astype(str) för att säkerställa att jämförelsen med best_guess funkar
    all_indices = np.where(y_test.astype(str) == str(best_guess))[0]
    indices = all_indices[:10]

    if len(indices) > 0:
        img_cols = st.columns(5)
        for i, idx in enumerate(indices):
            with img_cols[i % 5]:
                # Här tar vi bort .iloc eftersom X_test är en numpy-array
                # Vi plockar raden idx och formar om till 28x28
                sample_img = X_test[idx].reshape(28, 28)
                
                st.image(sample_img, width=60)
    else:
        st.warning(f"Kunde inte hitta några referensbilder för siffra {best_guess}.")

else:
    with col3:
        st.info("Väntar på bilden...")