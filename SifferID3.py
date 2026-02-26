import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageEnhance
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import center_of_mass, shift

# INIT-RUTIN (Körs bara en gång)

@st.cache_resource
def init_model(show_spinner=False):
    
    st.spinner("Laddar MNIST-data... (detta sker bara en gång)")
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Inte splitta upp nu, kör på hela
    # X_train_val, X_test, y_train_val, y_test = train_test_split(
    #    X_raw, y_raw, test_size=1000, random_state=42
    #)

    scaler = StandardScaler()
    # Nedan gäller när vi delade datan
    #X_train_scaled = scaler.fit_transform(X_train_val)
    #X_test_scaled = scaler.transform(X_test)# Bara transform på testdatan, inte fit_transform
    X_raw_scaled = scaler.fit_transform(X_raw).astype(np.float32) # På allt, med fit så värdena bräknas på hela datan
  
    #model = LogisticRegression(C=1.0,  tol=0.1) # Test accuracy: 0.7430
    #model = RandomForestClassifier(n_estimators=100, random_state=42) # Test accuracy: 0.9680   
    model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)# Test accuracy: 0.9710
    
    # model.fit(X_train_scaled, y_train_val)
    model.fit(X_raw_scaled, y_raw) # Träna på allt (för att få bästa möjliga precision)
    
    # acc = accuracy_score(y_test, model.predict(X_test_scaled))
    acc = 0.971 # Sätts den manuellt eftersom redan utvärderat
    
    return model, X_raw, y_raw, acc, scaler # Returnera det som behövs senare

# Kör init-rutinen
model, X_raw, y_raw, accuracy, scaler = init_model()


# BILDHANTERING & PREDIKTION



def predict_digit(img_array):
    if isinstance(img_array, np.ndarray) and len(img_array.shape) == 3:
        # 1. Gråskala och inversion
        img_gray = np.mean(img_array[:, :, :3], axis=2).astype(np.uint8)
        img_gray = 255 - img_gray 
        img_pil = Image.fromarray(img_gray)
        # Nu som gråskala och inverterad (svart bakgrund, vit penna teckning)
        #img_pil.save("C:\\Users\\ulfha\\OneDrive\\Bilder\\AI\\step1.png")
        
        # AUTOMATISK BESKÄRNING  fantastiskt
        bbox = img_pil.getbbox()# superfunktion, hämtar den rect som det är ritat i, dvs det som inte är vitt (0,0,280,280) är hela
        if bbox:
            img_pil = img_pil.crop(bbox) # Lägg den bilden här

        # Skala till 20x20 som det stod om i .DESC
        w, h = img_pil.size
        ratio = 20.0 / max(w, h) # Skala så att den längsta sidan blir 20 pixlar
        new_size = (int(w * ratio), int(h * ratio))
        img_20 = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        #img_20.save("C:\\Users\\ulfha\\OneDrive\\Bilder\\AI\\step2.png")        
        
        
        # skapa en 28x28 canvas och klistra in initialt i mitten
        canvas = Image.new('L', (28, 28), 0)
        w20, h20 = img_20.size # Ger 20, 20 eller mindre beroende på proportioner
        canvas.paste(img_20, ((28 - w20) // 2, (28 - h20) // 2)) # Lägg in den i mitten så att den är centrerad i det lilla 28x28-området
        #canvas.save("C:\\Users\\ulfha\\OneDrive\\Bilder\\AI\\step3.png")


        # MASSCENTRUM-JUSTERING För att siffran ska sitta mitt i bilden
        img_np = np.array(canvas).astype(np.float32)
        cy, cx = center_of_mass(img_np) # Vart i bilden är tyngdpuntens mitt av det som är ritat, fantastiskt igen
        
        # Flytta bilden så att tyngdpunkten hamnar exakt i mitten (13.5, 13.5, ibliden 0-27 dvs 28 pixlar)
        if not np.isnan(cy) and not np.isnan(cx):
            shift_y = 13.5 - cy
            shift_x = 13.5 - cx
            img_np = shift(img_np, [shift_y, shift_x]) # Vart skall bilden flyttad så att den är i mitten
        # shiften kan stöka till pixelvärdena så att dom hamnar utanför 0-255, img_np är en float array
        
        # Image.fromarray(img_np.astype(np.uint8)).save("C:\\Users\\ulfha\\OneDrive\\Bilder\\AI\\step4.png")
       
        # Säkerställ att pixelvärdena håller sig inom 0-255 efter shift
        img_final_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Image.fromarray(img_final_np).save("C:\\Users\\ulfha\\OneDrive\\Bilder\\AI\\step5.png")
        
        # Inför prediktion (Platta ut till 784 kolumner)
        final_input = img_final_np.reshape(1, 784)
        
        # Använd Scalern 
        final_input_scaled = scaler.transform(final_input)
        
        # Hämta sannolikheter
        probs = model.predict_proba(final_input_scaled)[0]
        
        # Skapa dict med resultat
        
        label_output = {}
        for i in range(10):
            current_Ynum = str(i)
            propability = float(probs[i])
            label_output[current_Ynum] = propability
    
        return img_final_np, label_output

    return None, {}
   

# GRÄNSSNITT

st.set_page_config(page_title="MNIST Labration", layout="centered")
st.title("Siffer-igenkänning ")
st.caption(f"Modellens noggrannhet: {accuracy:.2%}")

col1, col2, col3 = st.columns([2, 1, 1.5])

with col1:
    stroke_width = st.sidebar.slider("Pen storlek: ", 1, 25, 15)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Siffror från MNIST-databasen")
    digit_to_show = st.sidebar.selectbox("Välj en siffra att granska:", range(10))
    
    if st.sidebar.button(f"Visa exempel på {digit_to_show}"):
        # Hitta alla index för den valda siffran
        digit_indices = np.where(y_raw.astype(str) == str(digit_to_show))[0]
        
        # Slumpa fram 5 index för att göra det lite roligare varje gång man klickar
        sample_indices = np.random.choice(digit_indices, 5, replace=False)
        
        st.sidebar.write(f"Exempel på {digit_to_show}:or:")
        
        # Visa bilderna i sidomenyn i 5 kolumner
        side_cols = st.sidebar.columns(5)
        for i, idx in enumerate(sample_indices):
            #  X_raw är en numpy-array, så ingen .iloc här som i PD
            img = X_raw[idx].reshape(28, 28)
            
            with side_cols[i]:
                st.image(img, width=60)
            
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
        update_streamlit=True # Gör att den uppdateras när man ritat och släpper musen
    )
    # canvas_result.image_data är en numpy-array med RGBA-data (280x280x4)


# Prediktion 
processed_img = None
predictions = None
if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] < 255):
    # image_data är en numpy-array med RGBA-data (280x280x4) som kommer från canvasen
    processed_img, predictions = predict_digit(canvas_result.image_data)
    
# Kolla om användaren ritar något
    with col2:
        st.markdown("### 2. Input")
        # Kontrollera att variabeln inte är None
        if processed_img is not None:
            st.image(processed_img, width=120, caption="AI-vy (28x28)")
        else:
            st.caption("Ingen bild bearbetad")

    with col3:
        st.markdown("### 3. Analys")
        if predictions: # En dict med 
            best_guess = max(predictions, key=predictions.get)
            confidence = predictions[best_guess]
            st.metric("Gissning", f"Siffra {best_guess}", f"{confidence:.1%}")
            st.bar_chart(predictions)

    #  Galleri under kolumnerna 
    
    st.markdown("---")
    st.subheader(f"Tolkat resultat som '{best_guess}:or' av modellen från MNIST-databasen")

    # y_raw är en numpy-array (tack vare as_frame=False)
    # använd .astype(str) för att säkerställa att jämförelsen med best_guess funkar
    all_indices = np.where(y_raw.astype(str) == str(best_guess))[0]
    indices = all_indices[:10] # Dom 10 första indexen där modellen tror att det är best_guess 

    if len(indices) > 0:
        img_cols = st.columns(5)
        for i, idx in enumerate(indices):
            with img_cols[i % 5]:
                # X_test är en numpy-array
                # Vi plockar raden idx och formar om till 28x28
                sample_img = X_raw[idx].reshape(28, 28)
                st.image(sample_img, width=60)
    else:
        st.warning(f"Kunde inte hitta några referensbilder för siffra {best_guess}.")

else:
    with col3:
        st.info("Väntar på bilden...")