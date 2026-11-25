
import streamlit as st
import numpy as np
import joblib
import re

from PIL import Image
from skimage.feature import hog

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="Skin Disease AI Information System",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Skin Disease Classification & Information System")
st.subheader("Class 12 Artificial Intelligence Project")

st.write(
    "Upload a skin image to get an experimental AI prediction "
    "and learn about the possible causes, symptoms, prevention "
    "and general treatment information."
)

st.warning(
    "Educational project only. AI predictions are not medical "
    "diagnoses. Always consult a qualified dermatologist."
)

# -----------------------------------
# DISEASE INFORMATION DATABASE
# -----------------------------------

disease_info = {

    "acne": {
        "description": "Acne is a common skin condition in which hair follicles become clogged with oil and dead skin cells.",
        "causes": [
            "Excess oil production",
            "Blocked hair follicles",
            "Hormonal changes",
            "Genetic factors"
        ],
        "symptoms": [
            "Pimples",
            "Blackheads and whiteheads",
            "Redness and inflammation",
            "Sometimes painful bumps"
        ],
        "spread": "Acne is not contagious.",
        "prevention": [
            "Wash skin gently",
            "Avoid squeezing pimples",
            "Use non-comedogenic skincare products"
        ],
        "treatment": "Treatment depends on severity. Dermatologists may recommend suitable topical or other treatments."
    },

    "cellulitis": {
        "description": "Cellulitis is a bacterial infection affecting deeper layers of the skin and underlying tissues.",
        "causes": [
            "Bacteria entering through cuts or wounds",
            "Cracked or damaged skin",
            "Insect bites or other skin injuries"
        ],
        "symptoms": [
            "Red or discoloured skin",
            "Swelling",
            "Pain and tenderness",
            "Warm skin and sometimes fever"
        ],
        "spread": "It is not usually spread from person to person.",
        "prevention": [
            "Clean cuts and wounds",
            "Keep wounds covered",
            "Maintain good skin hygiene"
        ],
        "treatment": "Requires prompt medical assessment and usually prescription antibiotics. Rapidly spreading redness, fever or severe pain needs urgent care."
    },

    "dark spots": {
        "description": "Dark spots or hyperpigmentation are areas where the skin becomes darker due to increased melanin.",
        "causes": [
            "Sun exposure",
            "Hormonal changes",
            "Previous acne or skin inflammation",
            "Certain medicines or skin injuries"
        ],
        "symptoms": [
            "Brown or dark patches",
            "Uneven skin tone",
            "Flat areas of darker pigmentation"
        ],
        "spread": "Hyperpigmentation is not contagious.",
        "prevention": [
            "Use sun protection",
            "Avoid picking acne or skin injuries",
            "Use gentle skincare products"
        ],
        "treatment": "Treatment depends on the cause. A dermatologist can assess new or changing dark spots and recommend appropriate care."
    },

    "eczema": {
        "description": "Eczema is a group of conditions that cause dry, itchy and inflamed skin.",
        "causes": [
            "Genetic and skin-barrier factors",
            "Immune system activity",
            "Irritants such as soaps and detergents",
            "Environmental triggers"
        ],
        "symptoms": [
            "Dry skin",
            "Intense itching",
            "Red or discoloured patches",
            "Sometimes cracked skin"
        ],
        "spread": "Eczema is not contagious.",
        "prevention": [
            "Moisturise regularly",
            "Avoid known irritants",
            "Use gentle cleansers",
            "Avoid very hot showers"
        ],
        "treatment": "Moisturisers and dermatologist-recommended medicines may help control symptoms."
    },

    "herpes": {
        "description": "Herpes simplex is a viral infection that can cause painful blisters or sores around the mouth or genital area.",
        "causes": [
            "Herpes simplex virus type 1 (HSV-1)",
            "Herpes simplex virus type 2 (HSV-2)",
            "Direct contact with infected skin or sores"
        ],
        "symptoms": [
            "Small fluid-filled blisters",
            "Pain or burning",
            "Itching or tingling",
            "Recurring sores in some people"
        ],
        "spread": "Can spread through direct skin-to-skin contact, including kissing or sexual contact, depending on the infection site.",
        "prevention": [
            "Avoid direct contact with active sores",
            "Do not share items that contact active sores",
            "Follow medical advice about reducing transmission"
        ],
        "treatment": "Antiviral medicines prescribed by a healthcare professional can help manage outbreaks."
    },

    "impetigo": {
        "description": "Impetigo is a contagious bacterial skin infection that commonly affects children.",
        "causes": [
            "Staphylococcus aureus bacteria",
            "Streptococcus pyogenes bacteria",
            "Bacteria entering through damaged skin"
        ],
        "symptoms": [
            "Red sores",
            "Blisters",
            "Yellow or honey-coloured crusts",
            "Itching"
        ],
        "spread": "It can spread through direct contact or contaminated towels and other personal items.",
        "prevention": [
            "Wash hands regularly",
            "Keep affected areas clean and covered",
            "Avoid sharing towels",
            "Avoid scratching sores"
        ],
        "treatment": "A healthcare professional may prescribe antibiotic treatment depending on the extent of infection."
    },

    "normal skin": {
        "description": "Normal skin is a dataset category representing skin without the particular abnormality being classified.",
        "causes": [
            "Normal skin appearance varies naturally",
            "Skin characteristics depend on genetics and environment"
        ],
        "symptoms": [
            "No obvious abnormality in the examined image",
            "Generally healthy-looking skin appearance"
        ],
        "spread": "Normal skin is not contagious.",
        "prevention": [
            "Maintain gentle skin hygiene",
            "Use sun protection",
            "Keep skin moisturised"
        ],
        "treatment": "No disease treatment is required for healthy skin. A normal AI prediction does not rule out a skin condition."
    },

    "psoriasis": {
        "description": "Psoriasis is a chronic inflammatory skin condition that causes skin cells to build up too quickly.",
        "causes": [
            "Immune system dysfunction",
            "Genetic factors",
            "Certain infections or stress can trigger flare-ups",
            "Some medicines may trigger symptoms"
        ],
        "symptoms": [
            "Thick patches of skin",
            "Scaly plaques",
            "Itching or discomfort",
            "Cracked skin in some cases"
        ],
        "spread": "Psoriasis is not contagious.",
        "prevention": [
            "Identify personal triggers",
            "Moisturise skin",
            "Avoid skin injuries where possible",
            "Follow medical advice"
        ],
        "treatment": "There are several dermatologist-supervised treatments to control inflammation and symptoms."
    },

    "ringworm": {
        "description": "Ringworm is a fungal skin infection. Despite its name, it is not caused by a worm.",
        "causes": [
            "Dermatophyte fungi",
            "Contact with infected people or animals",
            "Contaminated towels or clothing",
            "Warm and moist environments"
        ],
        "symptoms": [
            "Ring-shaped rash",
            "Itching",
            "Scaly skin",
            "Red or discoloured patches"
        ],
        "spread": "It can spread through infected people, animals or contaminated objects.",
        "prevention": [
            "Keep skin clean and dry",
            "Avoid sharing towels",
            "Wear clean clothing",
            "Avoid contact with infected areas"
        ],
        "treatment": "Antifungal treatment is commonly used. A healthcare professional can advise on the appropriate treatment."
    },

    "rosacea": {
        "description": "Rosacea is a long-term skin condition that commonly affects the face.",
        "causes": [
            "The exact cause is not fully understood",
            "Genetic and immune factors may contribute",
            "Sunlight, heat and other triggers can worsen symptoms"
        ],
        "symptoms": [
            "Facial redness",
            "Flushing",
            "Visible small blood vessels",
            "Sometimes acne-like bumps"
        ],
        "spread": "Rosacea is not contagious.",
        "prevention": [
            "Use sun protection",
            "Identify and avoid personal triggers",
            "Use gentle skincare products"
        ],
        "treatment": "A dermatologist can recommend skincare, medicines or other treatments according to symptoms."
    },

    "scabies": {
        "description": "Scabies is a contagious skin infestation caused by tiny mites that burrow into the skin.",
        "causes": [
            "Infestation by Sarcoptes scabiei mites",
            "Prolonged direct skin contact with an infected person",
            "Sometimes contact with contaminated bedding or clothing"
        ],
        "symptoms": [
            "Intense itching, often worse at night",
            "Small red or discoloured bumps",
            "Thin burrow marks",
            "Rash around fingers and wrists"
        ],
        "spread": "Usually spreads through prolonged direct skin contact. It can also spread through shared bedding or clothing.",
        "prevention": [
            "Avoid close contact until treated",
            "Follow advice about treating close contacts",
            "Wash and appropriately handle bedding and clothing"
        ],
        "treatment": "Requires specific prescription or recommended anti-mite treatment. Close contacts may also need treatment."
    },

    "urticaria": {
        "description": "Urticaria, also called hives, is a skin reaction that causes raised, itchy welts.",
        "causes": [
            "Allergic reactions",
            "Infections",
            "Certain medicines",
            "Physical triggers such as heat, cold or pressure"
        ],
        "symptoms": [
            "Raised itchy welts",
            "Red or skin-coloured bumps",
            "Swelling",
            "Welts that change location or disappear"
        ],
        "spread": "Urticaria is not contagious.",
        "prevention": [
            "Identify personal triggers",
            "Avoid known allergens",
            "Seek advice about recurring episodes"
        ],
        "treatment": "Treatment depends on the cause. Healthcare professionals may recommend suitable antihistamines or other care."
    },

    "vitiligo": {
        "description": "Vitiligo is a condition in which the skin loses pigment, creating lighter or white patches.",
        "causes": [
            "Immune system attacks pigment-producing cells",
            "Genetic factors may contribute",
            "Other factors are still being studied"
        ],
        "symptoms": [
            "White or lighter skin patches",
            "Loss of skin pigmentation",
            "Sometimes changes in hair colour"
        ],
        "spread": "Vitiligo is not contagious.",
        "prevention": [
            "Protect affected areas from sunburn",
            "Use sun protection",
            "Consult a dermatologist about skin changes"
        ],
        "treatment": "Treatment options may include prescribed topical medicines, light therapy or other specialist care."
    }
}

# -----------------------------------
# NORMALIZE MODEL PREDICTION LABELS
# -----------------------------------

def normalize_label(label):
    label = str(label).lower().strip()
    label = re.sub(r"[_-]+", " ", label)
    label = re.sub(r"\s+", " ", label)

    aliases = {
        "acne vulgaris": "acne",
        "dark spots / hyperpigmentation": "dark spots",
        "hyperpigmentation": "dark spots",
        "dark spots hyperpigmentation": "dark spots",
        "atopic dermatitis": "eczema",
        "herpes simplex": "herpes",
        "healthy skin": "normal skin",
        "normal": "normal skin",
        "tinea": "ringworm",
        "tinea corporis": "ringworm",
        "hives": "urticaria",
        "urticaria hives": "urticaria"
    }

    return aliases.get(label, label)


# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("skin_disease_model.pkl")


# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------

st.header("📸 Upload Skin Image")

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Skin Image",
        width="stretch"
    )

    if st.button("🔍 Classify Skin Image", type="primary"):

        try:
            model = load_model()

            resized_image = image.resize((128, 128))

            features = hog(
                np.array(resized_image),
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                channel_axis=-1
            )

            features = features.reshape(1, -1)

            prediction = model.predict(features)[0]

            disease_name = str(prediction)
            key = normalize_label(disease_name)

            st.success("Prediction completed!")

            st.header("🧠 AI Prediction")
            st.subheader(disease_name)

            st.warning(
                "This is an experimental image classification result. "
                "It is not a confirmed medical diagnosis."
            )

            # -----------------------------------
            # DISPLAY DISEASE INFORMATION
            # -----------------------------------

            if key in disease_info:

                info = disease_info[key]

                st.divider()

                st.header("📚 About the Disease")

                st.write(info["description"])

                st.subheader("🔬 Possible Causes")

                for cause in info["causes"]:
                    st.markdown(f"- {cause}")

                st.subheader("⚠️ Common Symptoms")

                for symptom in info["symptoms"]:
                    st.markdown(f"- {symptom}")

                st.subheader("🔄 Is it contagious?")

                st.write(info["spread"])

                st.subheader("🛡️ Prevention and Care")

                for item in info["prevention"]:
                    st.markdown(f"- {item}")

                st.subheader("💊 General Treatment Information")

                st.write(info["treatment"])

            else:

                st.info(
                    "Detailed information is not yet available for "
                    f"the model category '{disease_name}'. "
                    "Please check that the disease name matches "
                    "the information database."
                )

            st.divider()

            st.subheader("👨‍⚕️ Medical Advice")

            st.write(
                "Consult a qualified dermatologist for proper "
                "examination, diagnosis and treatment. Seek urgent "
                "medical care for rapidly spreading redness, fever, "
                "severe pain or breathing difficulty."
            )

        except FileNotFoundError:
            st.error(
                "Model file not found. Please train your model first."
            )

        except Exception as error:
            st.error(f"An error occurred: {error}")


# -----------------------------------
# PROJECT FOOTER
# -----------------------------------

st.divider()

st.subheader("About This Project")

st.write(
    "This application is developed as a Class 12 Artificial "
    "Intelligence project. It combines machine learning image "
    "classification with an educational skin disease information "
    "database."
)

st.caption(
    "Developed for educational purposes. "
    "Information is not a substitute for professional medical advice."
)
