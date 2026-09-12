# ============================================================
# AUTOMATED ROAD DAMAGE CLASSIFICATION
# Feature-Rich Streamlit Application
# MobileNetV2 + Grad-CAM
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import Model


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Damage AI",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .recommendation-box {
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #ffb300;
        background-color: #fff8e1;
        margin: 10px 0 20px 0;
    }

    .footer {
        text-align: center;
        color: #777777;
        font-size: 14px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #dddddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = r"C:\Users\Devendra\Automated Road Damage Classification"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "road_damage_final.keras"
)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Crack",
    "Manhole",
    "Pothole"
]


# ============================================================
# 4. DAMAGE INFORMATION
# ============================================================

DAMAGE_INFO = {

    "Crack": {
        "icon": "⚠️",
        "severity": "Moderate",
        "description":
            "Surface cracking has been detected on the road.",
        "recommendation":
            "Inspect the crack and apply suitable sealing or surface repair to prevent further deterioration."
    },

    "Manhole": {
        "icon": "🕳️",
        "severity": "High",
        "description":
            "A manhole-related road feature or defect has been detected.",
        "recommendation":
            "Inspect and secure the manhole area to ensure road-user safety and proper infrastructure maintenance."
    },

    "Pothole": {
        "icon": "🚧",
        "severity": "High",
        "description":
            "A pothole has been detected on the road surface.",
        "recommendation":
            "Schedule road inspection and repair as soon as possible because potholes can create significant safety risks."
    }
}


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:

        loaded_model = tf.keras.models.load_model(
            MODEL_PATH
        )

        return loaded_model

    except Exception as e:

        st.error(
            f"Unable to load model: {e}"
        )

        return None


model = load_model()


# ============================================================
# 6. FIND MOBILENETV2 BASE MODEL
# ============================================================

def get_base_model(model):

    """
    Your trained model has approximately this structure:

    0 -> MobileNetV2
    1 -> GlobalAveragePooling2D
    2 -> Dense
    3 -> Dropout
    4 -> Dense

    This function safely finds the nested CNN.
    """

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            return layer

    return None


# ============================================================
# 7. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def get_last_conv_layer(base_model):

    """
    Searches inside MobileNetV2 for the last
    convolutional-type layer.
    """

    if base_model is None:
        return None

    for layer in reversed(
        base_model.layers
    ):

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D
            )
        ):

            return layer

    return None


# ============================================================
# 8. PREDICTION FUNCTION
# ============================================================

def predict_image(
    model,
    image
):

    start_time = time.perf_counter()

    image = image.convert(
        "RGB"
    )

    resized_image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        resized_image
    ).astype(
        "float32"
    )

    # Same preprocessing used during training
    image_array = (
        image_array / 255.0
    )

    model_input = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        model_input,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(
            predictions
        )
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[
            predicted_index
        ]
    )

    latency = (
        time.perf_counter()
        - start_time
    ) * 1000

    return (
        predicted_class,
        confidence,
        predictions,
        latency,
        resized_image,
        model_input
    )


# ============================================================
# 9. FIXED GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_array
):

    try:

        # ----------------------------------------------------
        # Find MobileNetV2 base model
        # ----------------------------------------------------

        base_model = get_base_model(
            model
        )

        if base_model is None:

            print(
                "Grad-CAM: Base model not found."
            )

            return None

        # ----------------------------------------------------
        # Find last convolutional layer
        # ----------------------------------------------------

        last_conv_layer = (
            get_last_conv_layer(
                base_model
            )
        )

        if last_conv_layer is None:

            print(
                "Grad-CAM: Last convolutional layer not found."
            )

            return None

        # ----------------------------------------------------
        # Build a model that outputs:
        #
        # 1. Last convolutional feature maps
        # 2. Base model output
        # ----------------------------------------------------

        grad_model = Model(
            inputs=base_model.input,
            outputs=[
                last_conv_layer.output,
                base_model.output
            ]
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, base_output = (
                grad_model(
                    image_array
                )
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # MobileNetV2 base output is feature maps.
            # The classifier is outside the base model.
            #
            # Therefore manually pass the feature maps
            # through the remaining classifier layers.
            # ------------------------------------------------

            x = base_output

            for layer in model.layers[1:]:

                x = layer(x)

            predictions = x

            predicted_index = tf.argmax(
                predictions[0]
            )

            loss = predictions[
                :,
                predicted_index
            ]

        # ----------------------------------------------------
        # Calculate gradients
        # ----------------------------------------------------

        gradients = tape.gradient(
            loss,
            conv_outputs
        )

        if gradients is None:

            print(
                "Grad-CAM: Gradients are None."
            )

            return None

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(
                0,
                1,
                2
            )
        )

        conv_outputs = conv_outputs[0]

        # ----------------------------------------------------
        # Weight feature maps
        # ----------------------------------------------------

        heatmap = (
            conv_outputs
            @ pooled_gradients[
                ...,
                tf.newaxis
            ]
        )

        heatmap = tf.squeeze(
            heatmap
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        heatmap = tf.maximum(
            heatmap,
            0
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        max_value = tf.reduce_max(
            heatmap
        )

        if float(
            max_value
        ) > 0:

            heatmap = (
                heatmap
                / max_value
            )

        heatmap = heatmap.numpy()

        # ----------------------------------------------------
        # Resize heatmap
        # ----------------------------------------------------

        heatmap_image = Image.fromarray(
            np.uint8(
                heatmap * 255
            )
        )

        heatmap_image = heatmap_image.resize(
            IMG_SIZE,
            Image.Resampling.BILINEAR
        )

        heatmap = (
            np.array(
                heatmap_image
            ).astype(
                "float32"
            ) / 255.0
        )

        return heatmap

    except Exception as e:

        print(
            "Grad-CAM Error:",
            repr(e)
        )

        return None


# ============================================================
# 10. CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap
):

    original_image = (
        original_image
        .convert("RGB")
        .resize(
            IMG_SIZE
        )
    )

    original_array = np.array(
        original_image
    )

    # --------------------------------------------------------
    # Jet heatmap
    # --------------------------------------------------------

    cmap = plt.get_cmap(
        "jet"
    )

    colored_heatmap = cmap(
        heatmap
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    heatmap_image = Image.fromarray(
        colored_heatmap
    )

    original_image = Image.fromarray(
        original_array
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = Image.blend(
        original_image,
        heatmap_image,
        alpha=0.45
    )

    return overlay


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🚧 Road Damage AI"
    )

    st.markdown("---")

    st.subheader(
        "📌 About"
    )

    st.write(
        """
        AI-powered road damage classification
        using deep learning and computer vision.
        """
    )

    st.subheader(
        "🔍 Supported Classes"
    )

    st.write(
        "⚠️ Crack"
    )

    st.write(
        "🕳️ Manhole"
    )

    st.write(
        "🚧 Pothole"
    )

    st.markdown("---")

    st.subheader(
        "🤖 Model"
    )

    st.write(
        "MobileNetV2"
    )

    st.write(
        "Transfer Learning"
    )

    st.write(
        "Input: 224 × 224"
    )

    st.write(
        "TensorFlow / Keras"
    )

    st.markdown("---")

    st.subheader(
        "🔥 Explainability"
    )

    st.write(
        "Grad-CAM"
    )

    st.markdown("---")

    st.subheader(
        "🏙️ Application"
    )

    st.write(
        "Smart Cities"
    )

    st.write(
        "Road Maintenance"
    )

    st.write(
        "Infrastructure Analytics"
    )


# ============================================================
# 12. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="title">'
    '🚧 Automated Road Damage Classification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered road damage detection using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 13. MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "❌ Trained model was not found."
    )

    st.info(
        f"""
        Expected model:

        `{MODEL_PATH}`
        """
    )

    st.stop()


# ============================================================
# 14. IMAGE UPLOAD
# ============================================================

st.subheader(
    "📤 Upload Road Image"
)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a clear road image for classification."
)


# ============================================================
# 15. PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    (
        predicted_class,
        confidence,
        probabilities,
        latency,
        resized_image,
        model_input
    ) = predict_image(
        model,
        image
    )

    damage = DAMAGE_INFO[
        predicted_class
    ]

    confidence_percent = (
        confidence * 100
    )


    # ========================================================
    # 16. IMAGE + PREDICTION
    # ========================================================

    image_col, result_col = (
        st.columns(
            [1, 1]
        )
    )

    with image_col:

        st.subheader(
            "🖼️ Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"Original size: "
            f"{image.size[0]} × "
            f"{image.size[1]} pixels"
        )


    with result_col:

        st.subheader(
            "🔍 Prediction Result"
        )

        st.markdown(
            f"# {damage['icon']} {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )

        st.progress(
            confidence
        )

        if confidence >= 0.70:

            st.success(
                "✅ High-confidence prediction."
            )

        elif confidence >= 0.50:

            st.warning(
                "⚠️ Moderate-confidence prediction."
            )

        else:

            st.error(
                "❗ Low-confidence prediction."
            )


    # ========================================================
    # 17. KPI SECTION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Prediction Metrics"
    )

    kpi1, kpi2, kpi3, kpi4 = (
        st.columns(4)
    )

    with kpi1:

        st.metric(
            "Predicted Damage",
            predicted_class
        )

    with kpi2:

        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )

    with kpi3:

        st.metric(
            "Severity",
            damage["severity"]
        )

    with kpi4:

        st.metric(
            "Prediction Time",
            f"{latency:.2f} ms"
        )


    # ========================================================
    # 18. DAMAGE ASSESSMENT
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📝 Damage Assessment"
    )

    st.info(
        damage["description"]
    )


    # ========================================================
    # 19. RECOMMENDATION
    # ========================================================

    st.subheader(
        "💡 Recommended Action"
    )

    st.markdown(
        f"""
        <div class="recommendation-box">

        <b>
        {damage["icon"]} {predicted_class}
        </b>

        <br><br>

        {damage["recommendation"]}

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # 20. PROBABILITY ANALYSIS
    # ========================================================

    st.subheader(
        "📈 Class Probability Analysis"
    )

    probability_df = pd.DataFrame(
        {
            "Damage Type":
                CLASS_NAMES,

            "Probability (%)":
                probabilities * 100
        }
    )

    st.bar_chart(
        probability_df.set_index(
            "Damage Type"
        )
    )

    display_df = (
        probability_df
        .copy()
    )

    display_df[
        "Probability (%)"
    ] = display_df[
        "Probability (%)"
    ].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # 21. GRAD-CAM
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔥 Explainable AI — Grad-CAM"
    )

    st.write(
        """
        Grad-CAM highlights the regions of the road image
        that contributed most strongly to the model's prediction.
        """
    )

    with st.spinner(
        "Generating Grad-CAM explanation..."
    ):

        heatmap = generate_gradcam(
            model,
            model_input
        )


    if heatmap is not None:

        overlay = create_gradcam_overlay(
            image,
            heatmap
        )

        grad_col1, grad_col2 = (
            st.columns(2)
        )

        with grad_col1:

            st.markdown(
                "#### Original Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        with grad_col2:

            st.markdown(
                "#### Grad-CAM Attention"
            )

            st.image(
                overlay,
                use_container_width=True
            )

        st.success(
            "🔥 Grad-CAM generated successfully."
        )

        st.caption(
            "Red/yellow regions represent stronger "
            "model attention, while blue regions "
            "represent lower attention."
        )

    else:

        st.warning(
            "Grad-CAM could not be generated for this model."
        )


    # ========================================================
    # 22. MODEL INFORMATION
    # ========================================================

    st.markdown("---")

    with st.expander(
        "🤖 Model Information"
    ):

        info1, info2 = (
            st.columns(2)
        )

        with info1:

            st.write(
                "**Architecture:** MobileNetV2"
            )

            st.write(
                "**Transfer Learning:** Yes"
            )

            st.write(
                "**Input Size:** 224 × 224"
            )

            st.write(
                "**Classes:** 3"
            )

        with info2:

            st.write(
                "**Framework:** TensorFlow / Keras"
            )

            st.write(
                "**Classification:** Multi-class"
            )

            st.write(
                "**Explainability:** Grad-CAM"
            )

            st.write(
                "**Domain:** Smart Infrastructure"
            )


    # ========================================================
    # 23. TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "🔬 Technical Prediction Details"
    ):

        technical_df = pd.DataFrame(
            {
                "Class":
                    CLASS_NAMES,

                "Probability":
                    probabilities,

                "Probability (%)":
                    probabilities * 100
            }
        )

        st.dataframe(
            technical_df.style.format(
                {
                    "Probability":
                        "{:.6f}",

                    "Probability (%)":
                        "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.write(
            f"Original image: {image.size}"
        )

        st.write(
            f"Model input: {IMG_SIZE}"
        )

        st.write(
            f"Prediction latency: {latency:.2f} ms"
        )


    # ========================================================
    # 24. DOWNLOAD REPORT
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📥 Prediction Report"
    )

    report_df = pd.DataFrame(
        {
            "Property": [

                "Predicted Damage",

                "Confidence",

                "Severity",

                "Recommendation",

                "Prediction Latency (ms)",

                "Crack Probability",

                "Manhole Probability",

                "Pothole Probability"
            ],

            "Value": [

                predicted_class,

                f"{confidence_percent:.2f}%",

                damage["severity"],

                damage["recommendation"],

                f"{latency:.2f}",

                f"{probabilities[0] * 100:.2f}%",

                f"{probabilities[1] * 100:.2f}%",

                f"{probabilities[2] * 100:.2f}%"
            ]
        }
    )

    report_csv = report_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Prediction Report",
        data=report_csv,
        file_name="road_damage_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # 25. AI PIPELINE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔄 AI Processing Pipeline"
    )

    p1, p2, p3, p4, p5 = (
        st.columns(5)
    )

    with p1:

        st.write("📤")
        st.write("Image Upload")

    with p2:

        st.write("🖼️")
        st.write("Preprocessing")

    with p3:

        st.write("🧠")
        st.write("MobileNetV2")

    with p4:

        st.write("🔍")
        st.write("Classification")

    with p5:

        st.write("🔥")
        st.write("Grad-CAM")


    # ========================================================
    # 26. ANALYZE ANOTHER IMAGE
    # ========================================================

    st.markdown("---")

    if st.button(
        "🔄 Analyze Another Image",
        use_container_width=True
    ):

        st.rerun()


# ============================================================
# 27. LANDING PAGE
# ============================================================

else:

    st.markdown("---")

    st.subheader(
        "🛣️ How It Works"
    )

    step1, step2, step3, step4 = (
        st.columns(4)
    )

    with step1:

        st.markdown(
            """
            ### 1️⃣ Upload

            Upload a road image
            from your device.
            """
        )

    with step2:

        st.markdown(
            """
            ### 2️⃣ Preprocess

            The image is resized
            to 224 × 224 and normalized.
            """
        )

    with step3:

        st.markdown(
            """
            ### 3️⃣ Classify

            MobileNetV2 predicts
            Crack, Manhole or Pothole.
            """
        )

    with step4:

        st.markdown(
            """
            ### 4️⃣ Explain

            Grad-CAM highlights
            important image regions.
            """
        )


    st.markdown("---")

    st.subheader(
        "🎯 Project Objectives"
    )

    obj1, obj2, obj3 = (
        st.columns(3)
    )

    with obj1:

        st.info(
            """
            **Automated Detection**

            Reduce manual road inspection
            using AI-based image classification.
            """
        )

    with obj2:

        st.info(
            """
            **Explainable AI**

            Grad-CAM helps visualize
            the regions influencing predictions.
            """
        )

    with obj3:

        st.info(
            """
            **Smart Maintenance**

            Damage-specific recommendations
            support road maintenance decisions.
            """
        )


# ============================================================
# 28. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🚧 <b>
    Automated Road Damage Classification Using Deep Learning
    </b>

    <br><br>

    Computer Vision |
    Deep Learning |
    Smart Cities |
    Infrastructure Analytics

    <br><br>

    Built with Python • TensorFlow • Keras • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)