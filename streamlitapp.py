import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile

file_path = "skindataall.csv"
try:
    skin_data = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Dataset file not found. Please check the file path.")
    st.stop()

skin_data = skin_data.rename(columns={
    "Product": "name",
    "Brand": "brand",
    "Skin_Type": "skin_type",
    "Category": "category",
    "Ingredients_Cleaned": "key_ingredients",
    "Product_Url": "url",
    "Good_Stuff": "good_stuff",
    "Rating_Stars": "rating"
})
skin_data["key_ingredients"] = skin_data["key_ingredients"].apply(lambda x: eval(x) if isinstance(x, str) else [])
skin_data["skin_type"] = skin_data["skin_type"].apply(lambda x: x.lower() if isinstance(x, str) else "all")
product_list = skin_data.to_dict(orient="records")

skin_concerns = ["acne", "dark circles", "dryness", "redness", "pores", "oiliness", "sensitivity", "hyperpigmentation", "wrinkles"]
ingredients = sorted(set(ingredient for product in product_list for ingredient in product["key_ingredients"]))

def recommend_products(skin_type, concerns):
    concerns = [c.strip().lower() for c in concerns]
    filtered = [
        product for product in product_list
        if (skin_type in product["skin_type"] or product["skin_type"] == "all")
        and any(concern in product["key_ingredients"] for concern in concerns)
        and product["good_stuff"] == 1
    ]
    if not filtered:
        fallback = [
            product for product in product_list
            if any(concern in product["key_ingredients"] for concern in concerns)
            and product["good_stuff"] == 1
        ]
        if fallback:
            return "No products for the selected skin type. Showing results for all skin types:", fallback
        else:
            return "No products found for the given concerns.", []
    return "Here are your recommendations:", filtered

def detect_skin_concerns_from_image(image_path):
    concerns_detected = set()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    roi_dark_circles = img[100:200, 150:300]
    if np.mean(roi_dark_circles) < 80:
        concerns_detected.add("dark circles")
    edges = cv2.Canny(img, 100, 200)
    acne_area = edges[200:300, 100:200]
    if np.sum(acne_area) > 1000:
        concerns_detected.add("acne")
    return list(concerns_detected)

# ---------------- Streamlit UI ----------------
st.title("Skincare Recommendation System")

st.subheader("Supported Skin Concerns")
st.write(", ".join(skin_concerns))


skin_type = st.selectbox("Select Skin Type", ["all", "oily", "dry", "normal", "combination"])
st.subheader("📸 Detect Concerns Using Webcam")
if st.button("Open Camera"):
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(camera_image.read())
            detected = detect_skin_concerns_from_image(temp_file.name)

        st.success(f"Detected Concerns: {', '.join(detected) if detected else 'None'}")


st.subheader("📂 Upload Image to Detect Concerns")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        detected = detect_skin_concerns_from_image(temp_file.name)
    st.success(f"Detected Concerns: {', '.join(detected) if detected else 'None'}")
concerns_input = st.text_input("Enter Concerns (comma-separated)")

if st.button("Get Direct Recommendations"):
    if concerns_input.strip():
        message, recommendations = recommend_products(skin_type, concerns_input.split(","))
        st.subheader(message)
        if recommendations:
            for product in recommendations:
                st.markdown(
                    f"**{product['name']}** by *{product['brand']}* ({product['category']})\n\n"
                    f"⭐ Rating: {product['rating']}\n\n"
                    f"[🔗 Product Link]({product['url']})"
                )
        else:
            st.warning("Try different criteria for better results.")
    else:
        st.warning("Please enter concerns!")
