import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import cv2
import numpy as np

# Load the dataset
file_path = r'C:/Users/KASMYA/OneDrive/Documents/comp science/skindataall.csv'
try:
    skin_data = pd.read_csv(file_path)
except FileNotFoundError:
    messagebox.showerror("Error", "Dataset file not found. Please check the file path.")
    exit()

# Preprocess dataset
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

# Define supported skin concerns and ingredients
skin_concerns = ["acne", "dark circles", "dryness", "redness", "pores", "oiliness", "sensitivity", "hyperpigmentation", "wrinkles"]
ingredients = sorted(set(ingredient for product in product_list for ingredient in product["key_ingredients"]))

# Recommendation engine
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

# Webcam-based detection
def detect_skin_concerns_from_webcam():
    concerns_detected = set()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Skin Concern Detection (Press Q to Stop)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_dark_circles = gray[100:200, 150:300]
        if np.mean(roi_dark_circles) < 80:
            concerns_detected.add("dark circles")
        edges = cv2.Canny(gray, 100, 200)
        acne_area = edges[200:300, 100:200]
        if np.sum(acne_area) > 1000:
            concerns_detected.add("acne")
        cv2.putText(frame, f"Detected: {', '.join(concerns_detected)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Skin Concern Detection (Press Q to Stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return list(concerns_detected)

# Image-based detection
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

# GUI Functionality
def display_recommendations():
    user_skin_type = skin_type_var.get().lower()
    user_concerns = concerns_var.get()
    
    if not user_skin_type or not user_concerns:
        messagebox.showwarning("Input Error", "Please fill in all fields!")
        return
    
    message, recommendations = recommend_products(user_skin_type, user_concerns.split(","))
    
    for widget in results_frame.winfo_children():
        widget.destroy()
    
    ttk.Label(results_frame, text=message, font=("Helvetica", 12, "bold"), background="#FFC0CB").pack(pady=10)
    
    if recommendations:
        for product in recommendations:
            ttk.Label(
                results_frame,
                text=f"{product['name']} by {product['brand']} ({product['category']})\n"
                     f"Rating: {product['rating']} | URL: {product['url']}",
                font=("Helvetica", 10),
                wraplength=500,
                anchor="w",
                background="#FFC0CB"
            ).pack(fill="x", padx=10, pady=5)
    else:
        ttk.Label(results_frame, text="Try different criteria for better results.", font=("Helvetica", 10, "italic"), background="#FFC0CB").pack(pady=10)

def use_webcam_for_concerns():
    detected_concerns = detect_skin_concerns_from_webcam()
    concerns_var.set(", ".join(detected_concerns))
    messagebox.showinfo("Detected Concerns", f"Detected concerns: {', '.join(detected_concerns)}")

def upload_image_for_concerns():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        detected_concerns = detect_skin_concerns_from_image(file_path)
        concerns_var.set(", ".join(detected_concerns))
        messagebox.showinfo("Detected Concerns", f"Detected concerns: {', '.join(detected_concerns)}")

# GUI Setup
root = tk.Tk()
root.title("Skincare Recommendation System")
root.geometry("800x800")
root.configure(bg="#FFC0CB")

# Title
tk.Label(root, text="Skincare Recommendation System", font=("Helvetica", 18, "bold"), bg="#FFC0CB").pack(pady=10)

# Supported Concerns and Ingredients
tk.Label(root, text="Supported Skin Concerns:", font=("Helvetica", 12, "bold"), bg="#FFC0CB", fg="black").pack(pady=5)
tk.Label(root, text=", ".join(skin_concerns), font=("Helvetica", 10), bg="#FFC0CB", fg="black").pack(pady=5)

tk.Label(root, text="Key Ingredients:", font=("Helvetica", 12, "bold"), bg="#FFC0CB", fg="black").pack(pady=5)
tk.Label(root, text=", ".join(ingredients), font=("Helvetica", 10), bg="#FFC0CB", fg="black").pack(pady=5)

# Inputs for direct recommendation
skin_type_label = tk.Label(root, text="Skin Type (e.g., oily, dry, all):", font=("Helvetica", 12), bg="#FFC0CB", fg="black")
skin_type_label.pack(pady=5)
skin_type_var = tk.StringVar(value="all")
skin_type_dropdown = ttk.Combobox(root, textvariable=skin_type_var, values=["all", "oily", "dry", "normal", "combination"])
skin_type_dropdown.pack(pady=5)

concerns_label = tk.Label(root, text="Concerns (comma-separated):", font=("Helvetica", 12), bg="#FFC0CB", fg="black")
concerns_label.pack(pady=5)
concerns_var = tk.StringVar()
concerns_entry = tk.Entry(root, textvariable=concerns_var, width=50)
concerns_entry.pack(pady=5)

# Buttons for modes
tk.Button(root, text="Direct Recommendation", command=display_recommendations, bg="#FF69B4", fg="white", font=("Helvetica", 12)).pack(pady=10)
tk.Button(root, text="Use Webcam", command=use_webcam_for_concerns, bg="#FF69B4", fg="white", font=("Helvetica", 12)).pack(pady=10)
tk.Button(root, text="Upload Image", command=upload_image_for_concerns, bg="#FF69B4", fg="white", font=("Helvetica", 12)).pack(pady=10)

# Results frame
results_frame = tk.Frame(root, bg="#FFC0CB")
results_frame.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()
