import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import cv2
import numpy as np
import webbrowser
import os
import io

# ---------------------------
# Load dataset (original behavior preserved)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'skindataall.csv')

try:
    skin_data = pd.read_csv(file_path)
except FileNotFoundError:
    messagebox.showerror("Error", f"Dataset file not found at: {file_path}")
    raise SystemExit(1)
except Exception as e:
    messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
    raise SystemExit(1)

# Keep original column mapping & preprocessing (unchanged)
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

# preserve existing eval behavior for key_ingredients
skin_data["key_ingredients"] = skin_data["key_ingredients"].apply(lambda x: eval(x) if isinstance(x, str) else [])
skin_data["skin_type"] = skin_data["skin_type"].apply(lambda x: x.lower() if isinstance(x, str) else "all")
product_list = skin_data.to_dict(orient="records")

# Supported concerns and ingredients (original)
skin_concerns = ["acne", "dark circles", "dryness", "redness", "pores", "oiliness", "sensitivity", "hyperpigmentation", "wrinkles"]
ingredients = sorted(set(ingredient for product in product_list for ingredient in product["key_ingredients"]))

# ---------------------------
# Concern -> ingredient mapping (enhancement)
# ---------------------------
# This mapping helps the recommender expand a concern into ingredient keywords.
CONCERN_TO_INGREDIENTS = {
    "acne": ["salicylic", "benzoyl", "niacinamide", "tea tree", "sulfur"],
    "dark circles": ["caffeine", "retinol", "vitamin c", "niacinamide", "hyaluronic"],
    "dryness": ["hyaluronic", "glycerin", "shea", "ceramide", "squalane"],
    "redness": ["aloe", "centella", "green tea", "niacinamide", "azelaic"],
    "pores": ["niacinamide", "salicylic", "retinol", "clay"],
    "oiliness": ["salicylic", "clay", "niacinamide", "zinc"],
    "sensitivity": ["aloe", "oat", "centella", "chamomile", "ceramide"],
    "hyperpigmentation": ["vitamin c", "niacinamide", "azelaic", "licorice"],
    "wrinkles": ["retinol", "peptide", "collagen", "vitamin c"]
}

def derive_ingredient_keywords_from_concerns(concerns):
    """Return a set of ingredient keyword strings derived from concerns and dataset ingredients."""
    keywords = set()
    for c in concerns:
        c_norm = c.strip().lower()
        # mapping-based
        for mapped in CONCERN_TO_INGREDIENTS.get(c_norm, []):
            keywords.add(mapped.lower())
        # find ingredients in dataset that contain the concern word (partial match)
        for ing in ingredients:
            if c_norm in ing.lower():
                keywords.add(ing.lower())
    # fallback to the concerns themselves if no mapping found
    if not keywords:
        for c in concerns:
            if c.strip():
                keywords.add(c.strip().lower())
    return keywords

# ---------------------------
# Enhanced recommendation engine (keeps original fallback)
# ---------------------------
def recommend_products(skin_type, concerns,
                       avoid_ingredients=None,
                       brand_filter=None,
                       category_filter=None,
                       sort_by="rating"):
    """
    Enhanced recommender (returns same type of results as original):
    - skin_type: string (e.g., 'oily', 'all')
    - concerns: list of concern strings
    - avoid_ingredients: list of strings to avoid (optional)
    - brand_filter: partial brand name to restrict to (optional)
    - category_filter: partial category to restrict to (optional)
    - sort_by: "rating", "brand", or "relevance"
    Returns: (message, list_of_products)
    """
    # normalize inputs
    concerns = [c.strip().lower() for c in concerns if c and c.strip()]
    avoid_ingredients = [a.strip().lower() for a in (avoid_ingredients or []) if a and a.strip()]
    brand_filter = brand_filter.strip().lower() if isinstance(brand_filter, str) and brand_filter.strip() else None
    category_filter = category_filter.strip().lower() if isinstance(category_filter, str) and category_filter.strip() else None

    # derive ingredient keywords
    target_keywords = derive_ingredient_keywords_from_concerns(concerns)

    # scoring function
    def product_score(product):
        score = 0
        product_ings = [str(x).lower() for x in product.get("key_ingredients", [])]
        # ingredient keyword matches (higher weight)
        for kw in target_keywords:
            if any(kw in ing for ing in product_ings):
                score += 3
        # if concern string appears in product category or name
        prod_category = str(product.get("category", "")).lower()
        prod_name = str(product.get("name", "")).lower()
        for c in concerns:
            if c in prod_category or c in prod_name:
                score += 1
        # brand/category filter: disqualify if filter set and doesn't match
        if brand_filter and brand_filter not in str(product.get("brand", "")).lower():
            return 0
        if category_filter and category_filter not in prod_category:
            return 0
        # avoid ingredients: disqualify if any present
        for a in avoid_ingredients:
            if any(a in ing for ing in product_ings):
                return 0
        return score

    # apply base filters (skin type and good_stuff)
    candidates = []
    for product in product_list:
        prod_skin = product.get("skin_type", "all")
        if not (skin_type in prod_skin or prod_skin == "all"):
            continue
        try:
            if int(product.get("good_stuff", 0)) != 1:
                continue
        except Exception:
            # if not castable, be conservative (skip)
            continue
        sc = product_score(product)
        if sc > 0:
            candidates.append((sc, product))

    # if nothing from enhanced matching -> fallback to original behavior
    if not candidates:
        fallback = []
        for product in product_list:
            try:
                if int(product.get("good_stuff", 0)) != 1:
                    continue
            except Exception:
                continue
            prod_ings = [str(x).lower() for x in product.get("key_ingredients", [])]
            if any(any(kw in ing for ing in prod_ings) for kw in target_keywords):
                fallback.append(product)
        if fallback:
            return "No products for the selected skin type. Showing results for all skin types:", fallback
        else:
            return "No products found for the given concerns.", []

    # sorting
    if sort_by == "rating":
        candidates.sort(key=lambda tup: (tup[0], float(tup[1].get("rating", 0) or 0)), reverse=True)
    elif sort_by == "brand":
        candidates.sort(key=lambda tup: tup[1].get("brand", "").lower())
    elif sort_by == "relevance":
        candidates.sort(key=lambda tup: (tup[0], float(tup[1].get("rating", 0) or 0)), reverse=True)
    else:
        candidates.sort(key=lambda tup: (tup[0], float(tup[1].get("rating", 0) or 0)), reverse=True)

    recommended_products = [tup[1] for tup in candidates]
    return "Here are your recommendations:", recommended_products

# ---------------------------
# Original detection functions (kept, with safe guards)
# ---------------------------
def detect_skin_concerns_from_webcam():
    concerns_detected = set()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Skin Concern Detection (Press Q to Stop)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # protect small frames
        h, w = gray.shape[:2]
        try:
            if h > 200 and w > 300:
                roi_dark_circles = gray[100:200, 150:300]
                if np.mean(roi_dark_circles) < 80:
                    concerns_detected.add("dark circles")
                edges = cv2.Canny(gray, 100, 200)
                acne_area = edges[200:300, 100:200]
                if np.sum(acne_area) > 1000:
                    concerns_detected.add("acne")
        except Exception:
            pass
        cv2.putText(frame, f"Detected: {', '.join(concerns_detected)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Skin Concern Detection (Press Q to Stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return list(concerns_detected)

def detect_skin_concerns_from_image(image_path):
    concerns_detected = set()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    try:
        h, w = img.shape[:2]
        if h > 200 and w > 300:
            roi_dark_circles = img[100:200, 150:300]
            if np.mean(roi_dark_circles) < 80:
                concerns_detected.add("dark circles")
            edges = cv2.Canny(img, 100, 200)
            acne_area = edges[200:300, 100:200]
            if np.sum(acne_area) > 1000:
                concerns_detected.add("acne")
    except Exception:
        pass
    return list(concerns_detected)

# ---------------------------
# GUI - enhanced but non-destructive
# ---------------------------
DEFAULT_BG = "#FFC0CB"
DEFAULT_FG = "black"
DARK_BG = "#2b2b2b"
DARK_FG = "white"

root = tk.Tk()
root.title("Skincare Recommendation System")
root.geometry("950x820")
root.configure(bg=DEFAULT_BG)

# Title & info (kept)
tk.Label(root, text="Skincare Recommendation System", font=("Helvetica", 18, "bold"), bg=DEFAULT_BG).pack(pady=10)
tk.Label(root, text="Supported Skin Concerns:", font=("Helvetica", 12, "bold"), bg=DEFAULT_BG, fg=DEFAULT_FG).pack(pady=5)
tk.Label(root, text=", ".join(skin_concerns), font=("Helvetica", 10), bg=DEFAULT_BG, fg=DEFAULT_FG).pack(pady=5)
tk.Label(root, text="Key Ingredients:", font=("Helvetica", 12, "bold"), bg=DEFAULT_BG, fg=DEFAULT_FG).pack(pady=5)
tk.Label(root, text=", ".join(ingredients), font=("Helvetica", 10), bg=DEFAULT_BG, fg=DEFAULT_FG).pack(pady=5)

# Input frame (grouping filters)
input_frame = tk.Frame(root, bg=DEFAULT_BG)
input_frame.pack(pady=10, fill="x", padx=10)

# Skin Type (same as before)
tk.Label(input_frame, text="Skin Type (e.g., oily, dry, all):", font=("Helvetica", 12), bg=DEFAULT_BG, fg=DEFAULT_FG).grid(row=0, column=0, sticky="w", padx=5, pady=3)
skin_type_var = tk.StringVar(value="all")
skin_type_dropdown = ttk.Combobox(input_frame, textvariable=skin_type_var, values=["all", "oily", "dry", "normal", "combination"], width=22)
skin_type_dropdown.grid(row=0, column=1, padx=5, pady=3, sticky="w")

# Concerns (same)
tk.Label(input_frame, text="Concerns (comma-separated):", font=("Helvetica", 12), bg=DEFAULT_BG, fg=DEFAULT_FG).grid(row=1, column=0, sticky="w", padx=5, pady=3)
concerns_var = tk.StringVar()
concerns_entry = tk.Entry(input_frame, textvariable=concerns_var, width=44)
concerns_entry.grid(row=1, column=1, padx=5, pady=3, sticky="w")

# Top N (user input)
tk.Label(input_frame, text="Number of Top Recommendations:", font=("Helvetica", 12), bg=DEFAULT_BG, fg=DEFAULT_FG).grid(row=2, column=0, sticky="w", padx=5, pady=3)
top_n_var = tk.StringVar(value="5")
top_n_entry = tk.Entry(input_frame, textvariable=top_n_var, width=10)
top_n_entry.grid(row=2, column=1, padx=5, pady=3, sticky="w")

# Brand filter
tk.Label(input_frame, text="Brand Filter (optional):", font=("Helvetica", 12), bg=DEFAULT_BG, fg=DEFAULT_FG).grid(row=3, column=0, sticky="w", padx=5, pady=3)
brand_var = tk.StringVar()
brand_entry = tk.Entry(input_frame, textvariable=brand_var, width=44)
brand_entry.grid(row=3, column=1, padx=5, pady=3, sticky="w")

# Category filter
tk.Label(input_frame, text="Category Filter (optional):", font=("Helvetica", 12), bg=DEFAULT_BG, fg=DEFAULT_FG).grid(row=4, column=0, sticky="w", padx=5, pady=3)
category_var = tk.StringVar()
category_entry = tk.Entry(input_frame, textvariable=category_var, width=44)
category_entry.grid(row=4, column=1, padx=5, pady=3, sticky="w")

# Avoid ingredients
tk.Label(input_frame, text="Avoid ingredients (comma-separated):", font=("Helvetica", 12), bg=DEFAULT_BG, fg=DEFAULT_FG).grid(row=5, column=0, sticky="w", padx=5, pady=3)
avoid_var = tk.StringVar()
avoid_entry = tk.Entry(input_frame, textvariable=avoid_var, width=44)
avoid_entry.grid(row=5, column=1, padx=5, pady=3, sticky="w")

# Sort by options
tk.Label(input_frame, text="Sort By:", font=("Helvetica", 12), bg=DEFAULT_BG, fg=DEFAULT_FG).grid(row=6, column=0, sticky="w", padx=5, pady=3)
sort_var = tk.StringVar(value="rating")
sort_dropdown = ttk.Combobox(input_frame, textvariable=sort_var, values=["rating", "brand", "relevance"], width=20)
sort_dropdown.grid(row=6, column=1, padx=5, pady=3, sticky="w")

# Buttons frame (keeps original buttons and adds export)
button_frame = tk.Frame(root, bg=DEFAULT_BG)
button_frame.pack(pady=8)

tk.Button(button_frame, text="Direct Recommendation", command=lambda: display_recommendations(), bg="#FF69B4", fg="white", font=("Helvetica", 12)).grid(row=0, column=0, padx=6)
tk.Button(button_frame, text="Use Webcam", command=lambda: use_webcam_for_concerns(), bg="#FF69B4", fg="white", font=("Helvetica", 12)).grid(row=0, column=1, padx=6)
tk.Button(button_frame, text="Upload Image", command=lambda: upload_image_for_concerns(), bg="#FF69B4", fg="white", font=("Helvetica", 12)).grid(row=0, column=2, padx=6)
tk.Button(button_frame, text="Export Recommendations", command=lambda: export_recommendations(), bg="#FF69B4", fg="white", font=("Helvetica", 12)).grid(row=0, column=3, padx=6)

# Dark mode toggle
dark_mode_var = tk.BooleanVar(value=False)
def toggle_theme_event():
    if dark_mode_var.get():
        root.configure(bg=DARK_BG)
        for w in root.winfo_children():
            try:
                w.configure(bg=DARK_BG, fg=DARK_FG)
            except Exception:
                pass
    else:
        root.configure(bg=DEFAULT_BG)
        for w in root.winfo_children():
            try:
                w.configure(bg=DEFAULT_BG, fg=DEFAULT_FG)
            except Exception:
                pass

dark_check = tk.Checkbutton(button_frame, text="Dark Mode", variable=dark_mode_var, command=toggle_theme_event, bg=DEFAULT_BG)
dark_check.grid(row=0, column=4, padx=6)

# Results panel (scrollable)
results_container = tk.Frame(root, bg=DEFAULT_BG)
results_container.pack(fill="both", expand=True, padx=10, pady=10)

canvas = tk.Canvas(results_container, bg=DEFAULT_BG, highlightthickness=0)
scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg=DEFAULT_BG)

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

results_frame = scrollable_frame  # keep old name

# store current recommendations for export
current_recommendations = []

# ---------------------------
# Display recommendations (integrated)
# ---------------------------
def display_recommendations():
    global current_recommendations
    user_skin_type = skin_type_var.get().lower()
    user_concerns = concerns_var.get()
    top_n_input = top_n_var.get()
    brand_filter = brand_var.get()
    category_filter = category_var.get()
    avoid_ingredients = avoid_var.get()
    sort_by = sort_var.get().lower()

    if not user_skin_type or not user_concerns:
        messagebox.showwarning("Input Error", "Please fill in all fields!")
        return

    try:
        top_n = int(top_n_input)
        if top_n <= 0:
            raise ValueError()
    except Exception:
        messagebox.showwarning("Input Error", "Please enter a positive integer for the Top N recommendations.")
        return

    concerns_list = [c.strip() for c in user_concerns.split(",") if c.strip()]
    avoid_list = [a.strip() for a in avoid_ingredients.split(",")] if avoid_ingredients.strip() else []

    # get recommendations via enhanced engine
    message, recommendations = recommend_products(user_skin_type, concerns_list,
                                                   avoid_ingredients=avoid_list,
                                                   brand_filter=brand_filter,
                                                   category_filter=category_filter,
                                                   sort_by=sort_by)

    # Clear previous results
    for widget in results_frame.winfo_children():
        widget.destroy()

    # show message
    ttk.Label(results_frame, text=message, font=("Helvetica", 12, "bold"), background=DEFAULT_BG).pack(pady=10)

    if recommendations:
        # keep original behavior of sorting by rating and selecting top_n for display
        try:
            sorted_recommendations = sorted(recommendations, key=lambda x: float(x.get('rating', 0) or 0), reverse=True)
        except Exception:
            sorted_recommendations = recommendations

        sorted_recommendations = sorted_recommendations[:top_n]
        current_recommendations = sorted_recommendations

        for product in sorted_recommendations:
            prod_text = f"{product.get('name', 'Unknown')} by {product.get('brand', 'Unknown')} ({product.get('category', 'N/A')})\nRating: {product.get('rating', 'N/A')}"
            frame = tk.Frame(results_frame, bg=DEFAULT_BG)
            frame.pack(fill="x", padx=10, pady=6)

            lbl = tk.Label(frame, text=prod_text, font=("Helvetica", 10), wraplength=700, justify="left", bg=DEFAULT_BG, fg=DEFAULT_FG, anchor="w")
            lbl.pack(side="left", fill="x", expand=True)

            url = product.get('url', '')
            if isinstance(url, str) and url.strip():
                def open_link(u=url):
                    try:
                        webbrowser.open(u)
                    except Exception:
                        messagebox.showerror("Error", "Unable to open the product URL.")
                link_btn = tk.Button(frame, text="Open Link", command=open_link)
                link_btn.pack(side="right", padx=6)
    else:
        ttk.Label(results_frame, text="Try different criteria for better results.", font=("Helvetica", 10, "italic"), background=DEFAULT_BG).pack(pady=10)


# ---------------------------
# Webcam/Image usage functions (preserve)
# ---------------------------
def use_webcam_for_concerns():
    detected_concerns = detect_skin_concerns_from_webcam()
    concerns_var.set(", ".join(detected_concerns))
    messagebox.showinfo("Detected Concerns", f"Detected concerns: {', '.join(detected_concerns)}")

def upload_image_for_concerns():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        detected_concerns = detect_skin_concerns_from_image(file_path)
        if detected_concerns:
            concerns_var.set(", ".join(detected_concerns))
            messagebox.showinfo("Detected Concerns", f"Detected concerns: {', '.join(detected_concerns)}")
        else:
            messagebox.showinfo("Detection", "No concerns detected from the image. Please try another image or enter concerns manually.")


# ---------------------------
# Export functionality (CSV / TXT)
# ---------------------------
def export_recommendations():
    if not current_recommendations:
        messagebox.showwarning("No Data", "Please generate recommendations first!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt")])
    if not file_path:
        return
    try:
        rows = []
        for p in current_recommendations:
            rows.append({
                "name": p.get("name"),
                "brand": p.get("brand"),
                "category": p.get("category"),
                "rating": p.get("rating"),
                "url": p.get("url"),
                "key_ingredients": ", ".join(map(str, p.get("key_ingredients", [])))
            })
        df_export = pd.DataFrame(rows)
        if file_path.endswith(".csv"):
            df_export.to_csv(file_path, index=False)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                for _, row in df_export.iterrows():
                    f.write(f"{row['name']} by {row['brand']} ({row['category']}) - Rating: {row['rating']}\nURL: {row['url']}\nIngredients: {row['key_ingredients']}\n\n")
        messagebox.showinfo("Export Successful", f"Recommendations saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export: {e}")


# ---------------------------
# Start GUI loop
# ---------------------------
root.mainloop()

