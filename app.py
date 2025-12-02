# ==================================================
# MEDSHARE UNIFIED API (Render Deployment Version)
# No Ngrok - Pure Flask
# ==================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import io
import json
import os
import re
import requests
import google.generativeai as genai
from symspellpy import SymSpell, Verbosity

app = Flask(__name__)
CORS(app)

# ========================================================
# 1. GOOGLE GEMINI SETUP
# ========================================================
# Ideally, use os.environ.get("GOOGLE_API_KEY") for security on Render
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found!")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Auto-Select Best Model
print("🔄 Configuring AI Model...")
active_model = None
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if 'flash' in m.name:
                active_model = m.name
                break
    if not active_model: active_model = 'gemini-pro-vision'
    
    model = genai.GenerativeModel(active_model)
    print(f"✅ Connected to: {active_model}")
except:
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Connected to Fallback Model")


# ========================================================
# 2. HELPER FUNCTIONS
# ========================================================
# --- SymSpell Setup (Optional) ---
DICT_PATH = "medicine_list.txt"
if not os.path.exists(DICT_PATH):
    try:
        url = "https://raw.githubusercontent.com/glutanimate/wordlist-medicalterms-en/master/wordlist.txt"
        r = requests.get(url)
        with open(DICT_PATH, "w", encoding="utf-8") as f:
            f.write(r.text)
    except: pass

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
try:
    if os.path.exists(DICT_PATH):
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                sym_spell.create_dictionary_entry(line.strip().lower(), 1)
except: pass

def fix_spelling(raw_name):
    if not raw_name or len(raw_name) < 3: return raw_name
    clean = re.sub(r'[^a-zA-Z\s]', '', raw_name).strip().lower()
    suggestions = sym_spell.lookup(clean, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term.title()
    return raw_name.title()

def clean_text_value(text):
    if not text or text == "Not Detected": return ""
    text = re.sub(r'[®™℞©*]', '', text)
    return text.strip()

def clean_date(text):
    if not text or text == "Not Detected": return ""
    return re.sub(r'[^0-9/.-]', '', text)

def clean_strength(text):
    if not text or text == "Not Detected": return ""
    text = text.lower().replace(" ", "").replace("o", "0").replace("l", "1")
    return text.upper()


# ========================================================
# 3. API ENDPOINTS
# ========================================================

@app.route("/")
def home():
    return jsonify({"status": "✅ MedShare API is Live on Render!"})

# --- ROUTE A: PRESCRIPTIONS (Raw AI) ---
@app.route("/predict-prescription", methods=["POST"])
def predict_prescription():
    try:
        if 'image' not in request.files: return jsonify({"error": "No image"}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = ImageOps.exif_transpose(image)

        prompt = """
        You are an expert pharmacist AI. Analyze this handwritten prescription image.
        Identify all medicines and their strengths.
        Return strict JSON: { "medicines": [ { "name": "Name", "strength": "500mg" } ] }
        """
        
        response = model.generate_content([prompt, image])
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(clean_text)
            medicines_raw = data.get("medicines", [])
        except:
            medicines_raw = []

        # No SymSpell here as requested (Raw Output)
        return jsonify({"status": "success", "medicines": medicines_raw})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- ROUTE B: MEDICINE BOXES (Cleaned) ---
@app.route("/predict-box", methods=["POST"])
def predict_box():
    try:
        if 'image' not in request.files: return jsonify({"error": "No image"}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = ImageOps.exif_transpose(image)

        prompt = """
        Analyze medicine box. Find: medicineName, strength, expiryDate, manufacturingDate.
        Return JSON keys exactly as listed.
        """
        
        response = model.generate_content([prompt, image])
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)

        final_response = {
            "medicineName": clean_text_value(data.get("medicineName", "")),
            "strength": clean_text_value(data.get("strength", "")),
            "expiryDate": clean_date(data.get("expiryDate", "")),
            "manufacturingDate": clean_date(data.get("manufacturingDate", "")),
            # Backup Keys
            "Medicine_Name": clean_text_value(data.get("medicineName", "")),
            "Strength": clean_text_value(data.get("strength", "")),
            "EXP_Date": clean_date(data.get("expiryDate", "")),
            "MFG_Date": clean_date(data.get("manufacturingDate", ""))
        }
        
        filtered_response = {k: v for k, v in final_response.items() if v and v != ""}

        return jsonify({"status": "success", "fields": filtered_response, "detections": filtered_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================================================
# MAIN EXECUTION
# ==================================================
if __name__ == "__main__":
    # Render assigns a port automatically, so we use os.environ.get
    port = int(os.environ.get("PORT", 5000))

    app.run(host='0.0.0.0', port=port)



