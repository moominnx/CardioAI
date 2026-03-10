"""
main.py — FastAPI backend สำหรับ CardioAI
วางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ model.pkl แล้วรัน:
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="CardioAI API")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

# ── CORS (อนุญาต HTML เรียกจาก localhost ทุก port) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── โหลดโมเดล ──
model      = joblib.load("model.pkl")       # Lab mode  (clinical features)
model_quiz = joblib.load("model_quiz.pkl")  # Quiz mode (indirect features)

# ── LabelEncoder mappings (ต้องตรงกับที่ train ไว้) ──
# LabelEncoder เรียง alphabetically เสมอ
ENCODINGS = {
    "Sex":           {"F": 0, "M": 1},
    "ChestPainType": {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3},
    "RestingECG":    {"LVH": 0, "Normal": 1, "ST": 2},
    "ExerciseAngina":{"N": 0, "Y": 1},
    "ST_Slope":      {"Down": 0, "Flat": 1, "Up": 2},
}

# ── Input schema ──
class PatientInput(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

class QuizInput(BaseModel):
    Age: int
    Sex: str            # "M" | "F"
    food_habit: int     # 0-3
    fitness_level: int  # 0-2
    bp_history: int     # 0-2
    chest_symptom: int  # 0-2
    sugar_history: int  # 0-1
    smoking: int = 0        # 0-2 (ใช้คำนวณ smoking_proxy ใน backend)
    family_history: int = 0 # 0-2 (เก็บไว้ใน patientContext)
    bmi_category: int = 1   # 0-3 (ใช้คำนวณ bmi_proxy ใน backend)

@app.post("/predict")
def predict(data: PatientInput):
    # ใช้ DataFrame เพื่อให้ชื่อ column ตรงกับตอน train
    X = pd.DataFrame([{
        "Age":            data.Age,
        "Sex":            ENCODINGS["Sex"][data.Sex],
        "ChestPainType":  ENCODINGS["ChestPainType"][data.ChestPainType],
        "RestingBP":      data.RestingBP,
        "Cholesterol":    data.Cholesterol,
        "FastingBS":      data.FastingBS,
        "RestingECG":     ENCODINGS["RestingECG"][data.RestingECG],
        "MaxHR":          data.MaxHR,
        "ExerciseAngina": ENCODINGS["ExerciseAngina"][data.ExerciseAngina],
        "Oldpeak":        data.Oldpeak,
        "ST_Slope":       ENCODINGS["ST_Slope"][data.ST_Slope],
    }])
    prob = float(model.predict_proba(X)[0][1])
    score = round(prob * 100)

    if prob >= 0.7:
        level = "สูง"
        level_en = "high"
        emoji = "🔴"
        advice = "ความเสี่ยงสูงมาก ควรพบแพทย์โดยเร็วที่สุด"
    elif prob >= 0.4:
        level = "ปานกลาง"
        level_en = "medium"
        emoji = "🟡"
        advice = "มีปัจจัยเสี่ยงที่ควรระวัง แนะนำพบแพทย์ภายใน 1–2 เดือน"
    else:
        level = "ต่ำ"
        level_en = "low"
        emoji = "🟢"
        advice = "ความเสี่ยงต่ำ ดูแลสุขภาพต่อเนื่องและตรวจสุขภาพประจำปี"

    return {
        "risk_score": score,
        "risk_level": level,
        "risk_level_en": level_en,
        "emoji": emoji,
        "advice": advice,
        "probability": round(prob, 4),
    }

@app.post("/predict_quiz")
def predict_quiz(data: QuizInput):
    age_group = 0 if data.Age <= 40 else 1 if data.Age <= 55 else 2 if data.Age <= 70 else 3
    sex_enc   = 1 if data.Sex == "M" else 0

    # แปลง quiz answers → model features (ตรงกับ train_model_quiz.py v3)
    smoking_proxy = min(2, (1 if data.smoking >= 1 else 0) + (1 if data.smoking == 2 else 0))
    bmi_proxy     = min(3, (data.bmi_category - 1) if data.bmi_category > 1 else 0)
    age_x_chest   = age_group * data.chest_symptom

    X = pd.DataFrame([{
        "food_habit":    data.food_habit,
        "fitness_level": data.fitness_level,
        "bp_history":    data.bp_history,
        "chest_symptom": data.chest_symptom,
        "sugar_history": data.sugar_history,
        "smoking_proxy": smoking_proxy,
        "bmi_proxy":     bmi_proxy,
        "age_group":     age_group,
        "sex":           sex_enc,
        "age_x_chest":   age_x_chest,
    }])

    prob  = float(model_quiz.predict_proba(X)[0][1])
    score = round(prob * 100)

    if prob >= 0.7:
        level, emoji = "สูง", "🔴"
        advice = "ความเสี่ยงสูง ควรพบแพทย์และตรวจเลือดโดยเร็ว"
    elif prob >= 0.4:
        level, emoji = "ปานกลาง", "🟡"
        advice = "มีปัจจัยเสี่ยง แนะนำพบแพทย์และตรวจสุขภาพประจำปี"
    else:
        level, emoji = "ต่ำ", "🟢"
        advice = "ความเสี่ยงต่ำ ดูแลสุขภาพและออกกำลังกายสม่ำเสมอ"

    return {
        "risk_score": score, "risk_level": level,
        "emoji": emoji, "advice": advice,
        "probability": round(prob, 4), "mode": "quiz"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest Heart Disease"}
