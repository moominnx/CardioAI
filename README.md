# 🫀 AI Health Risk Assessment

ระบบประเมินความเสี่ยงโรคหัวใจเบื้องต้นด้วย ML + LLM (Typhoon)

## 🚀 รันใน 3 คำสั่ง

```bash
# 1. ติดตั้ง dependencies
pip install -r requirements.txt

# 2. Train model (รันครั้งเดียวพอ)
python train_model.py

# 3. รัน app
streamlit run app.py
```

## 🔑 ตั้งค่า API Key

```bash
cp .env.example .env
# แล้วแก้ไข .env ใส่ GROQ_API_KEY ของคุณ
# สมัครฟรีที่ https://console.groq.com
```

## 🐳 รันด้วย Docker

```bash
docker compose up --build
```

## 🏗️ System Architecture

```
User Input (Streamlit)
    ↓
Random Forest Model → Risk Score (%)
    ↓
Groq LLM (Llama 3.3 70B) → Thai Explanation
    ↓
Display Results + Warnings
```

## 📊 Model Customization

- Dataset: Cleveland Heart Disease (UCI / OpenML)
- Algorithm: Random Forest with custom hyperparameters
  - `class_weight="balanced"` — จัดการ Imbalanced data
  - `n_estimators=200, max_depth=8` — ปรับเพื่อลด Overfitting
- Feature Engineering: ตรวจสอบค่าผิดปกติ (ความดัน, คอเลสเตอรอล)
- LLM Prompt Engineering: System prompt เฉพาะทางการแพทย์ภาษาไทย

## ⚠️ ข้อจำกัด

ผลลัพธ์เป็นการประเมินเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้

## SDG

- **SDG 3**: Good Health and Well-Being — ส่งเสริมการดูแลสุขภาพเชิงป้องกัน
