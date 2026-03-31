import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import random

# Load AI Model
try:
    rf_model = joblib.load('disease_model.pkl')
    symptoms_list = joblib.load('symptoms_list.pkl')
    print("✅ Real AI Model Loaded Successfully")
except Exception as e:
    print("⚠️ Warning: Could not load AI model. Did you run train_model.py?")

app = FastAPI(title="Vitalis AI - CDSS API")

# Allow frontend (Playcode/Vercel) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. TEMPORARY IN-MEMORY DATABASE
# (Replacing Neon Postgres temporarily so you don't hit connection errors while testing)
# ==========================================
doctors_db = {}
patients_db = {}
reports_db = []
prescriptions_db = []

# ==========================================
# 2. PYDANTIC SCHEMAS (Data Validation)
# ==========================================
class DoctorAuth(BaseModel):
    username: str
    password: str
    department: Optional[str] = "General Medicine"

class PatientAuth(BaseModel):
    patient_no: str
    dob: str
    name: Optional[str] = "Unknown"
    password: str = "default123"

class SymptomPayload(BaseModel):
    doctor_id: int
    patient_id: int
    symptoms: list[str] # Expecting a list of string symptoms

class PrescriptionPayload(BaseModel):
    report_id: int
    doctor_id: int
    medicines: str
    dosage: str
    notes: str

# ==========================================
# 3. DOCTOR ROUTES
# ==========================================
@app.post("/api/doctor/signup")
def doctor_signup(data: DoctorAuth):
    doc_id = len(doctors_db) + 1
    doctors_db[doc_id] = {"id": doc_id, "username": data.username, "password": data.password, "department": data.department}
    return {"message": "Doctor registered successfully!", "doctor_id": doc_id}

@app.post("/api/doctor/login")
def doctor_login(data: DoctorAuth):
    for doc_id, doc in doctors_db.items():
        if doc["username"] == data.username and doc["password"] == data.password:
            return {"message": "Login successful", "doctor_id": doc_id}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# ==========================================
# 4. PATIENT ROUTES
# ==========================================
@app.post("/api/patient/signup")
def patient_signup(data: PatientAuth):
    pat_id = len(patients_db) + 1
    patients_db[pat_id] = {"id": pat_id, "patient_no": data.patient_no, "name": data.name, "dob": data.dob}
    return {"message": "Patient registered successfully!", "patient_id": pat_id}

@app.post("/api/patient/login")
def patient_login(data: PatientAuth):
    for pat_id, pat in patients_db.items():
        if pat["patient_no"] == data.patient_no and pat["dob"] == data.dob:
            return {"message": "Login successful", "patient_id": pat_id}
    raise HTTPException(status_code=401, detail="Invalid patient credentials")

@app.get("/api/patient/{patient_id}/reports")
def get_patient_reports(patient_id: int):
    # Fetch reports and attach any prescriptions to them
    pat_reports = [r for r in reports_db if r["patient_id"] == patient_id]
    for report in pat_reports:
        report["prescription"] = next((p for p in prescriptions_db if p["report_id"] == report["id"]), None)
    
    if not pat_reports:
        return {"status": "empty", "message": "No records found."}
    return {"status": "success", "data": pat_reports[::-1]} # Return newest first

# ==========================================
# 5. AI PREDICTION ENGINE (Advanced Mock)
# ==========================================
@app.post("/api/predict")
def predict_disease(data: SymptomPayload):
    if not data.symptoms:
        raise HTTPException(status_code=400, detail="Please select at least one symptom.")

    # 1. Prepare the input matrix for the AI
    input_data = np.zeros(len(symptoms_list))
    for symp in data.symptoms:
        # Match formatting from the frontend to the dataset
        formatted_symp = symp.replace(" ", "_") 
        if formatted_symp in symptoms_list:
            input_data[symptoms_list.index(formatted_symp)] = 1

    # 2. Ask the AI to predict!
    prediction = rf_model.predict([input_data])[0]
    
    # 3. Get Confidence Score (Probability)
    probabilities = rf_model.predict_proba([input_data])[0]
    confidence_score = round(max(probabilities) * 100, 1)

    # Calculate dynamic risk level
    if confidence_score > 90:
        risk = "High"
    elif confidence_score > 60:
        risk = "Medium"
    else:
        risk = "Low"

    report_id = len(reports_db) + 1
    new_report = {
        "id": report_id,
        "doctor_id": data.doctor_id,
        "patient_id": data.patient_id,
        "symptoms_input": ", ".join(data.symptoms),
        "predicted_disease": prediction,
        "confidence_score": confidence_score,
        "risk_level": risk,
        "recommended_specialist": "General Physician", # Can be mapped dynamically later
        "precautions": ["Consult doctor immediately", "Rest and hydrate"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    reports_db.append(new_report)
    return new_report

# ==========================================
# 6. DOCTOR PRESCRIPTION ROUTE
# ==========================================
@app.post("/api/prescription")
def add_prescription(data: PrescriptionPayload):
    presc_id = len(prescriptions_db) + 1
    new_prescription = {
        "id": presc_id,
        "report_id": data.report_id,
        "doctor_id": data.doctor_id,
        "medicines": data.medicines,
        "dosage": data.dosage,
        "notes": data.notes,
        "issued_at": datetime.now().strftime("%Y-%m-%d")
    }
    prescriptions_db.append(new_prescription)
    return {"message": "Prescription added successfully!", "data": new_prescription}
