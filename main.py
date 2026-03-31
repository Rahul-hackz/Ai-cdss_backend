import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import datetime

# Import your database and models
import models
from database import engine, get_db

# Create all tables in Neon Postgres automatically
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Vitalis AI - CDSS API")

# Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD AI MODEL
# ==========================================
try:
    rf_model = joblib.load('disease_model.pkl')
    symptoms_list = joblib.load('symptoms_list.pkl')
    print("✅ Real AI Model Loaded Successfully")
except Exception as e:
    print("⚠️ Warning: Could not load AI model. Did you run train_model.py?")

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class DoctorAuth(BaseModel):
    username: str
    password: str
    department: Optional[str] = "General Medicine"

class PatientAuth(BaseModel):
    patient_no: str
    dob: str
    name: Optional[str] = "Unknown"

class SymptomPayload(BaseModel):
    doctor_id: int
    patient_id: int
    symptoms: list[str]

# ==========================================
# DOCTOR ROUTES (Now using Neon DB)
# ==========================================
@app.post("/api/doctor/signup")
def doctor_signup(data: DoctorAuth, db: Session = Depends(get_db)):
    db_doc = db.query(models.Doctor).filter(models.Doctor.username == data.username).first()
    if db_doc:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_doc = models.Doctor(username=data.username, password=data.password, department=data.department)
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    return {"message": "Doctor registered successfully!", "doctor_id": new_doc.id}

@app.post("/api/doctor/login")
def doctor_login(data: DoctorAuth, db: Session = Depends(get_db)):
    doc = db.query(models.Doctor).filter(models.Doctor.username == data.username, models.Doctor.password == data.password).first()
    if not doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "doctor_id": doc.id}

# ==========================================
# PATIENT ROUTES (Now using Neon DB)
# ==========================================
@app.post("/api/patient/signup")
def patient_signup(data: PatientAuth, db: Session = Depends(get_db)):
    db_pat = db.query(models.Patient).filter(models.Patient.patient_no == data.patient_no).first()
    if db_pat:
        raise HTTPException(status_code=400, detail="Patient number already registered")
        
    new_pat = models.Patient(patient_no=data.patient_no, name=data.name, dob=data.dob)
    db.add(new_pat)
    db.commit()
    db.refresh(new_pat)
    return {"message": "Patient registered successfully!", "patient_id": new_pat.id}

@app.post("/api/patient/login")
def patient_login(data: PatientAuth, db: Session = Depends(get_db)):
    pat = db.query(models.Patient).filter(models.Patient.patient_no == data.patient_no, models.Patient.dob == data.dob).first()
    if not pat:
        raise HTTPException(status_code=401, detail="Invalid patient credentials")
    return {"message": "Login successful", "patient_id": pat.id}

@app.get("/api/patient/{patient_id}/reports")
def get_patient_reports(patient_id: int, db: Session = Depends(get_db)):
    reports = db.query(models.Report).filter(models.Report.patient_id == patient_id).order_by(models.Report.id.desc()).all()
    if not reports:
        return {"status": "empty", "message": "No records found."}
    return {"status": "success", "data": reports}

# ==========================================
# AI PREDICTION ENGINE (Now saving to Neon DB)
# ==========================================
@app.post("/api/predict")
def predict_disease(data: SymptomPayload, db: Session = Depends(get_db)):
    if not data.symptoms:
        raise HTTPException(status_code=400, detail="Please select at least one symptom.")

    # Verify Patient Exists
    pat = db.query(models.Patient).filter(models.Patient.id == data.patient_id).first()
    if not pat:
        raise HTTPException(status_code=404, detail="Patient ID not found in database.")

    # Prepare Matrix for AI
    input_data = np.zeros(len(symptoms_list))
    for symp in data.symptoms:
        formatted_symp = symp.replace(" ", "_") 
        if formatted_symp in symptoms_list:
            input_data[symptoms_list.index(formatted_symp)] = 1

    # AI Prediction & Confidence
    prediction = rf_model.predict([input_data])[0]
    probabilities = rf_model.predict_proba([input_data])[0]
    confidence_score = round(max(probabilities) * 100, 1)

    # Determine Risk Level
    if confidence_score > 90:
        risk = "High"
    elif confidence_score > 60:
        risk = "Medium"
    else:
        risk = "Low"

    # Save to Neon Database
    new_report = models.Report(
        doctor_id=data.doctor_id,
        patient_id=data.patient_id,
        symptoms_input=", ".join(data.symptoms),
        predicted_disease=prediction,
        confidence_score=confidence_score,
        risk_level=risk,
        recommended_specialist="General Physician",
        ai_summary=f"AI detected a {confidence_score}% probability of {prediction} based on reported symptoms.",
        created_at=datetime.datetime.utcnow()
    )
    
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    return new_report
