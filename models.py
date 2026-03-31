from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from database import Base
import datetime

class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    department = Column(String)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    patient_no = Column(String, unique=True, index=True)
    name = Column(String)
    dob = Column(String)
    password = Column(String, default="default123")

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    patient_id = Column(Integer, ForeignKey("patients.id"))
    symptoms_input = Column(String)
    predicted_disease = Column(String)
    confidence_score = Column(Float)
    risk_level = Column(String)
    recommended_specialist = Column(String)
    ai_summary = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
