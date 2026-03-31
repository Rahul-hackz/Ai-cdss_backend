from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# We use os.getenv so Render can securely inject your database URL
# Replace the string in quotes with your ACTUAL Neon Connection String for local testing
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_H2tQ4XIjETfl@ep-polished-mouse-abz10xca-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
