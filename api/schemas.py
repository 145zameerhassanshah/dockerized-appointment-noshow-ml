from typing import Literal
from pydantic import BaseModel, Field

class AppointmentFeatures(BaseModel):
    age: int = Field(ge=18, le=100)
    days_until_appointment: int = Field(ge=0, le=365)
    previous_appointments: int = Field(ge=0, le=1000)
    previous_no_shows: int = Field(ge=0, le=1000)
    sms_reminder_sent: Literal["yes","no"]
    appointment_hour: int = Field(ge=0, le=23)
    weekday: Literal["Monday","Tuesday","Wednesday","Thursday","Friday"]
    department: Literal["General Medicine","Dermatology","Orthopedics","Pediatrics"]
