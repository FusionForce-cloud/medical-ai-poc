import json
from logging_config import logger

def get_patient_report(patient_name: str) -> str:
    logger.info(f"Patient lookup started: {patient_name}")

    with open("data/patients.json") as f:
        patients = json.load(f)

    matches = [p for p in patients
               if p["patient_name"].lower() == patient_name.lower()]

    if len(matches) == 0:
        logger.warning(f"No patient found for name: {patient_name}")
        return "No patient found with that name."
    elif len(matches) > 1:
        logger.warning(f"Multiple patients found for name: {patient_name}")
        return "Multiple patients found. Please provide full name."
    
    patient = matches[0]
    logger.info(f"Patient found: {patient_name}")
    return json.dumps(patient, indent=2)
