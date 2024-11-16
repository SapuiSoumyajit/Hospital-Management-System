# Hospital-Management-System
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Database Setup
def create_database():
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS Patients (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        age INTEGER,
                        gender TEXT,
                        illness TEXT)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS Doctors (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        specialization TEXT)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS Appointments (
                        id INTEGER PRIMARY KEY,
                        patient_id INTEGER,
                        doctor_id INTEGER,
                        date TEXT,
                        FOREIGN KEY(patient_id) REFERENCES Patients(id),
                        FOREIGN KEY(doctor_id) REFERENCES Doctors(id))''')
    conn.commit()
    conn.close()

# Data Management Functions
def add_patient(name, age, gender, illness):
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Patients (name, age, gender, illness) VALUES (?, ?, ?, ?)",
                   (name, age, gender, illness))
    conn.commit()
    conn.close()

def add_doctor(name, specialization):
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Doctors (name, specialization) VALUES (?, ?)", (name, specialization))
    conn.commit()
    conn.close()

def add_appointment(patient_id, doctor_id, date):
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Appointments (patient_id, doctor_id, date) VALUES (?, ?, ?)",
                   (patient_id, doctor_id, date))
    conn.commit()
    conn.close()

def fetch_patients():
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Patients")
    patients = cursor.fetchall()
    conn.close()
    return patients

def fetch_doctors():
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Doctors")
    doctors = cursor.fetchall()
    conn.close()
    return doctors

# Machine Learning Model Training
def train_risk_model():
    # Fetch data for training the model
    patients = fetch_patients()
    if not patients:
        print("No patient data available for training.")
        return None

    # Create a DataFrame
    data = pd.DataFrame(patients, columns=['id', 'name', 'age', 'gender', 'illness'])

    # Convert categorical illness to numeric
    illness_mapping = {illness: idx for idx, illness in enumerate(data['illness'].unique())}
    data['illness'] = data['illness'].map(illness_mapping)
    
    # Label gender as numeric
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
    
    # Create a "risk" label (example: high risk if age > 50 and severe illness)
    data['risk'] = data.apply(lambda x: 1 if x['age'] > 50 and x['illness'] > 0 else 0, axis=1)

    # Define features and target variable
    X = data[['age', 'illness', 'gender']]
    y = data['risk']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model training completed with accuracy: {accuracy * 100:.2f}%")
    
    return model, illness_mapping

# Main Code
if __name__ == "__main__":
    create_database()
    
    # Add sample doctors
    add_doctor("Dr. John Smith", "Cardiologist")
    add_doctor("Dr. Emily Davis", "Neurologist")
    
    # Add sample patients
    add_patient("Alice Brown", 30, "Female", "Flu")
    add_patient("Bob White", 45, "Male", "Diabetes")
    add_patient("Carol Black", 65, "Female", "Heart Disease")
    
    # Add sample appointments
    add_appointment(1, 1, "2024-11-15")
    add_appointment(2, 2, "2024-11-16")
    
    # Train the risk prediction model
    model, illness_mapping = train_risk_model()
    
    # Predicting risk for a new patient (example)
    if model:
        age = 40
        gender = 0  # Male
        illness = illness_mapping.get("Diabetes", -1)  # Get illness index
        if illness != -1:
            risk_prediction = model.predict([[age, illness, gender]])
            risk_level = "High" if risk_prediction[0] == 1 else "Low"
            print(f"Predicted risk level for patient with age {age} and illness 'Diabetes': {risk_level}")
        else:
            print("Illness type not found in the training data.")
