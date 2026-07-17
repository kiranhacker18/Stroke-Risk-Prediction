import sys
import os
import csv
from datetime import timedelta

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from flask import Flask, render_template, request, redirect, session, send_file
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from data_preprocessing import load_data, preprocess_data
from eda import generate_graphs

app = Flask(__name__)

# 🔐 SESSION CONFIG
app.secret_key = "supersecretkey"
app.permanent_session_lifetime = timedelta(days=7)

# ================= LOAD MODELS =================
base_path = os.path.join(os.path.dirname(__file__), '..', 'models')

rf_model = pickle.load(open(os.path.join(base_path, 'rf_model.pkl'), 'rb'))
lr_model = pickle.load(open(os.path.join(base_path, 'lr_model.pkl'), 'rb'))
knn_model = pickle.load(open(os.path.join(base_path, 'knn_model.pkl'), 'rb'))

# ================= GLOBAL DATA =================
latest_result = {}
confidence_msg = ""
doctor_msg = ""

# ================= USER FUNCTIONS =================

def save_user(username, password):
    path = os.path.join(os.path.dirname(__file__), 'users.csv')
    exists = os.path.isfile(path)

    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['username', 'password'])
        writer.writerow([username, password])


def user_exists(username):
    path = os.path.join(os.path.dirname(__file__), 'users.csv')

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                if row[0] == username:
                    return True
    return False


def validate_user(username, password):
    path = os.path.join(os.path.dirname(__file__), 'users.csv')

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                if row[0] == username and row[1] == password:
                    return True
    return False


# ================= AUTH ROUTES =================

@app.route('/')
def root():
    return redirect('/login')   # 🔥 ALWAYS START WITH LOGIN


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect('/dashboard')

    error = None

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if validate_user(username, password):
            session['user'] = username
            session.permanent = True
            return redirect('/dashboard')
        else:
            error = "Invalid Username or Password"

    return render_template('login.html', error=error)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm_password']

        if password != confirm:
            error = "Passwords do not match"

        elif user_exists(username):
            error = "User already exists"

        else:
            save_user(username, password)
            return redirect('/login')

    return render_template('signup.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ================= DASHBOARD =================

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')

    generate_graphs()
    return render_template('index.html')


# ================= SAVE HISTORY =================

def save_prediction(data):
    path = os.path.join(os.path.dirname(__file__), 'predictions.csv')
    exists = os.path.isfile(path)

    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(['Name','Phone','Age','BP','Heart','Glucose','BMI','Risk'])

        writer.writerow(data)


# ================= PREDICT =================

@app.route('/predict', methods=['POST'])
def predict():

    if 'user' not in session:
        return redirect('/login')

    global latest_result, confidence_msg, doctor_msg

    # 🔥 INPUTS
    patient_name = request.form['patient_name']
    phone = request.form['phone']
    age = request.form['age']
    bp_category = request.form['bp_category']
    heart_disease = request.form['heart_disease']
    glucose = request.form['avg_glucose_level']
    bmi = request.form['bmi']

    hypertension = 1 if bp_category == 'high' else 0

    input_data = {
        'gender': 'Male',
        'age': float(age),
        'hypertension': hypertension,
        'heart_disease': int(heart_disease),
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': float(glucose),
        'bmi': float(bmi),
        'smoking_status': 'never smoked'
    }

    # PREPROCESS
    df = load_data()
    df = preprocess_data(df)
    X = df.drop('stroke', axis=1)

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # 🔥 PREDICTION
    prob = rf_model.predict_proba(input_df)[0][1]

    result = f"High Risk ({round(prob*100,2)}%)" if prob > 0.5 else f"Low Risk ({round(prob*100,2)}%)"

    # SAVE RESULT
    latest_result = {
        "Patient Name": patient_name,
        "Phone": phone,
        "Age": age,
        "BP": bp_category,
        "Heart Disease": heart_disease,
        "Glucose": glucose,
        "BMI": bmi,
        "Risk": result
    }

    save_prediction([patient_name, phone, age, bp_category, heart_disease, glucose, bmi, result])

    # 🔥 GRAPH
    plt.figure()
    plt.bar(['No Risk', 'Stroke Risk'], [1 - prob, prob])
    plt.savefig(os.path.join(os.path.dirname(__file__), 'static', 'prediction.png'))
    plt.close()

    # 🔥 DETAILED MESSAGES
    if prob < 0.2:
        confidence_msg = (
            "Your stroke risk is low. Maintain a healthy lifestyle with regular exercise, "
            "balanced diet, and proper hydration. Avoid smoking and manage stress effectively."
        )

        doctor_msg = (
            "No immediate medical treatment required. Continue routine health checkups "
            "and monitor blood pressure periodically."
        )

    elif prob < 0.5:
        confidence_msg = (
            "You have a moderate risk of stroke. It is important to control blood pressure "
            "and glucose levels. Follow a healthy diet and increase physical activity."
        )

        doctor_msg = (
            "Regular monitoring is recommended. Consult a doctor if you notice symptoms "
            "like dizziness, headaches, or irregular heartbeat."
        )

    else:
        confidence_msg = (
            "High stroke risk detected. Immediate lifestyle changes are required including "
            "strict control of blood pressure, sugar levels, and cholesterol."
        )

        doctor_msg = (
            "Consult a doctor immediately. Medication such as blood pressure control, "
            "blood thinners, or diabetes management may be required."
        )

    return render_template(
        'index.html',
        prediction_text=result,
        confidence_msg=confidence_msg,
        doctor_msg=doctor_msg,
        patient_name=patient_name,
        phone=phone,
        age=age,
        bp_category=bp_category,
        heart_disease=heart_disease,
        glucose=glucose,
        bmi=bmi
    )


# ================= HISTORY =================

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/login')

    path = os.path.join(os.path.dirname(__file__), 'predictions.csv')
    data = []

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = list(csv.reader(f))

    return render_template('history.html', data=data)


# ================= CLEAR HISTORY =================

@app.route('/clear_history', methods=['POST'])
def clear_history():
    path = os.path.join(os.path.dirname(__file__), 'predictions.csv')

    if os.path.exists(path):
        os.remove(path)

    return redirect('/history')


# ================= PDF REPORT =================

@app.route('/download_report')
def download_report():

    if 'user' not in session:
        return redirect('/login')

    file_path = os.path.join(os.path.dirname(__file__), 'report.pdf')
    graph_path = os.path.join(os.path.dirname(__file__), 'static', 'prediction.png')

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Stroke Risk Prediction Report", styles['Title']))
    content.append(Spacer(1, 15))

    for k, v in latest_result.items():
        content.append(Paragraph(f"<b>{k}:</b> {v}", styles['Normal']))
        content.append(Spacer(1, 8))

    if os.path.exists(graph_path):
        content.append(Spacer(1, 15))
        content.append(Image(graph_path, width=400, height=250))

    content.append(Spacer(1, 15))
    content.append(Paragraph("<b>Precautions:</b>", styles['Heading2']))
    content.append(Paragraph(confidence_msg, styles['Normal']))

    content.append(Spacer(1, 10))
    content.append(Paragraph("<b>Doctor Recommendation:</b>", styles['Heading2']))
    content.append(Paragraph(doctor_msg, styles['Normal']))

    doc.build(content)

    return send_file(file_path, as_attachment=True)


# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)