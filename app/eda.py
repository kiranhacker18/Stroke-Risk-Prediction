import pandas as pd
import matplotlib.pyplot as plt
import os
from data_preprocessing import load_data

def generate_graphs():
    df = load_data()

    # 🔥 Correct static folder path (inside app)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    static_path = os.path.join(base_path, 'app', 'static')

    # Create folder if not exists
    os.makedirs(static_path, exist_ok=True)

    # 📊 1. Stroke Distribution
    plt.figure()
    df['stroke'].value_counts().plot(kind='bar')
    plt.title('Stroke Distribution')
    plt.xlabel('Stroke (0 = No, 1 = Yes)')
    plt.ylabel('Count')

    plt.savefig(os.path.join(static_path, 'stroke_dist.png'))  # 🔥 FIXED NAME
    plt.close()

    # 📊 2. Age Distribution
    plt.figure()
    df['age'].hist()
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    plt.savefig(os.path.join(static_path, 'age_dist.png'))
    plt.close()
