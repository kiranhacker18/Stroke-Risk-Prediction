import os

def load_data():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_path, 'data', 'healthcare-dataset-stroke-data.csv')
    
    df = pd.read_csv(file_path)
    return df