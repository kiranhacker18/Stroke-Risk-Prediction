from data_preprocessing import load_data, preprocess_data

df = load_data()
df = preprocess_data(df)

print("\nProcessed Data:\n")
print(df.head())