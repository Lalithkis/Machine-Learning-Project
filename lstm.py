import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
df = pd.read_csv(r"C:\Users\ASUS\Downloads\Car Data.csv")
print(df.head(100))
if 'engine_temp' not in df.columns:
    np.random.seed(42)
    df['engine_temp'] = np.random.normal(75, 10, size=len(df)) 
df.ffill(inplace=True)
df['engine_temp_mean'] = df['engine_temp'].rolling(window=7).mean()
df['engine_temp_std'] = df['engine_temp'].rolling(window=7).std()
print("Original Data:")
print(df[['Year', 'Mileage']])
numeric_cols = ['Year', 'Mileage']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("\nScaled Data:")
print(df[['Year', 'Mileage']])
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
selected_features = ['engine_temp', 'Mileage', 'fuel_consumption', 'engine_temp_mean', 'engine_temp_std']
for feature in selected_features:
    if feature not in df.columns:
        print(f"Warning: '{feature}' is not in the DataFrame columns.")
scaled_df = df[selected_features]
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)
SEQ_LENGTH = 30 
X, y = create_sequences(scaled_df.values, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQ_LENGTH, len(selected_features))))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))  
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()