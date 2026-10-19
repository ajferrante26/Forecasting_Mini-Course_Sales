import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 1. Load data
# -----------------------------
train = pd.read_csv('train.csv', parse_dates=['date'])
test = pd.read_csv('test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('sample_submission.csv')

print("=" * 60)
print("DATA DIAGNOSTICS")
print("=" * 60)
print(f"train dates: {train['date'].min().date()} -> {train['date'].max().date()}  shape={train.shape}")
print(f"test dates:  {test['date'].min().date()} -> {test['date'].max().date()}  shape={test.shape}")
print(f"sample_submission shape: {sample_sub.shape}")
print(f"train cols:  {train.columns.tolist()}")
print(f"test cols:   {test.columns.tolist()}")
print(f"sub cols:    {sample_sub.columns.tolist()}")
print("=" * 60)

# -----------------------------
# 2. Aggregate total daily sales
# -----------------------------
daily_sales = train.groupby('date')['num_sold'].sum().reset_index().sort_values('date')
full_dates = pd.date_range(daily_sales['date'].min(), daily_sales['date'].max(), freq='D')
daily_sales = (daily_sales.set_index('date').reindex(full_dates)
               .rename_axis('date').reset_index())
daily_sales['num_sold'] = daily_sales['num_sold'].interpolate()

train_end = daily_sales['date'].max()
print(f"Train range: {daily_sales['date'].min().date()} -> {train_end.date()}")

# -----------------------------
# 3. Normalize + create sequences
# -----------------------------
scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(daily_sales[['num_sold']])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(sales_scaled, seq_length)
target_dates = daily_sales['date'].iloc[seq_length:].reset_index(drop=True)

# -----------------------------
# 4. Split: last month of train as validation
# -----------------------------
val_start = train_end - pd.DateOffset(months=1)
val_mask = np.asarray(target_dates >= val_start)

if val_mask.sum() < 30:
    print(f"Too few validation samples ({val_mask.sum()}). Falling back to 90/10 split.")
    split_idx = int(len(X) * 0.9)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val,   y_val   = X[split_idx:], y[split_idx:]
    val_dates = target_dates.iloc[split_idx:].reset_index(drop=True)
else:
    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val,   y_val   = X[val_mask],  y[val_mask]
    val_dates = target_dates[val_mask].reset_index(drop=True)

print(f"Train samples: {len(X_train)}  Validation samples: {len(X_val)}")
print(f"Validation range: {val_dates.min().date()} -> {val_dates.max().date()}")

if len(X_val) == 0:
    raise ValueError("Validation is empty. Check that train.csv spans at least a few months.")

# -----------------------------
# 5. Build CNN-LSTM
# -----------------------------
model = Sequential([
    Conv1D(64, 3, activation='relu', padding='same', input_shape=(seq_length, 1)),
    MaxPooling1D(2),
    Conv1D(32, 3, activation='relu', padding='same'),
    MaxPooling1D(2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# -----------------------------
# 6. Train
# -----------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]
history = model.fit(X_train, y_train, epochs=80, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

# -----------------------------
# 7. Evaluate + visualize
# -----------------------------
results = model.evaluate(X_val, y_val, verbose=0)
loss = results[0]
mae  = results[1] if isinstance(results, list) and len(results) > 1 else float('nan')
print(f'Validation loss: {loss:.4f}  Validation MAE: {mae:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history['loss'], label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('Loss'); axes[0].legend()
axes[1].plot(history.history['mae'], label='Train')
axes[1].plot(history.history['val_mae'], label='Val')
axes[1].set_title('MAE'); axes[1].legend()
plt.tight_layout(); plt.show()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

val_pred = scaler.inverse_transform(model.predict(X_val, verbose=0))
y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))

plt.figure(figsize=(14, 5))
plt.plot(val_dates.values, y_val_inv, label='Actual')
plt.plot(val_dates.values, val_pred, label='Predicted')
plt.title('Validation Predictions vs Actual')
plt.legend(); plt.show()
plt.savefig('validation_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

# -----------------------------
# 8. Recursive forecast for test (2022)
# -----------------------------
test_start, test_end = test['date'].min(), test['date'].max()
forecast_start = train_end + pd.Timedelta(days=1)
all_dates = pd.date_range(forecast_start, test_end, freq='D')

print(f"\nForecast window: {forecast_start.date()} -> {test_end.date()} ({len(all_dates)} days)")
print(f"Test window:     {test_start.date()} -> {test_end.date()}")

last_seq = sales_scaled[-seq_length:].reshape(1, seq_length, 1).astype('float32')
preds = []
for _ in range(len(all_dates)):
    p = model(tf.constant(last_seq), training=False).numpy()
    preds.append(p[0, 0])
    last_seq = np.concatenate([last_seq[:, 1:, :], p.reshape(1, 1, 1)], axis=1)

preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
preds_inv = np.clip(preds_inv, 0, None)

forecast_df = pd.DataFrame({'date': all_dates, 'predicted_total': preds_inv.flatten()})
forecast_df = forecast_df[(forecast_df['date'] >= test_start) & (forecast_df['date'] <= test_end)]

# -----------------------------
# 9. Allocate aggregate forecast to each series
# -----------------------------
for df in (train, test):
    df['series_id'] = df['country'] + '_' + df['store'] + '_' + df['product']

# Use the last year that is essentially complete in training data
if train_end.month == 12 and train_end.day >= 28:
    ref_year = train_end.year
else:
    ref_year = train_end.year - 1
print(f"Reference year for seasonal proportions: {ref_year}")

reference = train[train['date'].dt.year == ref_year].copy()
if len(reference) < 10000:
    print(f"Sparse data for {ref_year}. Using all training data as reference.")
    reference = train.copy()
else:
    print(f"Using year {ref_year} as seasonal reference.")

tot = reference.groupby('date')['num_sold'].sum().rename('total_sales')
reference = reference.merge(tot, on='date')
reference['proportion'] = reference['num_sold'] / reference['total_sales']
reference['month_day'] = reference['date'].dt.strftime('%m-%d')

prop_map = reference.set_index(['month_day', 'series_id'])['proportion'].to_dict()
avg_prop = reference.groupby('series_id')['proportion'].mean().to_dict()

test['month_day'] = test['date'].dt.strftime('%m-%d')
test['proportion'] = test.apply(
    lambda r: prop_map.get((r['month_day'], r['series_id']),
                           avg_prop.get(r['series_id'], np.nan)),
    axis=1)
test['proportion'] = test['proportion'].fillna(test['series_id'].map(avg_prop)).fillna(0)

sums = test.groupby('date')['proportion'].transform('sum')
zero = sums == 0
if zero.any():
    counts = test.groupby('date')['series_id'].transform('count')
    test.loc[zero, 'proportion'] = 1.0 / counts[zero]
    sums = test.groupby('date')['proportion'].transform('sum')
test['proportion'] = test['proportion'] / sums

test = test.merge(forecast_df, on='date', how='left')
test['predicted_total'] = test['predicted_total'].fillna(
    daily_sales['num_sold'].iloc[-366:].mean())
test['num_sold'] = test['predicted_total'] * test['proportion']

# -----------------------------
# 10. Align to sample_submission
# -----------------------------
key_cols = ['date', 'country', 'store', 'product']
submission = None

# Strategy 1: merge on key columns
if all(c in sample_sub.columns for c in key_cols) and all(c in test.columns for c in key_cols):
    print("\nAligning submission on key columns:", key_cols)
    right = test[key_cols + ['num_sold']].drop_duplicates(subset=key_cols)
    sub_clean = sample_sub.drop(columns=['num_sold'], errors='ignore')
    submission = sub_clean.merge(right, on=key_cols, how='left')
    submission['num_sold'] = submission['num_sold'].fillna(0)
    print(f"Merged rows: {len(submission)}")

# Strategy 2: shared id column
if submission is None:
    id_candidates = ['row_id', 'id', 'ID', 'RowId', 'Id', 'index']
    tid = next((c for c in id_candidates if c in test.columns), None)
    sid = next((c for c in id_candidates if c in sample_sub.columns), None)
    print(f"\nDetected ids: test={tid}, sample_sub={sid}")
    if tid and sid:
        id_map = test.set_index(tid)['num_sold']
        submission = sample_sub.copy()
        submission['num_sold'] = submission[sid].map(id_map).fillna(0)
        print(f"Aligned by id: {len(submission)}")

# Strategy 3: positional
if submission is None:
    if len(sample_sub) != len(test):
        raise ValueError(
            f"Cannot align: sample_sub has {len(sample_sub)} rows, "
            f"test has {len(test)} rows, and no shared ids or key columns.\n"
            f"test cols: {test.columns.tolist()}\n"
            f"sub cols:  {sample_sub.columns.tolist()}")
    print("\nFalling back to positional alignment.")
    submission = sample_sub.copy()
    submission['num_sold'] = test['num_sold'].values

# Ensure final column set matches sample_submission
keep = list(sample_sub.columns)
if 'num_sold' not in keep:
    keep.append('num_sold')
extra = [c for c in submission.columns if c not in keep]
submission = submission.drop(columns=extra, errors='ignore')
submission = submission[keep].reset_index(drop=True)

print("\nSubmission preview:")
print(submission.head())
print(f"Total rows: {len(submission)}")
submission.to_csv('submission.csv', index=False)
print("Saved to submission.csv")
