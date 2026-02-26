import os
import time
import numpy as np
import pandas as pd
import joblib
from app.ai.preprocessing import StockDataPreprocessor
from app.config import (
    MODEL_DIR, LSTM_SEQUENCE_LENGTH, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    MODEL_FRESHNESS_HOURS, PREDICTION_HORIZONS,
    LSTM_EARLY_STOP_PATIENCE, LSTM_LR_REDUCE_PATIENCE,
    LSTM_LR_REDUCE_FACTOR, LSTM_MIN_LR, LSTM_WALKFORWARD_SPLITS,
    FINE_TUNE_EPOCHS, FINE_TUNE_FRESHNESS_HOURS, LSTM_LEARNING_RATE,
)


class LSTMPredictor:
    def __init__(self):
        self.sequence_length = LSTM_SEQUENCE_LENGTH
        self.epochs = LSTM_EPOCHS
        self.batch_size = LSTM_BATCH_SIZE

    def _build_model(self, num_features: int = 1, seq_len: int = None):
        if seq_len is None:
            seq_len = self.sequence_length

        if num_features > 1:
            return self._build_attention_model(num_features, seq_len)
        else:
            # Legacy single-feature
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1),
            ])
            model.compile(optimizer=Adam(learning_rate=LSTM_LEARNING_RATE), loss="mean_squared_error")
            return model

    def _build_attention_model(self, num_features: int, seq_len: int):
        """Attention-LSTM: Functional API with MultiHeadAttention + residual + LayerNorm."""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add
        )
        from tensorflow.keras.optimizers import Adam

        inputs = Input(shape=(seq_len, num_features))

        # First LSTM block
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)

        # Second LSTM block
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)

        # Multi-Head Attention with residual connection
        attn_output = MultiHeadAttention(
            num_heads=4, key_dim=32, dropout=0.1
        )(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)

        # Third LSTM block (reduces to single vector)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)

        # Dense output
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=LSTM_LEARNING_RATE), loss="mean_squared_error")
        return model

    def _model_path(self, symbol: str, suffix: str = "") -> str:
        safe = symbol.replace(" ", "_").replace("^", "")
        return str(MODEL_DIR / f"lstm_v3_{safe}{suffix}.keras")

    def _scaler_path(self, symbol: str, suffix: str = "") -> str:
        safe = symbol.replace(" ", "_").replace("^", "")
        return str(MODEL_DIR / f"scaler_v3_{safe}{suffix}.joblib")

    def _model_age_hours(self, path: str) -> float:
        if not os.path.exists(path):
            return float("inf")
        return (time.time() - os.path.getmtime(path)) / 3600

    def _model_is_fresh(self, path: str) -> bool:
        return self._model_age_hours(path) < MODEL_FRESHNESS_HOURS

    def _save_scalers(self, symbol: str, close_scaler, feature_scaler, suffix: str = ""):
        path = self._scaler_path(symbol, suffix)
        joblib.dump({"close_scaler": close_scaler, "feature_scaler": feature_scaler}, path)

    def _load_scalers(self, symbol: str, suffix: str = ""):
        path = self._scaler_path(symbol, suffix)
        if os.path.exists(path):
            data = joblib.load(path)
            return data["close_scaler"], data["feature_scaler"]
        return None, None

    def _get_callbacks(self, patience_early=None, patience_lr=None):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        if patience_early is None:
            patience_early = LSTM_EARLY_STOP_PATIENCE
        if patience_lr is None:
            patience_lr = LSTM_LR_REDUCE_PATIENCE

        return [
            EarlyStopping(
                monitor="val_loss",
                patience=patience_early,
                restore_best_weights=True,
                min_delta=0.0001,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=LSTM_LR_REDUCE_FACTOR,
                patience=patience_lr,
                min_lr=LSTM_MIN_LR,
            ),
        ]

    def _horizon_to_steps(self, horizon: str) -> int:
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        if cfg.get("intraday"):
            return cfg.get("bars", 1)
        return cfg.get("days", 1)

    def _walk_forward_confidence(self, X, y, num_features, close_scaler, n_splits=None):
        """Walk-forward validation for realistic confidence score."""
        from tensorflow.keras.callbacks import EarlyStopping

        if n_splits is None:
            n_splits = LSTM_WALKFORWARD_SPLITS

        total_len = len(X)
        fold_size = total_len // (n_splits + 1)

        if fold_size < 20:
            return 50.0

        seq_len = X.shape[1]
        mapes = []

        for i in range(n_splits):
            train_end = fold_size * (i + 2)
            test_start = train_end
            test_end = min(test_start + fold_size, total_len)

            if test_end <= test_start:
                continue

            X_tr = X[:train_end]
            y_tr = y[:train_end]
            X_te = X[test_start:test_end]
            y_te = y[test_start:test_end]

            model = self._build_model(num_features, seq_len)
            early_stop = EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
            model.fit(
                X_tr, y_tr,
                epochs=30,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0,
            )

            test_pred = model.predict(X_te, verbose=0)
            test_pred_inv = close_scaler.inverse_transform(test_pred)
            y_te_inv = close_scaler.inverse_transform(y_te.reshape(-1, 1))

            nonzero = y_te_inv.flatten() != 0
            if nonzero.any():
                fold_mape = np.mean(
                    np.abs((y_te_inv[nonzero] - test_pred_inv[nonzero]) / y_te_inv[nonzero])
                ) * 100
                mapes.append(fold_mape)

        if not mapes:
            return 50.0

        avg_mape = np.mean(mapes)
        return max(0, min(100, 100 - avg_mape))

    def fine_tune(self, df: pd.DataFrame, symbol: str, suffix: str = ""):
        """Fine-tune an existing model on the most recent 20% of data (quick update)."""
        from tensorflow.keras.models import load_model
        from tensorflow.keras.callbacks import EarlyStopping

        model_path = self._model_path(symbol, suffix)
        if not os.path.exists(model_path):
            return False

        model = load_model(model_path)
        preprocessor = StockDataPreprocessor(self.sequence_length)

        try:
            if suffix == "_intraday":
                result = preprocessor.prepare_multifeature_intraday(df, symbol=symbol)
                X_train, y_train = result[0], result[1]
                close_scaler, feature_scaler = result[4], result[8]
            else:
                result = preprocessor.prepare_multifeature_lstm_data(df, symbol=symbol)
                X_train, y_train = result[0], result[1]
                close_scaler, feature_scaler = result[4], result[7]

            # Use only the most recent 20% for fine-tuning
            recent_start = int(len(X_train) * 0.8)
            X_recent = X_train[recent_start:]
            y_recent = y_train[recent_start:]

            if len(X_recent) < 10:
                return False

            early_stop = EarlyStopping(
                monitor="loss", patience=2, restore_best_weights=True
            )
            model.fit(
                X_recent, y_recent,
                epochs=FINE_TUNE_EPOCHS,
                batch_size=self.batch_size,
                callbacks=[early_stop],
                verbose=0,
            )
            model.save(model_path)
            self._save_scalers(symbol, close_scaler, feature_scaler, suffix)
            return True
        except Exception:
            return False

    def predict(self, df: pd.DataFrame, symbol: str, horizon: str = "1d") -> dict:
        from tensorflow.keras.models import load_model

        preprocessor = StockDataPreprocessor(self.sequence_length)

        if len(df) < self.sequence_length + 70:
            raise ValueError(
                f"Not enough data for LSTM. Need at least {self.sequence_length + 70} data points."
            )

        X_train, y_train, X_test, y_test, close_scaler, num_features, scaled_data, feature_scaler = \
            preprocessor.prepare_multifeature_lstm_data(df, symbol=symbol)

        model_path = self._model_path(symbol)
        age = self._model_age_hours(model_path)

        # Check if cached model has matching feature count
        force_retrain = False
        if age < FINE_TUNE_FRESHNESS_HOURS and os.path.exists(model_path):
            try:
                cached_model = load_model(model_path)
                expected_features = cached_model.input_shape[-1]
                if expected_features != num_features:
                    force_retrain = True
            except Exception:
                force_retrain = True

        if not force_retrain and age < MODEL_FRESHNESS_HOURS:
            # Fresh model — use cached
            model = load_model(model_path)
            saved_close, saved_feat = self._load_scalers(symbol)
            if saved_close is not None:
                close_scaler = saved_close
        elif not force_retrain and age < FINE_TUNE_FRESHNESS_HOURS:
            # Stale but not too old — fine-tune
            model = load_model(model_path)
            self.fine_tune(df, symbol)
            model = load_model(model_path)
            self._save_scalers(symbol, close_scaler, feature_scaler)
        else:
            # Too old or doesn't exist — full retrain
            model = self._build_model(num_features)
            model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=self._get_callbacks(),
                verbose=0,
            )
            model.save(model_path)
            self._save_scalers(symbol, close_scaler, feature_scaler)

        steps = self._horizon_to_steps(horizon)

        # Multi-step prediction with feature evolution
        current_sequence = scaled_data[-self.sequence_length:].copy()
        predictions = []

        # Compute per-feature trend deltas from last 5 real steps for damped extrapolation
        if current_sequence.shape[0] >= 5:
            recent = current_sequence[-5:]
            feature_deltas = np.mean(np.diff(recent, axis=0), axis=0)  # avg change per step
        else:
            feature_deltas = np.zeros(num_features)

        for step in range(steps):
            input_seq = current_sequence.reshape(1, self.sequence_length, num_features)
            pred = model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(pred)

            new_row = current_sequence[-1].copy()
            new_row[0] = pred  # close feature always gets prediction

            # For non-close features, apply damped trend continuation
            decay = 0.9 ** (step + 1)
            for feat_idx in range(1, num_features):
                new_row[feat_idx] = new_row[feat_idx] + feature_deltas[feat_idx] * decay
                new_row[feat_idx] = np.clip(new_row[feat_idx], 0.0, 1.0)  # MinMaxScaler range

            current_sequence = np.vstack([current_sequence[1:], new_row])

        predictions = close_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        # Walk-forward confidence
        all_X = np.concatenate([X_train, X_test]) if len(X_test) > 0 else X_train
        all_y = np.concatenate([y_train, y_test]) if len(y_test) > 0 else y_train
        confidence = self._walk_forward_confidence(all_X, all_y, num_features, close_scaler)

        # Compute MAPE on test set
        if len(X_test) > 0:
            test_pred = model.predict(X_test, verbose=0)
            test_pred_inv = close_scaler.inverse_transform(test_pred)
            y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1))
            nonzero = y_test_inv.flatten() != 0
            if nonzero.any():
                mape = float(np.mean(np.abs(
                    (y_test_inv[nonzero] - test_pred_inv[nonzero]) / y_test_inv[nonzero]
                )) * 100)
            else:
                mape = 0
        else:
            mape = 0

        # Generate future dates
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        if cfg.get("intraday"):
            last_dt = pd.to_datetime(df["date"].iloc[-1])
            dates = [
                (last_dt + pd.Timedelta(minutes=15 * (i + 1))).strftime("%Y-%m-%d %H:%M")
                for i in range(steps)
            ]
        else:
            last_date = pd.to_datetime(df["date"].iloc[-1])
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
            dates = [d.strftime("%Y-%m-%d") for d in future_dates]

        return {
            "predictions": [round(float(p), 2) for p in predictions],
            "dates": dates,
            "confidence_score": round(float(confidence), 2),
            "mape": round(float(mape), 2),
        }

    def predict_intraday(self, df: pd.DataFrame, symbol: str, horizon: str = "15m") -> dict:
        """Predict using intraday (15-min) candle data with multi-feature input."""
        from tensorflow.keras.models import load_model

        preprocessor = StockDataPreprocessor(self.sequence_length)

        X_train, y_train, X_test, y_test, close_scaler, num_features, scaled_data, seq_len, feature_scaler = \
            preprocessor.prepare_multifeature_intraday(df, symbol=symbol)

        model_path = self._model_path(symbol, suffix="_intraday")
        age = self._model_age_hours(model_path)

        # Check if cached model has matching feature count
        force_retrain = False
        if age < FINE_TUNE_FRESHNESS_HOURS and os.path.exists(model_path):
            try:
                cached_model = load_model(model_path)
                expected_features = cached_model.input_shape[-1]
                if expected_features != num_features:
                    force_retrain = True
            except Exception:
                force_retrain = True

        if not force_retrain and age < MODEL_FRESHNESS_HOURS:
            model = load_model(model_path)
            saved_close, saved_feat = self._load_scalers(symbol, suffix="_intraday")
            if saved_close is not None:
                close_scaler = saved_close
        elif not force_retrain and age < FINE_TUNE_FRESHNESS_HOURS:
            model = load_model(model_path)
            self.fine_tune(df, symbol, suffix="_intraday")
            model = load_model(model_path)
            self._save_scalers(symbol, close_scaler, feature_scaler, suffix="_intraday")
        else:
            model = self._build_model(num_features, seq_len)
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=16,
                validation_split=0.1,
                callbacks=self._get_callbacks(patience_early=5, patience_lr=3),
                verbose=0,
            )
            model.save(model_path)
            self._save_scalers(symbol, close_scaler, feature_scaler, suffix="_intraday")

        steps = self._horizon_to_steps(horizon)
        current_sequence = scaled_data[-seq_len:].copy()
        predictions = []

        # Compute per-feature trend deltas from last 5 real steps
        if current_sequence.shape[0] >= 5:
            recent = current_sequence[-5:]
            feature_deltas = np.mean(np.diff(recent, axis=0), axis=0)
        else:
            feature_deltas = np.zeros(num_features)

        for step in range(steps):
            input_seq = current_sequence.reshape(1, seq_len, num_features)
            pred = model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(pred)

            new_row = current_sequence[-1].copy()
            new_row[0] = pred  # close feature

            # For non-close features, apply damped trend continuation
            decay = 0.9 ** (step + 1)
            for feat_idx in range(1, num_features):
                new_row[feat_idx] = new_row[feat_idx] + feature_deltas[feat_idx] * decay
                new_row[feat_idx] = np.clip(new_row[feat_idx], 0.0, 1.0)

            current_sequence = np.vstack([current_sequence[1:], new_row])

        predictions = close_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        # MAPE confidence for intraday
        if len(X_test) > 0:
            test_pred = model.predict(X_test, verbose=0)
            test_pred_inv = close_scaler.inverse_transform(test_pred)
            y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1))
            nonzero = y_test_inv.flatten() != 0
            if nonzero.any():
                mape = float(np.mean(np.abs(
                    (y_test_inv[nonzero] - test_pred_inv[nonzero]) / y_test_inv[nonzero]
                )) * 100)
            else:
                mape = 0
            confidence = max(0, min(100, 100 - mape))
        else:
            mape = 0
            confidence = 50

        last_dt = pd.to_datetime(df["datetime"].iloc[-1]) if "datetime" in df.columns else pd.Timestamp.now()
        dates = [
            (last_dt + pd.Timedelta(minutes=15 * (i + 1))).strftime("%Y-%m-%d %H:%M")
            for i in range(steps)
        ]

        return {
            "predictions": [round(float(p), 2) for p in predictions],
            "dates": dates,
            "confidence_score": round(float(confidence), 2),
            "mape": round(float(mape), 2),
        }
