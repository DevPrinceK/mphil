import numpy as np
import pandas as pd


def build_mask(n: int, missing_rate: float = 0.3, seed: int = 42, boundary: int = 5) -> np.ndarray:
	"""Create a boolean mask where True means observed, False means missing."""
	rng = np.random.default_rng(seed)
	mask = np.ones(n, dtype=bool)
	missing_indices = rng.choice(n, size=int(n * missing_rate), replace=False)
	mask[missing_indices] = False
	if boundary > 0:
		mask[:boundary] = False
		mask[-boundary:] = False
	return mask


def make_lag_features(values_filled: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
	"""Create a feature matrix using lagged values and lagged masks.

	Features per time t:
	  [x_{t-1}, ..., x_{t-L}, m_{t-1}, ..., m_{t-L}]
	where x is filled (missing -> 0.0) and m is 1.0 observed / 0.0 missing.
	"""
	n = len(values_filled)
	m_float = mask.astype(np.float32)
	feats = []
	for k in range(1, max_lag + 1):
		xk = np.roll(values_filled, k)
		mk = np.roll(m_float, k)
		xk[:k] = 0.0
		mk[:k] = 0.0
		feats.append(xk)
	for k in range(1, max_lag + 1):
		mk = np.roll(m_float, k)
		mk[:k] = 0.0
		feats.append(mk)
	return np.stack(feats, axis=1)


def report_metrics(name: str, imputed: np.ndarray, data: np.ndarray, mask: np.ndarray) -> None:
	missing_true = data[~mask]
	missing_pred = imputed[~mask]
	mae = float(np.mean(np.abs(missing_pred - missing_true)))
	rmse = float(np.sqrt(np.mean((missing_pred - missing_true) ** 2)))
	print(f"{name:<28s} MAE: {mae:8.4f}  RMSE: {rmse:8.4f}")


def main() -> None:
	# ---------------------------------------------------------------------
	# Load data (same dataset assumptions as other scripts)
	# ---------------------------------------------------------------------
	col_names = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']
	df = pd.read_csv('data.txt', sep=r'\s+', names=col_names, header=None)

	sensor_id = 1
	df = df[df.moteid == sensor_id].sort_values(['date', 'time']).reset_index(drop=True)
	df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
	df = df.set_index('datetime')

	data = df['temperature'].astype(float).values
	n = len(data)
	print(f"Loaded {n} readings for sensor {sensor_id}")

	# ---------------------------------------------------------------------
	# Masking (random + boundary), consistent with baseline-stat notebook
	# ---------------------------------------------------------------------
	missing_rate = 0.3
	seed = 42
	boundary = 5
	mask = build_mask(n, missing_rate=missing_rate, seed=seed, boundary=boundary)
	print(f"Masked points: {(~mask).sum()} / {n} ({(~mask).mean()*100:.1f}%)")

	# Observed with placeholder fill
	observed_filled = data.copy().astype(np.float32)
	observed_filled[~mask] = 0.0

	# ---------------------------------------------------------------------
	# Feature engineering (univariate): lagged values + lagged masks
	# ---------------------------------------------------------------------
	max_lag = 48
	X = make_lag_features(observed_filled, mask, max_lag=max_lag)
	y = data.astype(np.float32)

	train_idx = np.where(mask)[0]
	test_idx = np.where(~mask)[0]

	X_train, y_train = X[train_idx], y[train_idx]
	X_test = X[test_idx]

	# ---------------------------------------------------------------------
	# Baselines (ML)
	# ---------------------------------------------------------------------
	try:
		from sklearn.pipeline import Pipeline
		from sklearn.preprocessing import StandardScaler
		from sklearn.neighbors import KNeighborsRegressor
		from sklearn.linear_model import Ridge
		from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
		from sklearn.experimental import enable_iterative_imputer  # noqa: F401
		from sklearn.impute import IterativeImputer
		from sklearn.linear_model import BayesianRidge
	except ImportError as e:
		raise ImportError(
			"scikit-learn is required for baseline-ml.py. Install with: pip install scikit-learn"
		) from e

	# Helper: reconstruct full series preserving observed values
	def merge_predictions(pred_missing: np.ndarray) -> np.ndarray:
		out = data.copy().astype(np.float32)
		out[~mask] = pred_missing.astype(np.float32)
		return out

	print("\nML Baselines (univariate, lag features):")

	# 1) KNN regression baseline
	knn = Pipeline([
		("scaler", StandardScaler()),
		("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
	])
	knn.fit(X_train, y_train)
	pred_knn = knn.predict(X_test)
	imputed_knn = merge_predictions(pred_knn)
	report_metrics("KNN Regressor", imputed_knn, data, mask)

	# 2) Ridge regression baseline
	ridge = Pipeline([
		("scaler", StandardScaler()),
		("model", Ridge(alpha=1.0, random_state=seed)),
	])
	ridge.fit(X_train, y_train)
	pred_ridge = ridge.predict(X_test)
	imputed_ridge = merge_predictions(pred_ridge)
	report_metrics("Ridge Regression", imputed_ridge, data, mask)

	# 3) Gradient boosting regression baseline (no scaling required)
	hgb = HistGradientBoostingRegressor(random_state=seed, max_depth=6, learning_rate=0.05)
	hgb.fit(X_train, y_train)
	pred_hgb = hgb.predict(X_test)
	imputed_hgb = merge_predictions(pred_hgb)
	report_metrics("HistGradientBoosting", imputed_hgb, data, mask)

	# 4) Iterative imputation on a Hankel/lag matrix (MICE-style)
	#    Matrix columns: [x_t, x_{t-1}, ..., x_{t-L}]
	x_nan = data.copy().astype(np.float32)
	x_nan[~mask] = np.nan
	lag_mat = [x_nan]
	for k in range(1, max_lag + 1):
		col = np.roll(x_nan, k)
		col[:k] = np.nan
		lag_mat.append(col)
	lag_mat = np.stack(lag_mat, axis=1)  # [N, L+1]

	mice = IterativeImputer(
		estimator=BayesianRidge(),
		random_state=seed,
		max_iter=10,
		sample_posterior=False,
		initial_strategy="mean",
		skip_complete=True,
	)
	lag_filled_mice = mice.fit_transform(lag_mat)
	imputed_mice = data.copy().astype(np.float32)
	imputed_mice[~mask] = lag_filled_mice[~mask, 0].astype(np.float32)
	report_metrics("IterativeImputer (BR)", imputed_mice, data, mask)

	# 5) MissForest-like iterative imputation (ExtraTreesRegressor)
	missforest_like = IterativeImputer(
		estimator=ExtraTreesRegressor(
			n_estimators=200,
			random_state=seed,
			n_jobs=-1,
			min_samples_leaf=2,
		),
		random_state=seed,
		max_iter=10,
		sample_posterior=False,
		initial_strategy="mean",
		skip_complete=True,
	)
	lag_filled_mf = missforest_like.fit_transform(lag_mat)
	imputed_mf = data.copy().astype(np.float32)
	imputed_mf[~mask] = lag_filled_mf[~mask, 0].astype(np.float32)
	report_metrics("IterativeImputer (ET)", imputed_mf, data, mask)


if __name__ == "__main__":
	main()

