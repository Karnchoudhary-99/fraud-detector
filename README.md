# Credit Card Fraud Detector

Detect fraudulent credit card transactions from tabular transaction
features: unsupervised anomaly scoring for the cold-start case, plus
supervised classifiers once labels exist — with precision/recall
evaluation that respects the extreme class imbalance of fraud data.

## Roadmap

- [x] Project scaffold (packaging, README, CI)
- [ ] Transaction dataset loading + feature engineering
- [ ] Anomaly detection (isolation forest, z-score baselines)
- [ ] Supervised classifiers (logistic regression, gradient boosting)
- [ ] Imbalance-aware evaluation (precision/recall/F1, PR-AUC)
- [ ] CLI + demo, README, then public

## Usage

```bash
pip install -e .
python -m pytest -q
```

## License

MIT
