# Chosen transistor models (deliverable 5)

Written by `scripts/03_train_models.py` (`make train`). One model per case:

| File | Content |
|------|---------|
| `<case>.joblib` | fitted model (`dvd.fetmodel.FetModel`), loaded with `joblib.load` |
| `<case>.json` | model type, hyperparameters, features, validation and test metrics, dataset checksum |

## Use

```python
import sys, joblib, pandas as pd
sys.path.insert(0, "src")                  # the model class lives in src/dvd/fetmodel.py
model = joblib.load("models/nmos_off.joblib")
X = pd.DataFrame({"vdd": [1.0], "temp": [27.0], "w": [44e-9], "vd": [1.0], "vs": [0.0],
                  "vg": [0.0], "vb": [0.0], "vth0": [0.3692], "toxe": [6.5e-10],
                  "ndep": [1.2e19], "u0": [0.0181], "xj": [7.2e-9]})
I = model.predict(X)                       # shape (n, 4): I(Vd), I(Vg), I(Vs), I(Vb)
```

Inputs are the 12 input columns of the datasets, in the same units. Outputs use the
dataset sign convention: positive = current out of the device terminal into its source.
The four predicted currents sum to zero exactly (Kirchhoff is built into the model).
Each model is valid inside the section 4 sampling ranges and for its own case's gate
and body voltages only.
