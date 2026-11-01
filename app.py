"""
Breast Cancer Prediction - Advanced App
Chalane ke liye: python3 -m streamlit run app.py

Features:
1. Single Prediction  - saari 30 values khud daalo (ya sample load karo)
2. Batch Prediction   - CSV upload karo, sabka result ek saath, download bhi
3. Model Info         - model comparison, confusion matrix, accuracy
4. Input validation   - galat/out-of-range values par warning
"""
import io
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

#  here we load model our model

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_bundle():
    return joblib.load(os.path.join(HERE, "model.pkl"))


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("model.pkl nahi mila. Pehle terminal mein `python3 train.py` chalao.")
    st.stop()

model = bundle["model"]
features = bundle["features"]
means, mins, maxs = bundle["means"], bundle["mins"], bundle["maxs"]

ds = load_breast_cancer(as_frame=True)
BENIGN_SAMPLE = ds.data[ds.target == 1].iloc[0].to_dict()
MALIGNANT_SAMPLE = ds.data[ds.target == 0].iloc[0].to_dict()


# ------------------------------------------------------------------ helpers
def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame (30 feature columns) -> prediction + confidence columns."""
    X = df[features].astype(float)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    out = df.copy()
    out["Prediction"] = ["Benign" if p == 1 else "Malignant" for p in pred]
    out["Malignant %"] = (proba[:, 0] * 100).round(1)
    out["Benign %"] = (proba[:, 1] * 100).round(1)
    out["Confidence %"] = (proba.max(axis=1) * 100).round(1)
    return out


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:

    """Column names ko match karta hai (case, underscore, space ignore)."""

    key = lambda s: str(s).strip().lower().replace("_", " ")
    lookup = {key(f): f for f in features}
    return df.rename(columns={c: lookup[key(c)] for c in df.columns if key(c) in lookup})


def top_reasons(row: dict, n: int = 5):

    """Logistic Regression ke liye: kaunsi features ne result ko sabse zyada push kiya."""

    try:
        scaler, clf = model.named_steps["scaler"], model.named_steps["model"]
        if not hasattr(clf, "coef_"):
            return None
        x = scaler.transform(pd.DataFrame([row])[features])[0]
        contrib = pd.Series(x * clf.coef_[0], index=features)
        # coef positive => Benign ki taraf, negative => Malignant ki taraf
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:n]
        return pd.DataFrame({
            "Feature": top.index,
            "Push towards": ["Benign" if v > 0 else "Malignant" for v in top.values],
            "Strength": top.abs().round(2).values,
        })
    except Exception:
        return None


def set_sample(sample: dict):
    for f in features:
        st.session_state[f"in_{f}"] = float(sample[f])


for f in features:
    st.session_state.setdefault(f"in_{f}", float(means[f]))

# Header

st.title("🩺 Breast Cancer Prediction System")
st.caption(f"Model: **{bundle['name']}** | Dataset: Wisconsin Breast Cancer (569 samples, 30 features)")

tab1, tab2, tab3 = st.tabs(["🔬 Single Prediction", "📂 Batch Prediction (CSV)", "📊 Model Info"])

# tab 1: single

with tab1:
    c1, c2, c3, _ = st.columns([1.3, 1.3, 1.3, 3])
    c1.button("Benign sample", on_click=set_sample, args=(BENIGN_SAMPLE,))
    c2.button("Malignant sample", on_click=set_sample, args=(MALIGNANT_SAMPLE,))
    c3.button("Reset (average)", on_click=set_sample, args=(means,))
    st.write("Neeche saari values apni marzi se badlo. Sample buttons se real patient ka data load ho jata hai.")

    groups = {"Mean values": [f for f in features if f.startswith("mean ")],
              "Error values": [f for f in features if f.endswith(" error")],
              "Worst values": [f for f in features if f.startswith("worst ")]}
    for gname, flist in groups.items():
        with st.expander(gname, expanded=(gname == "Mean values")):
            cols = st.columns(3)
            for i, f in enumerate(flist):
                cols[i % 3].number_input(
                    f, key=f"in_{f}", format="%.5f", step=float((maxs[f] - mins[f]) / 100 or 0.01),
                    help=f"Dataset range: {mins[f]:.4g} to {maxs[f]:.4g}")

    row = {f: st.session_state[f"in_{f}"] for f in features}

    # validation

    negatives = [f for f in features if row[f] < 0]
    far_out = [f for f in features if row[f] > maxs[f] * 2 and row[f] >= 0]
    outside = [f for f in features if not (mins[f] <= row[f] <= maxs[f]) and f not in negatives]
    if negatives:
        st.error("Ye values negative nahi ho sakti: " + ", ".join(negatives))
    if far_out:
        st.warning("Ye values dataset ke maximum se bahut zyada hain, result bharosemand nahi hoga: "
                   + ", ".join(far_out))
    elif outside:
        st.info(f"{len(outside)} value(s) dataset ki range ke bahar hain. Prediction phir bhi chalega, "
                "par model ne aisa data pehle nahi dekha.")

    if st.button("Predict", type="primary", disabled=bool(negatives)):
        res = predict_df(pd.DataFrame([row])).iloc[0]
        a, b = st.columns(2)
        if res["Prediction"] == "Malignant":
            a.error(f"### Malignant (cancerous)\nConfidence: **{res['Confidence %']}%**")
        else:
            a.success(f"### Benign (non-cancerous)\nConfidence: **{res['Confidence %']}%**")
        b.write("**Probability**")
        b.progress(float(res["Malignant %"]) / 100, text=f"Malignant {res['Malignant %']}%")
        b.progress(float(res["Benign %"]) / 100, text=f"Benign {res['Benign %']}%")
        if res["Confidence %"] < 70:
            st.warning("Confidence kam hai. Model ko is case mein pakka nahi pata.")
        reasons = top_reasons(row)
        if reasons is not None:
            st.write("**Result ke peeche ke sabse bade 5 reasons**")
            st.dataframe(reasons, hide_index=True, width="stretch")

# tab 2: batch

with tab2:
    st.write("CSV file upload karo jisme 30 feature columns hon. Extra columns (jaise id, diagnosis) ignore ho jayenge.")
    sample_csv = pd.DataFrame([BENIGN_SAMPLE, MALIGNANT_SAMPLE]).to_csv(index=False)
    st.download_button("Sample CSV download karo", sample_csv, "sample_input.csv", "text/csv")

    up = st.file_uploader("CSV upload", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"CSV padh nahi paye: {e}")
            st.stop()
        df = normalise_columns(raw)
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"{len(missing)} column(s) missing hain: " + ", ".join(missing[:8])
                     + (" ..." if len(missing) > 8 else ""))
        else:
            feat = df[features].apply(pd.to_numeric, errors="coerce")
            bad = feat.isna().any(axis=1)
            if bad.any():
                st.warning(f"{int(bad.sum())} row(s) mein khali ya galat values hain, wo skip ho gayi "
                           f"(rows: {[int(i) + 2 for i in feat.index[bad][:10]]} ...)")
            good = feat[~bad]
            if good.empty:
                st.error("Koi valid row nahi mili.")
            else:
                result = predict_df(good)
                n_mal = int((result["Prediction"] == "Malignant").sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Total rows", len(result))
                m2.metric("Malignant", n_mal)
                m3.metric("Benign", len(result) - n_mal)
                show = result[["Prediction", "Confidence %", "Malignant %", "Benign %"]]
                st.dataframe(show, width="stretch")
                st.download_button("Result CSV download karo", result.to_csv(index=False),
                                   "predictions.csv", "text/csv")

#  tab 3: info

with tab3:
    st.write(f"**Best model:** {bundle['name']}")
    st.write("Train.py ne 4 models compare kiye: Logistic Regression, KNN, SVM, Random Forest "
             "(StandardScaler + 5-fold cross validation).")
    c1, c2 = st.columns(2)
    for col, fn, cap in [(c1, "model_comparison.png", "Model comparison"),
                         (c2, "confusion_matrix.png", "Confusion matrix")]:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            col.image(p, caption=cap)
        else:
            col.info(f"{fn} nahi mili. `python3 train.py` chalao.")
