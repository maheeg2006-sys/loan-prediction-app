import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(
    page_title="Loan Status Prediction",
    page_icon="🏦",
    layout="wide",
)

MODEL_FILE = "loan_model_deploy.joblib"


@st.cache_resource

def load_bundle():
    return joblib.load(MODEL_FILE)


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error(
        f"`{MODEL_FILE}` was not found. Run `train_model_deploy.py` first."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info("Make sure the deployment environment uses the same scikit-learn and imbalanced-learn versions used during training.")
    st.stop()

models = bundle["models"]
classes = bundle["classes"]
numeric = bundle["numeric"]
categorical = bundle["categorical"]
order = bundle["feature_order"]
training = bundle.get("training", {})
data_summary = bundle.get("data_summary", {})
versions = bundle.get("versions", {})

# -------------------- Header --------------------
st.title("🏦 Loan Status Prediction")
st.caption(
    "Educational ML demonstration using Naïve Bayes and KNN with SMOTE-balanced training data."
)

if versions.get("scikit-learn") and versions["scikit-learn"] != sklearn.__version__:
    st.warning(
        f"Model was trained with scikit-learn {versions['scikit-learn']} but this app runs "
        f"{sklearn.__version__}. Use the pinned requirements and redeploy/retrain if needed."
    )

st.warning(
    "Educational use only: this model should not be used as the sole basis for a real credit decision. "
    "A prediction is not a guarantee of approval or rejection."
)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Project information")
    st.write(f"**Training rows:** {data_summary.get('train_rows', '-'):,}")
    st.write(f"**Test rows:** {data_summary.get('test_rows', '-'):,}")
    st.write(f"**KNN k:** {training.get('knn_k', '-')}")
    st.write(f"**Split:** {training.get('split', '-')}")
    st.write("**SMOTE:** Training data only")

    st.divider()
    st.subheader("Dataset classes")
    counts = data_summary.get("class_counts", {})
    if counts:
        st.dataframe(
            pd.DataFrame({"Class": list(counts), "Count": list(counts.values())})
            .set_index("Class"),
            use_container_width=True,
        )

    st.divider()
    st.caption("Model versions")
    st.code(
        f"scikit-learn {versions.get('scikit-learn', '?')}\n"
        f"imbalanced-learn {versions.get('imbalanced-learn', '?')}"
    )

# -------------------- Model comparison --------------------
st.subheader("Model comparison")
comparison = pd.DataFrame({name: info["metrics"] for name, info in models.items()}).T
st.dataframe(
    comparison.style.format("{:.2%}"),
    use_container_width=True,
)

model_name = st.radio(
    "Choose model",
    list(models),
    horizontal=True,
)
model_info = models[model_name]
model = model_info["pipeline"]
metrics = model_info["metrics"]

# -------------------- Evaluation --------------------
with st.expander("Detailed model evaluation", expanded=False):
    st.write(f"### {model_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("F1", f"{metrics['f1']:.2%}")

    st.markdown("**Confusion matrix**")
    cm = pd.DataFrame(
        model_info.get("confusion_matrix", []),
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes],
    )
    st.dataframe(cm, use_container_width=True)

    report = model_info.get("classification_report", {})
    report_rows = []
    for label in classes:
        if label in report:
            report_rows.append({
                "Class": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1": report[label]["f1-score"],
                "Support": int(report[label]["support"]),
            })
    if report_rows:
        st.markdown("**Per-class performance**")
        st.dataframe(
            pd.DataFrame(report_rows).set_index("Class").style.format({
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1": "{:.2%}",
            }),
            use_container_width=True,
        )

# -------------------- Applicant form --------------------
st.subheader("Applicant details")
inputs = {}

with st.form("applicant"):
    cols = st.columns(2)
    i = 0

    for name, options in categorical.items():
        inputs[name] = cols[i % 2].selectbox(
            name.replace("_", " ").title(),
            options,
        )
        i += 1

    for name, meta in numeric.items():
        label = name.replace("_", " ").title()
        help_text = f"Training-data range: {meta['min']:,.2f} to {meta['max']:,.2f}"

        if meta["is_binary"]:
            inputs[name] = cols[i % 2].selectbox(
                label,
                [0, 1],
                index=int(meta["median"]),
                format_func=lambda v: "Yes" if v == 1 else "No",
                help=help_text,
            )
        elif meta["is_int"]:
            inputs[name] = cols[i % 2].number_input(
                label,
                min_value=int(meta["min"]),
                max_value=int(meta["max"]),
                value=int(meta["median"]),
                step=1,
                help=help_text,
            )
        else:
            inputs[name] = cols[i % 2].number_input(
                label,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["median"]),
                step=1.0,
                help=help_text,
            )
        i += 1

    submitted = st.form_submit_button("🔍 Predict", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([inputs])[order]
    pred_raw = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    pred_label = classes[int(pred_raw)]

    st.divider()
    st.subheader("Prediction")

    if pred_label == "Approved":
        st.success(f"### {pred_label}")
    else:
        st.error(f"### {pred_label}")

    result = pd.DataFrame({
        "Class": model.classes_.astype(int).map(lambda x: classes[x]) if hasattr(model.classes_, 'map') else [classes[int(x)] for x in model.classes_],
        "Model probability": proba,
    }).set_index("Class")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(result.style.format("{:.2%}"), use_container_width=True)
    with c2:
        st.bar_chart(result["Model probability"])

    # Explain what the probability means for KNN.
    if model_name.startswith("KNN"):
        k = int(model.named_steps["clf"].n_neighbors)
        st.info(
            f"KNN uses k={k}. Its probability is based on the class votes among the nearest "
            f"training samples, so values can be coarse (for example 0%, 33.3%, 66.7%, 100% "
            f"when k=3). This is not a guaranteed real-world approval probability."
        )
    else:
        st.info(
            "Naïve Bayes probabilities are model estimates and can be overconfident when its "
            "feature-independence assumption is not a good fit for the data."
        )

    # Compare both models on the same applicant.
    st.markdown("### Both models on this applicant")
    rows = []
    for name, info in models.items():
        p = info["pipeline"]
        raw = p.predict(row)[0]
        probs = p.predict_proba(row)[0]
        label = classes[int(raw)]
        class_to_prob = {classes[int(c)]: float(v) for c, v in zip(p.classes_, probs)}
        rows.append({
            "Model": name,
            "Prediction": label,
            "Approved": class_to_prob.get("Approved", 0.0),
            "Rejected": class_to_prob.get("Rejected", 0.0),
        })

    comparison_pred = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        comparison_pred.style.format({"Approved": "{:.2%}", "Rejected": "{:.2%}"}),
        use_container_width=True,
    )

    if len({r["Prediction"] for r in rows}) > 1:
        st.warning(
            "The two models disagree. This is a useful reminder that the result depends on "
            "the chosen algorithm and should not be treated as a definitive decision."
        )

    # Downloadable prediction record.
    export = dict(inputs)
    export.update({
        "selected_model": model_name,
        "prediction": pred_label,
        "approved_probability": float(result.loc["Approved", "Model probability"]),
        "rejected_probability": float(result.loc["Rejected", "Model probability"]),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })
    st.download_button(
        "⬇️ Download prediction details",
        data=json.dumps(export, indent=2),
        file_name="loan_prediction.json",
        mime="application/json",
    )

# -------------------- Footer --------------------
st.divider()
st.caption(
    "This project is an educational demonstration of preprocessing, SMOTE, Naïve Bayes, "
    "KNN and model evaluation. The dataset and model should not be used for real lending decisions."
)
