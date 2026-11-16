import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank_deposit_pipeline.pkl")

# ترتيب الأعمدة زي ما الموديل اتدرّب عليه بالظبط
FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome",
]
NUM_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CAT_COLS = [c for c in FEATURES if c not in NUM_COLS]

# الاختيارات دي هي نفس الـ categories اللي الـ OneHotEncoder اتدرّب عليها
CHOICES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "other", "success", "unknown"],
}

# الموديل StackingClassifier جواه 5 Pipelines (SVC, ExtraTrees, RandomForest, XGBoost, LightGBM)
# وكل واحد فيه الـ preprocessing بتاعه (imputer + scaler + one-hot)، فمش محتاج أي معالجة يدوية هنا
model = joblib.load(MODEL_PATH)

# التنبؤ بعينة واحدة أسرع بكتير من غير multi-threading
for est in [model, *getattr(model, "estimators_", [])]:
    for obj in ([est] + ([est.steps[-1][1]] if hasattr(est, "steps") else [])):
        try:
            if hasattr(obj, "n_jobs"):
                obj.n_jobs = 1
        except Exception:
            pass


def predict_deposit(*values):
    row = dict(zip(FEATURES, values))

    # الخانة الفاضية بتيجي None، والـ imputers جوه الموديل بتعوّضها
    for c in NUM_COLS:
        v = row[c]
        row[c] = np.nan if v is None or v == "" else float(v)
    for c in CAT_COLS:
        v = row[c]
        row[c] = np.nan if v is None or v == "" else v

    nums = [row[c] for c in NUM_COLS]
    if all(pd.isna(v) for v in row.values()):
        return "⚠️ أدخل قيمة واحدة على الأقل."
    if any(np.isinf(v) for v in nums):
        return "⚠️ فيه قيمة غير صالحة (infinity)."
    age, day = row["age"], row["day"]
    if not np.isnan(age) and not (18 <= age <= 100):
        return "⚠️ العمر لازم يكون بين 18 و 100."
    if not np.isnan(day) and not (1 <= day <= 31):
        return "⚠️ اليوم لازم يكون بين 1 و 31."
    for c in ("duration", "campaign", "previous"):
        if not np.isnan(row[c]) and row[c] < 0:
            return f"⚠️ قيمة {c} لازم تكون موجبة (أو صفر)."
    if not np.isnan(row["pdays"]) and row["pdays"] < -1:
        return "⚠️ قيمة pdays لازم تكون -1 (ما اتصلش بيه قبل كده) أو أكبر."

    input_data = pd.DataFrame([row], columns=FEATURES)
    for c in NUM_COLS:
        input_data[c] = pd.to_numeric(input_data[c], errors="coerce")
    for c in CAT_COLS:
        input_data[c] = input_data[c].astype(object)

    prob = float(model.predict_proba(input_data)[0][1])

    if prob >= 0.5:
        return f"✅ غالباً هيشترك في الوديعة (Deposit) — الاحتمال: {prob * 100:.1f}%"
    return f"🚫 غالباً مش هيشترك (No Deposit) — احتمال عدم الاشتراك: {(1 - prob) * 100:.1f}%"


inputs = [
    gr.Number(label="age (عمر العميل)", value=39),
    gr.Dropdown(CHOICES["job"], value="management", label="job (الوظيفة)"),
    gr.Dropdown(CHOICES["marital"], value="married", label="marital (الحالة الاجتماعية)"),
    gr.Dropdown(CHOICES["education"], value="secondary", label="education (التعليم)"),
    gr.Dropdown(CHOICES["default"], value="no", label="default (عنده ائتمان متعثر؟)"),
    gr.Number(label="balance (متوسط الرصيد السنوي بالـ Euro)", value=557),
    gr.Dropdown(CHOICES["housing"], value="no", label="housing (قرض سكني؟)"),
    gr.Dropdown(CHOICES["loan"], value="no", label="loan (قرض شخصي؟)"),
    gr.Dropdown(CHOICES["contact"], value="cellular", label="contact (وسيلة التواصل)"),
    gr.Number(label="day (يوم آخر اتصال في الشهر)", value=15),
    gr.Dropdown(CHOICES["month"], value="may", label="month (شهر آخر اتصال)"),
    gr.Number(label="duration (مدة آخر مكالمة بالثواني)", value=256),
    gr.Number(label="campaign (عدد الاتصالات في الحملة الحالية)", value=2),
    gr.Number(label="pdays (أيام من آخر اتصال سابق، -1 = ما اتصلش بيه)", value=-1),
    gr.Number(label="previous (عدد الاتصالات قبل الحملة دي)", value=0),
    gr.Dropdown(CHOICES["poutcome"], value="unknown", label="poutcome (نتيجة الحملة السابقة)"),
]

# أمثلة توضيحية للتجربة بس (مش صفوف حقيقية من الداتا)
examples = [
    [32, "student", "single", "tertiary", "no", 1200, "no", "no", "cellular", 10, "mar", 480, 1, 90, 2, "success"],
    [45, "blue-collar", "married", "primary", "no", 150, "yes", "yes", "unknown", 20, "may", 60, 4, -1, 0, "unknown"],
]

demo = gr.Interface(
    fn=predict_deposit,
    inputs=inputs,
    outputs=gr.Textbox(label="نتيجة التقييم"),
    title="🏦 Bank Term Deposit Prediction",
    description=(
        "أدخل بيانات العميل لمعرفة احتمال اشتراكه في وديعة بنكية لأجل. "
        "ممكن تسيب أي خانة فاضية وهيتم تعويضها تلقائياً. "
        "⚠️ عمود duration (مدة المكالمة) بيتعرف بعد المكالمة، فالنتيجة مناسبة لتحليل ما بعد الاتصال مش لتوقع قبل الاتصال. "
        "للتجربة والتعلم فقط."
    ),
    examples=examples,
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
