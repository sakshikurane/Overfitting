import math
import pickle
from pathlib import Path
 
import pandas as pd
from flask import Flask, render_template_string, request
 
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ridge_model.pkl"
 
app = Flask(__name__)
 
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
# Order must match the columns the model was trained on.
FEATURES = list(getattr(model, "feature_names_in_", [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]))
 
# Form definition: drives both the HTML form and server-side validation.
FIELDS = {
    "Age": dict(label="Age", unit="years", min=1, max=120, step=1, example=45,
                hint="Your age in years", group="about"),
    "Pregnancies": dict(label="Pregnancies", unit="times", min=0, max=20, step=1, example=2,
                        hint="Number of times pregnant (0 if not applicable)", group="about"),
    "BMI": dict(label="BMI", unit="kg/m²", min=0, max=70, step=0.1, example=31.5,
                hint="Body mass index", group="about"),
    "DiabetesPedigreeFunction": dict(label="Family history score", unit="", min=0, max=2.5, step=0.001,
                                     example=0.55,
                                     hint="Diabetes pedigree function; higher means stronger family history",
                                     group="about"),
    "Glucose": dict(label="Glucose", unit="mg/dL", min=0, max=300, step=1, example=140,
                    hint="Plasma glucose, 2 hours after an oral glucose test", group="blood"),
    "BloodPressure": dict(label="Blood pressure", unit="mm Hg", min=0, max=200, step=1, example=80,
                          hint="Diastolic (the lower number)", group="blood"),
    "Insulin": dict(label="Insulin", unit="µU/mL", min=0, max=900, step=1, example=130,
                    hint="2-hour serum insulin", group="blood"),
    "SkinThickness": dict(label="Skin thickness", unit="mm", min=0, max=100, step=1, example=30,
                          hint="Triceps skin fold thickness", group="blood"),
}
 
GROUPS = [
    ("about", "About you"),
    ("blood", "Blood tests and vitals"),
]
 
 
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Diabetes risk check</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
:root {
  --sea: #0b3c49;
  --sea-deep: #072a34;
  --mist: #eef5f4;
  --card: #ffffff;
  --ink: #12262b;
  --muted: #5a6f74;
  --line: #d5e3e1;
  --teal: #0e8a8f;
  --mint: #6fd6a0;
  --amber: #f2b441;
  --coral: #ff6f59;
  --radius-lg: 28px;
  --radius-md: 12px;
}
 
* { box-sizing: border-box; }
 
html { color-scheme: light; }
 
body {
  margin: 0;
  font-family: "Figtree", system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(900px 500px at 100% 0%, #d8ece9 0%, transparent 60%),
    var(--mist);
  min-height: 100vh;
  line-height: 1.5;
}
 
.shell {
  max-width: 1120px;
  margin: 0 auto;
  padding: clamp(16px, 4vw, 48px);
  display: grid;
  grid-template-columns: minmax(300px, 380px) 1fr;
  gap: clamp(16px, 3vw, 32px);
  align-items: start;
}
 
/* ---------- Left panel ---------- */
.panel {
  position: sticky;
  top: 24px;
  background: linear-gradient(170deg, var(--sea) 0%, var(--sea-deep) 100%);
  color: #e9f4f2;
  border-radius: var(--radius-lg);
  padding: 32px 28px 28px;
  --accent: var(--mint);
  overflow: hidden;
}
.panel.is-low      { --accent: var(--mint); }
.panel.is-moderate { --accent: var(--amber); }
.panel.is-high     { --accent: var(--coral); }
 
.drop {
  display: block;
  width: 34px; height: 34px;
  background: var(--coral);
  border-radius: 0 50% 50% 50%;
  transform: rotate(45deg);
  margin: 4px 0 22px 4px;
}
 
h1 {
  font-family: "Bricolage Grotesque", "Figtree", sans-serif;
  font-weight: 700;
  font-size: clamp(2rem, 4vw, 2.6rem);
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 0 0 12px;
}
 
.lede { margin: 0; color: #b9d3d0; max-width: 32ch; }
 
/* Gauge */
.gauge { position: relative; margin: 28px 0 8px; }
.gauge svg { width: 100%; display: block; }
.gauge__track,
.gauge__value {
  fill: none;
  stroke-width: 16;
  stroke-linecap: round;
}
.gauge__track { stroke: rgba(255, 255, 255, 0.12); }
.gauge__value {
  stroke: var(--accent);
  stroke-dasharray: 251.33;
  stroke-dashoffset: var(--offset);
}
.is-low .gauge__value,
.is-moderate .gauge__value,
.is-high .gauge__value {
  animation: fill 1s cubic-bezier(.2,.8,.2,1) both;
}
@keyframes fill { from { stroke-dashoffset: 251.33; } }
 
.gauge__label {
  position: absolute;
  left: 0; right: 0; bottom: 4px;
  text-align: center;
  display: flex; flex-direction: column; align-items: center;
}
.gauge__label strong {
  font-family: "Bricolage Grotesque", sans-serif;
  font-size: 3rem; line-height: 1; font-weight: 700;
  color: #fff;
}
.gauge__label small { font-size: 1.2rem; margin-left: 2px; color: var(--accent); }
.gauge__label span { font-size: .95rem; color: var(--accent); font-weight: 600; margin-top: 4px; }
 
.verdict { margin: 18px 0 20px; color: #d6e8e5; }
.verdict em { font-style: normal; color: #fff; font-weight: 600; }
 
.disclaimer {
  margin: 0;
  padding-top: 16px;
  border-top: 1px solid rgba(255,255,255,.14);
  font-size: .82rem;
  color: #8fb0ac;
}
 
/* ---------- Form ---------- */
.form-card {
  background: var(--card);
  border-radius: var(--radius-lg);
  padding: clamp(22px, 3.5vw, 40px);
  box-shadow: 0 1px 0 var(--line), 0 24px 48px -28px rgba(11, 60, 73, .35);
}
 
fieldset { border: 0; padding: 0; margin: 0 0 30px; }
 
legend {
  font-family: "Bricolage Grotesque", sans-serif;
  font-weight: 700;
  font-size: 1.3rem;
  letter-spacing: -0.01em;
  padding: 0;
  margin-bottom: 16px;
  color: var(--sea);
}
 
.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px 20px;
}
 
.field label { display: block; font-weight: 600; font-size: .95rem; margin-bottom: 6px; }
 
.input {
  display: flex; align-items: center;
  background: var(--mist);
  border: 2px solid transparent;
  border-radius: var(--radius-md);
  transition: border-color .15s, background .15s;
}
.input:focus-within { border-color: var(--teal); background: #fff; }
 
.input input {
  flex: 1; min-width: 0;
  border: 0; background: transparent;
  font: inherit; font-size: 1.05rem; font-weight: 500;
  color: var(--ink);
  padding: 12px 14px;
  outline: none;
}
.unit { padding-right: 14px; color: var(--muted); font-size: .85rem; white-space: nowrap; }
 
.field small { display: block; margin-top: 6px; color: var(--muted); font-size: .8rem; line-height: 1.35; }
 
.errors {
  background: #fff1ee;
  border-left: 4px solid var(--coral);
  border-radius: var(--radius-md);
  padding: 14px 18px;
  margin-bottom: 26px;
  color: #7a2a1d;
}
.errors p { margin: 0 0 6px; font-weight: 600; }
.errors ul { margin: 0; padding-left: 18px; }
 
.actions { display: flex; flex-wrap: wrap; gap: 12px; }
 
.btn {
  font: inherit; font-weight: 600;
  border-radius: 999px;
  padding: 14px 28px;
  cursor: pointer;
  border: 2px solid transparent;
  transition: transform .1s, background .15s, border-color .15s;
}
.btn:active { transform: translateY(1px); }
.btn--primary { background: var(--coral); color: #fff; }
.btn--primary:hover { background: #f25c45; }
.btn--ghost { background: transparent; color: var(--sea); border-color: var(--line); }
.btn--ghost:hover { border-color: var(--teal); }
 
.btn:focus-visible,
.input input:focus-visible { outline: 3px solid var(--teal); outline-offset: 2px; }
 
/* ---------- Responsive ---------- */
@media (max-width: 860px) {
  .shell { grid-template-columns: 1fr; }
  .panel { position: static; }
}
@media (max-width: 520px) {
  .grid { grid-template-columns: 1fr; }
  .btn { width: 100%; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
 
</style>
</head>
<body>
<main class="shell">
 
  <!-- Left: intro + result gauge -->
  <aside class="panel {{ 'is-' ~ result.level if result else '' }}" aria-live="polite">
    <div class="panel__top">
      <span class="drop" aria-hidden="true"></span>
      <h1>Diabetes risk check</h1>
      <p class="lede">Enter eight health measurements and get an instant risk score from a trained Ridge regression model.</p>
    </div>
 
    <div class="gauge">
      <svg viewBox="0 0 200 118" role="img"
           aria-label="{% if result %}Risk score {{ result.percent }} percent{% else %}Risk score will appear here{% endif %}">
        <path class="gauge__track" d="M20 100 A80 80 0 0 1 180 100" />
        <path class="gauge__value" d="M20 100 A80 80 0 0 1 180 100"
              {% if result %}style="--offset: {{ (251.33 * (1 - result.score)) | round(2) }}"{% else %}style="--offset: 251.33"{% endif %} />
      </svg>
      <div class="gauge__label">
        {% if result %}
          <strong>{{ result.percent }}<small>%</small></strong>
          <span>{{ result.title }}</span>
        {% else %}
          <strong>--</strong>
          <span>Waiting for your numbers</span>
        {% endif %}
      </div>
    </div>
 
    {% if result %}
      <p class="verdict">{{ result.message }}</p>
    {% else %}
      <p class="verdict">Your score will appear here after you press <em>Check my risk</em>.</p>
    {% endif %}
 
    <p class="disclaimer">This is a screening aid, not a diagnosis. The score is a model estimate and is not a medical test. Please see a doctor for advice.</p>
  </aside>
 
  <!-- Right: form -->
  <section class="form-card">
    <form method="post" novalidate id="risk-form">
 
      {% if errors %}
      <div class="errors" role="alert">
        <p>Fix these to continue:</p>
        <ul>{% for e in errors %}<li>{{ e }}</li>{% endfor %}</ul>
      </div>
      {% endif %}
 
      {% for gkey, gtitle in groups %}
      <fieldset>
        <legend>{{ gtitle }}</legend>
        <div class="grid">
          {% for name, f in fields.items() if f.group == gkey %}
          <div class="field">
            <label for="{{ name }}">{{ f.label }}</label>
            <div class="input">
              <input type="number" id="{{ name }}" name="{{ name }}"
                     min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}"
                     value="{{ values[name] }}" inputmode="decimal"
                     data-example="{{ f.example }}" required>
              {% if f.unit %}<span class="unit">{{ f.unit }}</span>{% endif %}
            </div>
            <small>{{ f.hint }}</small>
          </div>
          {% endfor %}
        </div>
      </fieldset>
      {% endfor %}
 
      <div class="actions">
        <button type="submit" class="btn btn--primary">Check my risk</button>
        <button type="button" class="btn btn--ghost" id="fill-example">Fill sample values</button>
      </div>
    </form>
  </section>
 
</main>
 
<script>
  document.getElementById('fill-example').addEventListener('click', function () {
    document.querySelectorAll('#risk-form input[data-example]').forEach(function (el) {
      el.value = el.dataset.example;
    });
  });
</script>
</body>
</html>
"""
 
 
def interpret(score: float) -> dict:
    """Turn the model's raw output into a friendly result."""
    if score < 0.35:
        return dict(level="low", title="Lower risk",
                    message="Your numbers point to a lower likelihood of diabetes. "
                            "Keep up regular activity, balanced meals and routine check-ups.")
    if score < 0.50:
        return dict(level="moderate", title="Moderate risk",
                    message="Some of your numbers lean toward elevated risk. "
                            "A conversation with a doctor and a fasting glucose or HbA1c test is a good next step.")
    return dict(level="high", title="Higher risk",
                message="Your numbers point to a higher likelihood of diabetes. "
                        "Please book a screening with a doctor for a proper diagnosis.")
 
 
def parse_form(form):
    """Return (values_as_str, parsed_floats, errors)."""
    values, parsed, errors = {}, {}, []
    for name, spec in FIELDS.items():
        raw = (form.get(name) or "").strip()
        values[name] = raw
        if raw == "":
            errors.append(f"{spec['label']} is required.")
            continue
        try:
            num = float(raw)
            if not math.isfinite(num):
                raise ValueError
        except ValueError:
            errors.append(f"{spec['label']} must be a number.")
            continue
        if not spec["min"] <= num <= spec["max"]:
            errors.append(f"{spec['label']} must be between {spec['min']} and {spec['max']}.")
            continue
        parsed[name] = num
    return values, parsed, errors
 
 
@app.route("/", methods=["GET", "POST"])
def index():
    values = {name: "" for name in FIELDS}
    errors, result = [], None
 
    if request.method == "POST":
        values, parsed, errors = parse_form(request.form)
        if not errors:
            row = pd.DataFrame([[parsed[f] for f in FEATURES]], columns=FEATURES)
            raw = float(model.predict(row)[0])
            score = min(max(raw, 0.0), 1.0)      # clamp the regression output to 0-1
            result = interpret(score)
            result.update(score=score, percent=round(score * 100))
 
    return render_template_string(
        HTML_TEMPLATE,
        fields=FIELDS, groups=GROUPS, values=values, errors=errors, result=result,
    )
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
