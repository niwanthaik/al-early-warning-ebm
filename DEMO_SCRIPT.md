# Demo video script (target: 4 minutes)

Record the screen with the app running (`streamlit run app.py`) at full-screen browser width.
Speak calmly; each block lists **what to show** and **what to say**. Times are cumulative.

---

### 0:00–0:25 · Introduction
**Show:** the app's title and sidebar.
**Say:** "This is an early-warning system for Sri Lankan A/L students. It predicts, before results come out,
whether a student is at risk of not passing all three subjects, which is the basic requirement for university entrance.
It was trained on the full 2020 national results, about 290,000 candidates. The model is an Explainable Boosting
Machine, a glass-box algorithm we didn't cover in lectures."

### 0:25–1:30 · Check a student (main feature)
**Show:** "Check a student" tab. Keep the default student (Combined Maths, Physics, Chemistry; CGT 40; English S; male).
**Say:** "I enter what a school knows before results: the three subjects, the Common General Test score, the General English grade,
birth year and month, gender and syllabus."
**Show:** the probability, the AT RISK flag and the "Why?" chart (EBM exact).
**Say:** "The model gives a probability of passing and flags the student as at risk. On the right is the explanation.
Because the EBM is additive, this is the model's exact reasoning, not an approximation: the Physical Science stream and a CGT score of 40 pull the chance down."
**Do:** raise the CGT score to 65 and change English to B.
**Say:** "If the CGT score is higher and English is a B, the probability rises and the flag disappears. The explanation updates to show why."
**Do:** switch the explanation toggle to **LIME**.
**Say:** "I can also explain the same prediction with LIME, a model-agnostic method. It points to the same main reasons."

### 1:30–2:05 · Fairness switch
**Do:** in the sidebar, choose **EBM (no gender)**; then switch gender female ↔ male in the form.
**Say:** "Gender is predictive in this data, but using it means at-risk girls are under-detected. This version of the model
leaves gender out: the prediction no longer changes when I switch gender, and at-risk detection becomes equal for girls and boys."

### 2:05–2:40 · Check a class
**Show:** "Check a class (CSV)" tab with the demo file.
**Say:** "A teacher can upload a whole class. The app ranks students from highest to lowest risk, and here it also scores itself
against the real results: it catches about 70% of at-risk students. The list can be downloaded."

### 2:40–3:20 · Model comparison
**Show:** "Model comparison" tab: the table, then the AUC chart.
**Say:** "I compared the EBM with the models from the lectures: logistic regression, decision tree, k-NN, random forest and XGBoost,
plus LightGBM. The EBM beats all the interpretable baselines and random forest. It's within half a percentage point of the best
black-box model, and it's the least overfitted. The bootstrap tests show these differences are statistically significant."

### 3:20–3:55 · What the model learned
**Show:** "What the model learned" tab: the shape functions, then scroll to the domain-alignment table.
**Say:** "These curves are the model. The chance of passing rises steadily with the CGT score, science streams are hardest,
and January-born students do better. That matches the 31 January school-entry cut-off, which puts them in the year-earlier cohort.
All seven checks against what we know about the A/L system agree."

### 3:55–4:15 · Ethics and close
**Show:** "Ethics & model card" tab.
**Say:** "The tool is for supportive help only, never for selection. Day of birth is never used, because together with gender and subjects
it made a quarter of students unique in the public file. Thank you."
