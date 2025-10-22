# Talent Match Intelligence

This project analyzes employee success factors and builds an AI-powered matching system to identify potential high performers based on psychological and behavioral data.

---

## 🚀 Project Overview
This case study aims to:
- Identify success patterns from assessment data (Deliverable #1)
- Build SQL logic for benchmark-based talent matching (Deliverable #2)
- Develop an interactive dashboard to visualize candidate-to-benchmark match results

---

## 🧠 Success Factor Discovery
Exploration and modeling were done in Python to determine key predictors of success using Random Forest feature importance.

Final success factors grouped into **Talent Group Variables (TGV)**:
| TGV | Included Talent Variables (TV) |
|------|-------------------------------|
| Social | SEA, PAPI_X |
| Discipline | QDD, PAPI_T |
| Decision | IDS |
| Leadership | PAPI_P |
| Followership | PAPI_F |
| Cognitive | Pauli, Faxtor |

---

## 🧮 SQL Logic
SQL scripts implement benchmark median calculation, match rate computation, and final weighted match scoring using modular CTEs.

📂 **Folder**: `sql/`

| File | Description |
|------|--------------|
| `1_employee_benchmark_match_rate.sql` | Calculates match rates vs. high performers |
| `2_talent_benchmark.sql` | Defines the `talent_benchmarks` table |
| `3_add_talent_benchmark.sql` | Inserts sample benchmark for “Brand Executive” |
| `4_benchmark_based_matching_score.sql` | Final SQL algorithm with weighted TGV logic |

---

## 💻 Streamlit App
Interactive dashboard built with Streamlit to visualize:
- Candidate-to-benchmark match rates
- TGV-level and overall fit scores
- Filtering by role and level

📂 **Folder**: `app/`

### Run Locally
```bash
pip install -r talent-match-app/requirements.txt
streamlit run talent-match-app/app.py
