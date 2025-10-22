# app.py
import os
import json
import pandas as pd
import psycopg2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Optional: AI
try:
    import openai
except Exception:
    openai = None

a=load_dotenv()
print(a)  

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  

if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

# --- Helper DB functions ---------------------------------------------------
def get_conn():
    if DATABASE_URL is None:
        raise RuntimeError("Set DATABASE_URL environment variable.")
    return psycopg2.connect(DATABASE_URL)

def ensure_talent_benchmarks_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS talent_benchmarks (
              job_vacancy_id serial PRIMARY KEY,
              role_name text,
              job_level text,
              role_purpose text,
              selected_talent_ids text[],
              weights_config jsonb,
              created_at timestamp DEFAULT now()
            );
            """)
            conn.commit()

def insert_talent_benchmark(role_name, job_level, role_purpose, selected_ids, weights_json):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO talent_benchmarks (role_name, job_level, role_purpose, selected_talent_ids, weights_config)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING job_vacancy_id;
            """, (role_name, job_level, role_purpose, selected_ids, json.dumps(weights_json) if weights_json else None))
            job_id = cur.fetchone()[0]
            conn.commit()
            return job_id

def run_matching_query(job_vacancy_id, target_role=None):
    """Runs the parameterized matching SQL. Returns pandas DataFrame."""
    sql = """
    WITH tb AS (
      SELECT *
      FROM talent_benchmarks
      WHERE job_vacancy_id = %s
    ),
    selected_benchmarks AS (
      SELECT unnest(selected_talent_ids) AS employee_id
      FROM tb
    ),
    benchmark_scores AS (
      SELECT
        m.tgv_name,
        es.tv_name,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY es.score) AS baseline_score
      FROM employee_score es
      JOIN selected_benchmarks sb ON sb.employee_id = es.employee_id
      JOIN tv_meta m ON es.tv_name = m.tv_name
      GROUP BY m.tgv_name, es.tv_name
    ),
    tv_match AS (
      SELECT
        e.employee_id,
        e.fullname,
        e.directorate_name,
        e.position_name,
        e.grade_name,
        m.tgv_name,
        es.tv_name,
        b.baseline_score,
        es.score AS user_score,
        CASE
          WHEN es.score >= b.baseline_score THEN 100.0
          ELSE ROUND((es.score::numeric / NULLIF(b.baseline_score,0) * 100)::numeric, 2)
        END AS tv_match_rate,
        COALESCE(
          (tb.weights_config ->> es.tv_name)::numeric,
          (tb.weights_config ->> m.tgv_name)::numeric,
          1.0
        ) AS tv_weight
      FROM employee_score es
      JOIN employee e ON e.employee_id = es.employee_id
      JOIN tv_meta m ON es.tv_name = m.tv_name
      JOIN benchmark_scores b ON b.tv_name = es.tv_name
      CROSS JOIN tb
      WHERE e.position_name = tb.role_name
    ),
    tgv_match AS (
      SELECT
        tm.employee_id,
        tm.tgv_name,
        ROUND(
          SUM(tm.tv_match_rate * tm.tv_weight) / NULLIF(SUM(tm.tv_weight),0)
        ::numeric, 2) AS tgv_match_rate,
        COALESCE((tb.weights_config ->> tm.tgv_name)::numeric, 1.0) AS tgv_weight_for_final
      FROM tv_match tm
      JOIN tb ON true
      GROUP BY tm.employee_id, tm.tgv_name, tb.weights_config
    ),
    final_match AS (
      SELECT
        employee_id,
        ROUND(
          SUM(tgv_match_rate * tgv_weight_for_final) / NULLIF(SUM(tgv_weight_for_final),0)
        ::numeric, 2) AS final_match_rate
      FROM tgv_match
      GROUP BY employee_id
    )
    SELECT
      tm.employee_id AS candidate_id,
      tm.fullname,
      tm.directorate_name,
      tm.position_name,
      tm.grade_name,
      tm.tgv_name,
      tm.tv_name,
      tm.baseline_score,
      tm.user_score,
      tm.tv_match_rate,
      tg.tgv_match_rate,
      fm.final_match_rate
    FROM tv_match tm
    JOIN tgv_match tg
      ON tm.employee_id = tg.employee_id
      AND tm.tgv_name = tg.tgv_name
    JOIN final_match fm
      ON tm.employee_id = fm.employee_id
    ORDER BY fm.final_match_rate DESC, tm.employee_id, tm.tgv_name, tm.tv_name;
    """
    try:
        with get_conn() as conn:
            df = pd.read_sql(sql, conn, params=(job_vacancy_id,))
            if target_role:
                df = df[df["position_name"].str.lower() == target_role.lower()]
        return df
    except Exception as e:
        st.error(f"SQL Error: {e}")
        raise


# --- AI Job Profile generator ------------------------------------------------
def generate_ai_job_profile(role_name, job_level, role_purpose, selected_ids):
    """Uses OpenAI (if available) to generate job profile text. If API not available, returns a template."""
    prompt = f"""
You are a concise recruiting writer. Generate a short job profile (job requirements, description and key competencies)
for role:
Role name: {role_name}
Job level: {job_level}({grade_label})
Role purpose: {role_purpose}
Selected benchmark employee IDs: {', '.join(selected_ids)}

Provide:
- Top 10 Job requirements (in bullet list, could be harskill, softskill, and language needed) with the detail of minimum competency (example: SQL expertise: complex joins, window functions, CTEs, performance tuning basics.)
- A short paragraph describing job description, what the role will do with the tools required and what the expected outcome.
- Key competencies for the role (list hardskill only, without detail)
Keep it business-ready and succinct.
"""
    if openai and OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":"You are a helpful assistant."},
                          {"role":"user","content":prompt}],
                max_tokens=450,
                temperature=0.3
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(AI generation failed: {e})\n\nFallback template:\n" + fallback_template(role_name, job_level, role_purpose, selected_ids)
    else:
        return fallback_template(role_name, job_level, role_purpose, selected_ids)

def fallback_template(role_name, job_level, role_purpose, selected_ids):
    bullets = [
        "SQL expertise: complex joins, window functions, CTEs, query performance tuning",
        "R or Python for analysis (pandas/tidyverse, numpy, scikit-learn)",
        "BI tooling (Tableau/Power BI/Looker) and visualization best practices",
        "Data modeling fundamentals (star schema, SCDs, metrics layer)",
        "Statistical thinking: hypothesis testing, A/B evaluation, causal caveats",
        "Communication & storytelling for non-technical stakeholders",
        "Attention to data quality, lineage, and reproducible analyses",
        "Bias-awareness: sampling, survivorship, and mitigation practices"
    ]
    text = f"Role: {role_name} (Level: {job_level})\nPurpose: {role_purpose}\n\nTop requirements:\n"
    text += "\n".join([f"- {b}" for b in bullets])
    text += f"\n\nSelected benchmark employees: {', '.join(selected_ids)}"
    return text

# --- Streamlit UI -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="Talent Match Dashboard")
st.title("Talent Match Intelligence — Interactive")

ensure_talent_benchmarks_table()

with st.sidebar:
    st.header("Create / Run Job Vacancy")
    role_name = st.text_input("Role name", value="Brand Executive")
    job_level = st.selectbox("Grade level", ["3", "4", "5"], index=1)
    # Context of grade
    grade_map = {"3": "Junior", "4": "Middle", "5": "Senior"}
    grade_label = grade_map.get(job_level, "Unknown")
    role_purpose = st.text_area("Role purpose", value="Develope and execute brand strategies that enhance brand awareness, drive customer engagement, and increase market share.")
    selected_ids_input = st.text_input("Selected benchmark employee IDs (comma separated)", value="EMP100605,EMP101226")
    selected_ids = [s.strip() for s in selected_ids_input.split(",") if s.strip()]
    weights_json_input = st.text_area("Optional Weights, there are Cognitive, Decision, Followership, Leadership, Discipline, and Social. Example:\n{\"Leadership\":0.3,\"Cognitive Ability\":0.4}", height=120)
    weights_json = None
    if weights_json_input:
        try:
            weights_json = json.loads(weights_json_input)
        except Exception as e:
            st.error(f"Invalid weights JSON: {e}")

    run_button = st.button("Save & Run")

if run_button:
    if not selected_ids:
        st.error("Provide at least one benchmark employee ID.")
    else:
        try:
            job_id = insert_talent_benchmark(role_name, job_level, role_purpose, selected_ids, weights_json)
            st.success(f"Saved job_vacancy_id = {job_id}")
            # Run matching
            df = run_matching_query(job_id)
            st.session_state["last_job_id"] = job_id
            st.session_state["last_df"] = df
        except Exception as e:
            st.exception(e)

# If have previous result, show it
df = st.session_state.get("last_df")
job_id = st.session_state.get("last_job_id")

if df is None and st.checkbox("Load latest benchmark from DB"):
    # option to select a job vacancy id from DB
    with get_conn() as conn:
        vacancies = pd.read_sql("SELECT job_vacancy_id, role_name, created_at FROM talent_benchmarks ORDER BY created_at DESC LIMIT 50", conn)
    st.dataframe(vacancies)
    chosen = st.number_input("Enter job_vacancy_id to load", min_value=1, step=1)
    if st.button("Load"):
        try:
            df = run_matching_query(int(chosen))
            job_id = int(chosen)
            st.session_state["last_df"] = df
            st.session_state["last_job_id"] = job_id
        except Exception as e:
            st.error(e)

if df is not None:
    st.markdown(f"### Results for job_vacancy_id = **{job_id}**")
    # Show AI generated job profile
    with st.expander("AI-generated Job Profile", expanded=True):
        ai_profile = generate_ai_job_profile(role_name, job_level, role_purpose, selected_ids) if job_id else generate_ai_job_profile(role_name, job_level, role_purpose, selected_ids)
        st.markdown(ai_profile.replace("\n", "  \n"))

    # === Ranked Talent List with strengths & gaps ===
    st.subheader("Ranked Talent List — Top Candidates")

    # Employee-level summary
    emp_summary = (
        df.groupby(["candidate_id", "fullname", "position_name", "directorate_name", "grade_name"])
        .agg(final_match_rate=("final_match_rate", "first"))
        .reset_index())

    # Identify strengths (TGVs>baseline)
    strengths = (
    df[df["user_score"] >= df["baseline_score"]]
    .groupby(["candidate_id", "tgv_name"])["user_score"]
    .mean()
    .reset_index()
)
    strengths = (
    strengths.groupby("candidate_id")["tgv_name"]
    .apply(lambda x: ", ".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"tgv_name": "strengths"})
)
    gaps = (
    df[df["user_score"] < df["baseline_score"]]
    .groupby(["candidate_id", "tgv_name"])["user_score"]
    .mean()
    .reset_index()
)
    gaps = (
    gaps.groupby("candidate_id")["tgv_name"]
    .apply(lambda x: ", ".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"tgv_name": "gaps"})
)
    # Identify gaps (bottom TVs below baseline)
    gaps = (
        df[df["tv_match_rate"] < 100]
        .groupby("candidate_id")["tv_name"]
        .apply(lambda x: ", ".join(x.head(3)))
        .reset_index()
            .rename(columns={"tv_name": "gaps"})
)

# Merge all
    emp_summary = (
     emp_summary.merge(strengths, on="candidate_id", how="left")
               .merge(gaps, on="candidate_id", how="left")
)

# Display in Streamlit
    st.dataframe(
        emp_summary.rename(columns={
        "candidate_id": "Employee ID",
        "fullname": "Name",
        "position_name": "Position",
        "grade_name": "Grade",
        "final_match_rate": "Match Rate (%)",
        "strengths": "Top Strengths (TGVs/TVs)",
        "gaps": "Development Gaps"
    }).sort_values("Match Rate (%)", ascending=False).head(30)
)

# Top strengths/gaps across TGVs (average across top N candidates)
    st.subheader("TGV Summary (Top 5 Candidates by Final Match Rate)")
    top5_ids = emp_summary.sort_values("final_match_rate", ascending=False).head(5)["candidate_id"].tolist()
    tgv_df_top5 = df[df["candidate_id"].isin(top5_ids)]
    pivot_tgv = tgv_df_top5.pivot_table(
        index="tgv_name",
        columns="candidate_id",
        values="tgv_match_rate",
        aggfunc="mean" 
)


    fig_heat = px.imshow(
        pivot_tgv.fillna(0),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Candidate", y="TGV", color="Match Rate (%)")
)
    st.plotly_chart(fig_heat, use_container_width=True)

# Bar: final match rates
    st.subheader("Top candidates by final match rate")
    fig_bar = px.bar(
    emp_summary.sort_values("final_match_rate", ascending=False).head(20),
    x="candidate_id", y="final_match_rate", text="final_match_rate"
)
    fig_bar.update_layout(yaxis_title="Final Match Rate (%)", xaxis_title="Candidate ID")
    st.plotly_chart(fig_bar, use_container_width=True)

# Radar per top candidate (show top 3)
    # Radar per top candidate (show top 3)
    top_candidates = emp_summary.sort_values("final_match_rate", ascending=False).head(3)["candidate_id"].tolist()
    if top_candidates:
        st.subheader("Radar: Top candidates TGV profile")

    # Compute benchmark average TGV profile
        benchmark_tgv = (
            df[df["candidate_id"].isin(selected_ids)]
            .groupby("tgv_name")["tgv_match_rate"]
            .mean()
            .reset_index()
    )

        for cid in top_candidates:
            r = df[df["candidate_id"] == cid].groupby("tgv_name")["tgv_match_rate"].mean().reset_index()
            if r.empty:
              continue

            fig = go.Figure()

        # Candidate radar
            fig.add_trace(go.Scatterpolar(
                r=r["tgv_match_rate"].tolist() + [r["tgv_match_rate"].tolist()[0]],
                theta=list(r["tgv_name"]) + [r["tgv_name"].tolist()[0]],
                fill='toself',
                name=f"Candidate {cid}",
                line_color="blue"
        ))

        # Benchmark radar
            fig.add_trace(go.Scatterpolar(
                r=benchmark_tgv["tgv_match_rate"].tolist() + [benchmark_tgv["tgv_match_rate"].tolist()[0]],
                theta=list(benchmark_tgv["tgv_name"]) + [benchmark_tgv["tgv_name"].tolist()[0]],
                fill='toself',
                name="Benchmark Avg",
                line_color="red",
                opacity=0.5
        ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title=f"Candidate {cid} vs Benchmark",
                showlegend=True
        )
            st.plotly_chart(fig, use_container_width=True)

# Detailed table (employee x TV)
    st.subheader("Detailed rows (employee × TV)")
    st.dataframe(df)

# Allow download of CSV
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name=f"talent_match_job_{job_id}.csv", mime="text/csv")

else:
    st.info("No results yet — create and run a job vacancy on the left (Save & Run).")