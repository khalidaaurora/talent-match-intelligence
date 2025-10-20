-- Step 1: Select the specific job benchmark configuration 
  -- (contains selected benchmark employees and weighting setup)
WITH tb AS (
  SELECT *
  FROM talent_benchmarks
  WHERE job_vacancy_id = 1 -- Replace job_vacancy_id = 1 with the ID you want to evaluate
),

-- Step 2: Get the list of benchmark employees (top talents)
-- using the selected_talent_ids from the benchmark config
selected_benchmarks AS (
  SELECT unnest(selected_talent_ids) AS employee_id
  FROM tb
),

-- Step 3: Compute the median (baseline) score per Talent Variable (TV)
-- based on those benchmark employees
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

-- Step 4: Calculate each employee’s score vs. the baseline per TV
-- and assign weighted importance based on the benchmark config
tv_match AS (
  SELECT
    e.employee_id,
    e.directorate_name,
    e.position_name,
    e.grade_name,
    m.tgv_name,
    es.tv_name,
    b.baseline_score,
    es.score AS user_score,
    CASE
      -- if employee’s score >= benchmark median, cap at 100
      WHEN es.score >= b.baseline_score THEN 100.0
      ELSE ROUND((es.score::numeric / NULLIF(b.baseline_score,0) * 100)::numeric, 2)
    END AS tv_match_rate,
    -- weight priority: use TV-level weight > TGV-level > default 1
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
),

-- Step 5: Aggregate scores by TGV (Talent Group Variable)
-- weighted average of all TVs under each TGV for each employee
tgv_match AS (
  SELECT
    tm.employee_id,
    tm.tgv_name,
    ROUND(
      SUM(tm.tv_match_rate * tm.tv_weight) / NULLIF(SUM(tm.tv_weight),0)
    ::numeric, 2) AS tgv_match_rate,
    -- store TGV-level weights (if defined), else 1
    COALESCE((tb.weights_config ->> tm.tgv_name)::numeric, 1.0) AS tgv_weight_for_final
  FROM tv_match tm
  JOIN tb ON true
  GROUP BY tm.employee_id, tm.tgv_name, tb.weights_config
),

-- Step 6: Calculate final overall match rate
-- weighted average across all TGVs per employee
final_match AS (
  SELECT
    employee_id,
    ROUND(
      SUM(tgv_match_rate * tgv_weight_for_final) / NULLIF(SUM(tgv_weight_for_final),0)
    ::numeric, 2) AS final_match_rate
  FROM tgv_match
  GROUP BY employee_id
)

-- Step 7: Final output
-- Combine all results and show employee details with each TV/TGV and final scores
SELECT
  tm.employee_id,
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
