# MSIN0097: Comparative analysis of autonomous AI coding agents — flight delay prediction

**Module:** MSIN0097 | **Submission:** March 2026

---

## 1. Executive summary

This report evaluates three AI coding agents — Claude (Anthropic), Codex (OpenAI), and Antigravity — against an identical eight-task machine learning pipeline applied to the 2015 US DOT Aviation Dataset (5.8 million domestic flight records). The pipeline covers the full ML lifecycle: data ingestion, exploratory analysis, baseline classification, model optimisation, leakage auditing, reproducible packaging, and production documentation. Each agent worked independently on the same task brief, running in separate OS environments, under identical seed and sampling constraints.

Across all eight tasks, Claude was the only agent to produce zero ML errors. Codex introduced look-ahead data leakage in Task 3 and then compounded it across Task 4, artificially inflating its ROC-AUC by 26.85 percentage points before self-correcting in Task 5. Antigravity accumulated five documented errors across the evaluation, the most severe of which was including `ARRIVAL_DELAY` — the source column from which the binary prediction target is derived — as a training feature in Task 4. The result was a ROC-AUC of 1.0000 with no predictive content: a model that looked up rather than learned. After mandatory leakage correction in Task 5, all three honest models converge to ROC-AUC 0.66–0.71. This range is the true production ceiling for 24-hour advance delay prediction on this dataset, and any figure materially above it should be treated as a leakage signal.

---

## 2. Methodology and harness design

The evaluation harness was built to isolate agent decision-making from environmental noise. Three constraints were enforced uniformly across all agents.

All agents used `random_state=42` throughout: train/test splits, cross-validation folds, SMOTE where applied, and hyperparameter search. This ensures performance differences reflect modelling choices rather than sampling variation. Each agent resolved data paths dynamically using `pathlib.Path` anchored to the repository root, making all scripts portable across the three test machines — Apple Silicon macOS for Claude, Intel macOS for Codex, and Windows for Antigravity. The full dataset was stored in a git-ignored `data/` vault; no script contains a hardcoded absolute path. Row counts were controlled by a 500,000-row stratified sample (`random_state=42`), applied before any modelling step.

The central data integrity rule is the 24-hour prediction constraint: no feature may be used that is unknowable at the time of scheduling, roughly 24 hours before scheduled departure. This excludes all post-gate-push-back operational columns — `DEPARTURE_DELAY`, `TAXI_OUT`, `ELAPSED_TIME`, `AIR_TIME`, `TAXI_IN`, and five post-arrival delay attribution codes among others. This rule was stated in each task brief but not provided as a checklist. Agents were expected to apply it as a consistent modelling principle across the full pipeline, not just in the tasks where the brief mentioned leakage explicitly.

Four traps were embedded to test how robustly each agent held this constraint. The *domain logic trap* in Task 1 tested whether agents recognised that `ARRIVAL_DELAY = NaN` for a cancelled flight is structurally undefined — not a missingness problem amenable to imputation. A *look-ahead leakage trap* recurred across Tasks 1.5, 2, 3, and 4. In Task 2, a *volume-versus-rate trap* tested whether agents ranked airlines by raw delay counts rather than true delay rate. Task 4 embedded a *target leakage scenario*: would an agent notice it had included `ARRIVAL_DELAY` as a training feature after deriving a binary target directly from it?

All outputs were evaluated across six dimensions: correctness, statistical validity, reproducibility, code quality, computational efficiency, and safety/compliance.

---

## 3. Task definitions and execution evidence

The pipeline consists of eight tasks. Each is defined below, followed by a cross-agent execution summary table.

**Task 1 — Data ingestion and missingness** required agents to load the raw 5.8M-row flight dataset, LEFT JOIN with `airlines.csv` and `airports.csv`, apply domain logic to `ARRIVAL_DELAY = NaN` for cancelled flights, apply memory-efficient dtypes, and subsample to 500,000 rows.

**Task 1.5 — Visual EDA** required three plots investigating drivers of `ARRIVAL_DELAY > 15 min`, restricted to pre-departure features only (24-hour rule enforced).

**Task 2 — Structured EDA** covered four experiments: worst-offender ranking by delay rate (not raw count), descriptive statistics including P90 and P99 of the skewed target, correlation analysis with correct categorical feature handling, and a VIF multicollinearity audit plus formal heteroscedasticity diagnosis — all on the clean pre-departure feature set.

**Task 3 — Baseline classification** required a binary classifier predicting `ARRIVAL_DELAY > 15`, trained on pre-departure features only, evaluated on ROC-AUC, PR-AUC, and F1-Score against a held-out test set.

**Task 4 — Model optimisation** asked agents to improve on their Task 3 baseline through feature engineering and hyperparameter tuning without introducing or inheriting leakage from prior tasks.

**Task 5 — Leakage audit** required each agent to formally self-audit its Task 4 model, identify and remove all look-ahead or target-variable leakage, retrain on a clean pre-departure feature set, and verify the result using permutation importance.

**Task 6 — Reproducible packaging** required a fully version-pinned `requirements.txt`, a cross-platform `run_pipeline.py` with dry-run capability, and a `README_RUN.md` covering macOS and Windows environments.

**Task 8 — Production documentation** required a `README.md` documenting repository architecture and setup, and a `model_card.md` with actual evaluated metrics, a complete feature inventory, and a temporal limitations section addressing the 2015 dataset vintage.

### Execution summary

| Task | Objective | Claude | Codex | Antigravity |
|---|---|---|---|---|
| **1 — Ingestion** | Load, JOIN, handle cancelled-flight NaN, optimise dtypes, sample 500k | ✅ Correct domain logic (dropped cancelled flights); autonomous 83% dtype compression; ❌ loaded full 5.8M rows (agent overrode sampling constraint) | ✅ Correct domain logic; ✅ 500k sampling (only agent to comply); ⚠️ standard dtypes, no aggressive downcast | ❌ `SimpleImputer(strategy='median')` applied to cancelled flights — imputed fictional arrival times for flights that never departed; ❌ no sampling |
| **1.5 — Visual EDA** | 3 pre-departure plots on delay >15 min | ✅ No leakage; Wilson 95% CI carrier×month heatmap; temporal danger-zone plot; distance×time-of-day interaction | ✅ No leakage; Empirical Bayes route-risk frontier with bubble sizing; ❌ context dropout — missing required business insight comment | ✅ No leakage; ⚠️ structurally basic plots; implicitly self-corrected Task 1 imputation error before plotting |
| **2 — Structured EDA** | Worst offenders (rate), descriptive stats, categorical-aware correlation, VIF + heteroscedasticity | ✅ Breusch-Pagan p=8.47e-35 with LOWESS diagnostic; manual feature exclusion list; no leakage | ✅ Highest statistical rigour: Empirical Bayes posterior (Beta conjugate prior); Correlation Ratio η for categorical features; no leakage | ❌ Composite severity metric (rate × mean_delay — methodologically invalid); raw `select_dtypes` admitted `DEPARTURE_DELAY` into correlation; `DEPARTURE_DELAY` in VIF feature set; background process killed by infinite `groupby` loop |
| **3 — Baseline model** | Binary classifier, pre-departure only; ROC-AUC, PR-AUC, F1 | ✅ XGBoost; 500k; **ROC-AUC 0.7060**; PR-AUC 0.3607; F1 0.3992; zero leakage | ❌ Random Forest; 70k; **ROC-AUC ~~0.9375~~** (genuine: ~0.71); `DEPARTURE_DELAY` in feature set (Pearson r ≈ 0.93 with target); **+23.1pp artificial inflation** | ❌ Random Forest; 500k; **ROC-AUC ~~0.7471~~** (genuine: ~0.67); `TAXI_OUT` (rank #1), `ELAPSED_TIME` (rank #3), `AIR_TIME` (rank #6), `TAXI_IN` (rank #7) — 4 of top 7 features are post-departure; **+4.1pp inflation** |
| **4 — Optimisation** | Feature engineering + HP tuning; no new leakage | ✅ +0.73pp via 7 business-logic features (ROUTE, CARRIER_ROUTE, DEP_PERIOD, IS_WEEKEND, IS_PEAK_SUMMER, DISTANCE_BIN); 5-candidate manual HP search on held-out validation; **ROC-AUC 0.7108** | ❌ HistGradientBoosting upgrade; added `DEPARTURE_DELAY_CLIPPED` and `DEPARTURE_DELAY_15_PLUS` (two further derivatives of leakage column); ROC-AUC *degraded* to **~~0.9332~~** (−0.43pp — leakage saturation, noise compounding) | ❌❌ `ARRIVAL_DELAY` (column from which `DELAY_15 = ARRIVAL_DELAY > 15` is derived) included as a training feature; `DELAY_DIFF = ARRIVAL_DELAY − DEPARTURE_DELAY` also added; **ROC-AUC ~~1.0000~~** — deterministic rule lookup, not a classifier |
| **5 — Leakage audit** | Self-audit Task 4; remove leakage; retrain; verify with permutation importance | ✅ No leakage to remove; permutation importance (8 repeats, 5k subsample, ROC-AUC scoring); top feature `MONTH` = 0.0635 units; **Δ AUC = 0.0000**; honest ROC-AUC **0.7108** | ✅ Removed `DEPARTURE_DELAY` + 2 derivatives; permutation importance (5 repeats, 4k subsample); honest ROC-AUC **0.6647**; **Δ = −26.85pp** confirmed as pure leakage inflation | ✅ Removed `ARRIVAL_DELAY`, `DEPARTURE_DELAY`, and 6 further leakage cols (`AIR_TIME`, `TAXI_OUT`, `TAXI_IN`, `DELAY_DIFF`, `AVG_SPEED`, `TAXI_SUM`); honest ROC-AUC **0.6741**; **Δ = −32.59pp** — largest correction in cohort |
| **6 — Packaging** | Pinned `requirements.txt`, cross-platform runner, 3-OS README | ✅ A — 7 direct deps, all pinned with `==`; zero phantom deps; `--dry-run` and `--step` CLI flags; 3-OS README with expected output metrics table | ✅ B+ — all deps pinned; `pyarrow==23.0.1` phantom dep (unused by any script); dry-run mode; 3-OS README | ❌ B− — `pytz` listed without version pin (not in venv; will install untested version silently); `pytz` and `python-dateutil` are transitive, not direct deps; no dry-run mode; no Windows CMD section |
| **8 — Documentation** | `README.md` + `model_card.md` with exact metrics, feature list, temporal limitations | ✅ A — full `data/` vault diagram; exact Task 5 metrics (AUC 0.7108, F1 0.4050, Precision 0.2974, Recall 0.6349); COVID-19 disruption, airline consolidation, 737 MAX grounding named as 2015 temporal decay risks | ✅ B+ — exact metrics throughout; limitations section present but uses generic model-drift language with no 2015-specific callouts | ❌ C — model card contains literal unfilled placeholder: `"~[real run value from run output]"`; `DEPARTURE_DELAY` listed as a required input field — directly contradicts own Task 5 leakage audit |

**Cumulative error count:** Claude 0 | Codex 2 (self-corrected) | Antigravity 5 (2 critical, 3 high; all resolved in Task 5)

---

## 4. Comparative analysis

### Master comparative matrix

| Dimension | Claude | Codex | Antigravity |
|---|---|---|---|
| **Correctness** | ✅ Excellent — 0 errors across all 8 tasks | ⚠️ Adequate — 2 leakage errors (Tasks 3–4), fully self-corrected in Task 5 | ❌ Poor — 5 errors (2 critical, 3 high); Task 5 corrected ML errors; Task 8 contradiction unresolved |
| **Statistical validity** | ✅ Excellent — Δ AUC = 0.0000 on audit; clean permutation importance distribution | ⚠️ Adequate — highest EDA rigour (Empirical Bayes); +23.1pp AUC inflation corrected | ❌ Poor — target variable as training feature; ROC-AUC 1.0000 artefact; Δ = −32.59pp on correction |
| **Reproducibility** | ✅ A — 7 pinned direct deps; zero phantom; `--dry-run` and `--step` flags; 3-OS README | ✅ B+ — fully pinned; `pyarrow` phantom dep; dry-run mode; 3-OS README | ❌ B− — `pytz` floating and absent from venv; transitive deps listed as direct; no dry-run; no Win CMD |
| **Code quality** | ✅ Excellent — `pathlib`, thread-limit env vars, `argparse` CLI, internally consistent docs | ✅ Good — clean structure; Empirical Bayes block uncommented; phantom dep in requirements | ⚠️ Adequate — functional; no leakage guard in Task 4 feature list; placeholder in model card; contradictory docs |
| **Efficiency** | ✅ Excellent — `tree_method='hist'`, thread-limiting, 500k sampling throughout | ✅ Good — conservative 50k audit sample; `n_jobs=1`; stable throughout | ❌ Poor — infinite `os.system` loop (process killed); cloud API outage halted session mid-execution |
| **Safety/compliance** | ✅ Excellent — no credential exposure; explicit 2015 temporal decay callout (COVID, 737 MAX, consolidation) | ✅ Good — no secrets issues; generic temporal limitations only | ❌ Critical — hardcoded `ghp_...` GitHub PAT in terminal command; placeholder model card; contradictory required-field documentation |

---

### Correctness

Claude's zero-error record is the clearest differentiator in this cohort. It is the only agent that never required a correction run: domain logic held from Task 1 (cancelled flights dropped, not imputed), the 24-hour constraint was applied consistently through Tasks 3 and 4 without prompting, and the Task 8 model card contained no placeholder text or internal contradictions. Codex committed two errors, both leakage-related and both self-corrected in Task 5, with strong work across EDA and packaging on either side of those failures. Antigravity's five errors span the full evaluation: a fatal imputation error in Task 1, post-departure operational leakage in Task 3, target-variable leakage in Task 4, an unfilled model card placeholder in Task 8, and an internal contradiction that re-listed `DEPARTURE_DELAY` as a required input field after the Task 5 audit had removed it as illegal. That last error is not a coding mistake — it is a documentation failure that would actively mislead any engineer attempting to retrain the model from the submitted artefacts.

### Statistical validity

This is where the evaluation finds its most consequential results. The embedded look-ahead leakage trap functioned exactly as intended: two of the three agents introduced post-departure features into their predictive models, inflating reported AUCs by margins that would have been operationally catastrophic if left uncorrected.

Codex included `DEPARTURE_DELAY` in its Task 3 feature matrix. This column has a Pearson correlation of approximately 0.93 with `ARRIVAL_DELAY`, which means the model was not predicting delay — it was reading it. The reported ROC-AUC of 0.9375 overstates genuine predictive power by 23.1 percentage points relative to Claude's clean baseline of 0.7060. In Task 4, Codex then engineered two further derivatives of the leakage column (`DEPARTURE_DELAY_CLIPPED`, `DEPARTURE_DELAY_15_PLUS`), deepening the contamination rather than resolving it. The paradoxical result — AUC degrading from 0.9375 to 0.9332 despite added features — is symptomatic: when a model is already near-saturated by one dominant leakage variable, additional features introduce noise rather than signal.

Antigravity's Task 4 failure is categorically different in kind. By including `ARRIVAL_DELAY` itself as a training feature — the column from which `DELAY_15 = (ARRIVAL_DELAY > 15)` is algebraically derived — the model achieved ROC-AUC 1.0000. This is not a classifier; it is a conditional lookup rule that will fail on any live deployment where `ARRIVAL_DELAY` is, by definition, unavailable at prediction time. The Task 5 reality check removes all ambiguity: correcting the leakage drops the AUC by 32.59 percentage points to 0.6741, the largest single correction in the cohort and a confirmation that the Task 4 model had learned nothing about flight delay.

Claude's permutation importance audit in Task 5 — eight repeats over a 5,000-row test subsample — returned Δ AUC = 0.0000. No single feature dominates; the top feature (`MONTH`) contributes only 0.0635 AUC units. This distribution is the expected signature of a clean model trained on real scheduling signals rather than operational proxies.

### Reproducibility

Claude and Codex both submitted packaging that would survive a clean-clone test by a new engineer. Claude's `requirements.txt` lists exactly seven directly-imported libraries, all pinned with `==`, with zero transitive or phantom entries. The runner supports `--dry-run` (inspect step order without executing) and `--step` (run a single task in isolation) flags, both standard requirements for CI/CD pre-flight checks. Codex's one weakness is `pyarrow==23.0.1`, listed in requirements but unused by any pipeline script. This adds approximately 20 MB of unnecessary installation overhead and obscures the true dependency graph for future security audits. Antigravity's floating `pytz` entry is the more serious failure: `pytz` is absent from the project's virtual environment, so any fresh `pip install -r requirements.txt` installs an untested version silently. On a date-handling library, this is not theoretical — it is a routine integration failure waiting for a date change to surface it.

### Code quality

Claude's scripts are the most consistent in structure across tasks: `pathlib.Path` throughout, environment-variable thread limits with inline rationale, `matplotlib.use("Agg")` to prevent display errors in headless environments, and an `argparse` CLI with meaningful flags. Comments explain *why* choices were made, not just what the code does — the LOWESS smoother purpose is stated, the VIF singularity guard is justified. Codex is clean and well-structured but has a notable gap: the Empirical Bayes posterior block in Task 2 — the most statistically sophisticated section in the evaluation — carries no inline comments explaining the Beta conjugate prior or the shrinkage logic. This is the section a new engineer would most benefit from having annotated. Antigravity's code is functional across most tasks but lacks defensive programming where it matters. The Task 4 feature list included `ARRIVAL_DELAY` at line 69 without a guard, an assertion, or a comment — a silent correctness failure that produced a perfect AUC without triggering any runtime signal.

### Efficiency

Claude managed resources carefully throughout. `tree_method="hist"` and `OMP_NUM_THREADS=2` are appropriate choices for Apple Silicon unified memory under load, and 500k-row sampling was applied consistently from Task 1 onward. Codex was conservative — a 50k-row sample for the audit step, `n_jobs=1` for permutation importance — which increases wall-clock time but eliminates memory risk on the evaluation machine. Both agents ran stably from start to finish.

Antigravity experienced two failures of a type that a production MLOps pipeline cannot tolerate. In Task 2, `task2_eda.py` triggered an infinite `os.system` background loop after setting `observed=False` on a categorical `groupby`. This created a Cartesian product of all airline and airport category combinations — tens of millions of rows — consuming all available CPU cores with no timeout and no memory ceiling. The process could not be interrupted without terminating the entire agent session. The root cause is structural: Antigravity executes commands by injecting `os.system()` calls directly into the IDE process rather than spawning isolated subprocesses with resource controls. A runaway job therefore has no kill switch, making it incompatible with any shared or scheduled compute environment. Separately, during the same evaluation window, a cloud API outage affecting Antigravity's IDE integration dropped the agent session entirely mid-execution, requiring a full restart and resubmission of the task. Claude and Codex ran locally throughout and were unaffected by both incidents. An agent that requires a persistent live cloud connection — with no offline fallback — carries an availability risk that is structural, not incidental.

### Safety and compliance

One incident in this evaluation has no equivalent in the other agents. Antigravity attempted to authenticate to GitHub by embedding a `ghp_...` Personal Access Token directly in a terminal command string. A `ghp_` prefix identifies a GitHub PAT with repository-scope write access. Inlining it in a command exposes the credential in shell history, process lists, system logs, and any screen capture or CI/CD recording of the session. In a production environment this pattern would be caught by a static application security testing tool (`gitleaks`, `detect-secrets`), blocked by a pre-commit hook, and logged as a mandatory security incident under standard enterprise policy. The correct pattern is to inject via the `GH_TOKEN` environment variable or a secrets manager. It was not used. No other agent exposed a credential at any point during evaluation.

On model documentation compliance, Claude is the only submission to name the specific structural discontinuities that make the 2015 dataset vintage risky: COVID-19 capacity collapse, post-pandemic crew shortages, airline consolidation (the absorption of Virgin America and others), and the 737 MAX grounding. Codex's limitations section defaults to generic model-drift language. Both fall below the threshold for responsible model release on a decade-old training set, though for different reasons — Codex through omission, Antigravity through contradiction. Antigravity's model card lists `DEPARTURE_DELAY` as a required input field in the training data description, directly contradicting the Task 5 audit that removed it as an illegal post-departure feature. A stakeholder reading both documents would receive irreconcilable instructions about how to retrain the model.

---

## 5. Conclusion

The convergence finding from Task 5 deserves to be the starting point rather than the end: once leakage is corrected, all three agents produce models in the 0.66–0.71 ROC-AUC band. This is the real information ceiling for 24-hour advance delay prediction on 2015 US aviation data — significant, but far from the near-perfect figures that two of the three agents reported mid-evaluation. The distance between 0.71 and 0.93 is not a measure of model quality. It is a measure of how much a pipeline can mislead if it is not audited.

Claude navigated the full eight-task pipeline without a correctness failure. Its one deviation — loading the full 5.8M rows in Task 1 rather than the specified 500k — was an initiative error: it did more than asked, and did so soundly. The distinction matters. An agent that exceeds a brief without compromising correctness is a different failure mode from an agent that silently corrupts a training dataset. Codex demonstrated that serious ML errors — look-ahead leakage escalated across two consecutive tasks — are recoverable when an agent can self-audit cleanly. Its honest AUC of 0.6647 after correction is competitive, and its EDA work (Empirical Bayes posterior, Correlation Ratio η for categoricals) was the most statistically rigorous of the three. Antigravity showed the failure mode that matters most for autonomous deployment: errors that compound silently and produce no runtime signal. A ROC-AUC of 1.0000 does not trigger an exception. It does not log a warning. Without a mandatory leakage audit, it proceeds to the next task, and potentially to a stakeholder, as a result.

For any team considering autonomous agents in production ML pipelines, the practical implication is direct: output confidence cannot substitute for output verification. The Task 5 leakage audit is not an optional quality gate: it is a prerequisite for treating any agent-generated model as valid. At the current level of capability, the agents that perform best are those that internalize domain rules as persistent constraints rather than applying them reactively when a brief makes them explicit. That gap — between rule-following and rule-internalizing — is where the next generation of agentic ML tooling will have to close ground.

---

## Appendices

### Appendix A: Task specifications and orchestrator prompts

*[Placeholder — insert screenshots of the full task brief and orchestrator prompt for each of the eight pipeline tasks. Include the exact wording of the 24-hour prediction constraint as it appeared in the brief.]*

---

### Appendix B: Agent Error Tracker log

| Task Phase | Agent | Error Type / Severity | Description of Failure | Resolution / Outcome |
|---|---|---|---|---|
| Task 1: Ingestion | Antigravity | Domain Logic Failure / HIGH | Applied `SimpleImputer(strategy='median')` to `ARRIVAL_DELAY` for cancelled flights, imputing a fictitious arrival delay for flights that never departed. This silently corrupted the training set with statistically meaningless target values. | Self-corrected before Task 1.5 — dropped cancelled flights prior to EDA plotting, suggesting the agent detected the inconsistency without explicit prompting. |
| Task 1: Ingestion | Claude | None | — | — |
| Task 1: Ingestion | Codex | None | — | — |
| Task 2: Structured EDA | Antigravity | Look-Ahead Leakage / HIGH | Reintroduced `DEPARTURE_DELAY` into the VIF feature set in `task2_advanced_stats.py`, despite correctly excluding it in basic EDA (Task 1.5). The 24-hour prediction constraint was applied selectively rather than held as a persistent domain rule, contaminating the multicollinearity analysis. | Not self-corrected in Task 2. The leaky VIF result carried forward as an invalid statistical output. |
| Task 2: Structured EDA | Antigravity | Infinite Loop / CRITICAL | `task2_eda.py` triggered a runaway `os.system` groupby with `observed=False` on categorical columns, creating a cartesian product of tens of millions of rows. The process consumed all CPU cores and hung indefinitely with no timeout or memory ceiling. Required force-kill of the entire agent session. | Required human intervention (process kill). Agent restarted and task re-submitted. |
| Task 2: Structured EDA | Antigravity | Cloud API Outage / HIGH | IDE-integrated cloud API connection dropped mid-execution during the same evaluation window as the infinite loop, requiring a full session restart. No offline fallback was available. | Required full agent session restart. |
| Task 3: Baseline Model | Codex | Look-Ahead Leakage / CRITICAL | Included `DEPARTURE_DELAY` (Pearson r = 0.93 with `ARRIVAL_DELAY`) in the feature set at `task3_baseline_model.py:95`. The model was not predicting future delays but confirming that a flight was already late at the gate. ROC-AUC inflated by +23.1 pp (0.9375 vs. clean baseline 0.7060). | Carried forward to Task 4. Self-corrected in Task 5 audit; honest ROC-AUC dropped to 0.6647. |
| Task 3: Baseline Model | Antigravity | Look-Ahead Leakage / HIGH | Retained four post-departure operational columns (`TAXI_OUT`, `ELAPSED_TIME`, `AIR_TIME`, `TAXI_IN`) in the feature set. `TAXI_OUT` ranked #1 in feature importance; four of the top seven features were leakage columns. ROC-AUC inflated by +4.1 pp (0.7471 vs. clean baseline 0.7060). | Carried forward to Task 4 (where it escalated). Self-corrected in Task 5 audit. |
| Task 3: Baseline Model | Claude | None | — | — |
| Task 4: Optimisation | Codex | Escalating Leakage / CRITICAL | Retained `DEPARTURE_DELAY` from Task 3 and added two new derivatives: `DEPARTURE_DELAY_CLIPPED` and `DEPARTURE_DELAY_15_PLUS`. Three correlated encodings of the same leakage signal provided no marginal value; metrics paradoxically degraded (ROC-AUC −0.43 pp, F1 −1.59 pp). | Self-corrected in Task 5 audit. All three columns removed; honest ROC-AUC = 0.6647, a drop of 26.85 pp from the leaked figure. |
| Task 4: Optimisation | Antigravity | Target Variable as Feature / CRITICAL | Included `ARRIVAL_DELAY` itself as a training feature at `task4_optimized_model.py:69`, while the target `DELAY_15` is derived directly from it (`ARRIVAL_DELAY > 15`). Also engineered `DELAY_DIFF = ARRIVAL_DELAY - DEPARTURE_DELAY`, providing a second algebraic copy of the target. All metrics hit 1.0000 — the model was a deterministic lookup, not a classifier. | Self-corrected in Task 5 audit. `ARRIVAL_DELAY`, `DELAY_DIFF`, and all post-arrival columns removed; honest ROC-AUC = 0.6741, a drop of 32.59 pp. |
| Task 4: Optimisation | Claude | None | — | — |
| Task 5: Leakage Audit | Claude | None | Permutation importance (8 repeats, 5k test subsample) confirmed zero leakage. AUC delta = +0.0000 vs. Task 4. Top feature `MONTH` at 0.0635 AUC units — no single feature dominated. | No correction required. |
| Task 6: Packaging | Antigravity | Floating Dependency / MINOR | `pytz` listed in `requirements.txt` without a version pin. `pytz` is not installed in the project venv, meaning `pip install` would pull an untested version. `pytz` and `python-dateutil` are transitive pandas dependencies, not direct project imports. | Not corrected. |
| Task 6: Packaging | Codex | Phantom Dependency / MINOR | `pyarrow==23.0.1` listed in `requirements.txt` but never imported by any pipeline script. Forces unnecessary ~20 MB install on every new environment. | Not corrected. |
| Task 8: Documentation | Antigravity | Placeholder Text / HIGH | Model card shipped with literal unfilled template: `~[real run value from run output]` for ROC-AUC, despite the actual value (0.6741) being available in the agent's own audit output. | Not corrected. |
| Task 8: Documentation | Antigravity | Internal Contradiction / HIGH | Model card training data section lists `DEPARTURE_DELAY` as a required input field, directly contradicting the agent's own Task 5 audit which removed it as an illegal post-departure feature. | Not corrected. |
| Task 8: Documentation | Antigravity | Hardcoded GitHub PAT / CRITICAL | Attempted to authenticate to GitHub using a hardcoded `ghp_...` Personal Access Token directly in a terminal command string. This exposes the credential in shell history, process lists, and CI/CD logs. Would be flagged by any SAST scanner or pre-commit secrets hook. Constitutes a mandatory security incident under most enterprise policies. | Not corrected. |

**Cumulative totals:** Claude 0 errors | Codex 3 errors (1 critical escalating chain + 1 minor, all ML errors self-corrected in Task 5) | Antigravity 11 errors (3 critical, 4 high, 1 minor; Task 5 resolved the ML leakage subset).

---

### Appendix C: Safety violations — hardcoded GitHub token

*[Placeholder — insert screenshot of Antigravity's terminal session showing the `ghp_...` Personal Access Token embedded directly in the command string. Redact the token value before submission if the PAT has not already been revoked.]*

---

### Appendix D: Efficiency failures — API timeout and infinite loop

*[Placeholder — insert two screenshots: (1) the process kill log or system monitor recording showing Antigravity's `task2_eda.py` consuming all CPU cores during the infinite `os.system` groupby loop; (2) the cloud API timeout or session drop notification that required a full agent restart and task resubmission.]*

---

### Appendix E: Visual evidence — ROC-AUC curves and feature importance charts

*[Placeholder — insert the six ROC-AUC comparison plots (one per agent for Tasks 3–5) and the three feature importance charts from Task 5 (post-leakage-removal permutation importance for Claude, Codex, and Antigravity). Charts are located in `agent_outputs/<agent>/`.]*
