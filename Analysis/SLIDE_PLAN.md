# Slide Plan: Data Science Track, 7-Minute Video

**Constraints (from the rules and handbook):** at most 7:00 from first word to last (penalties after 7:00, judges stop at 8:00). Slides must be **static**: no animations, no videos, no live demos. Every claim needs evidence *on the slide*, because judges can't ask questions in Round 1. A one-slide workflow diagram is recommended. Declare the **Data Science** track.

**Judged on (official D1–D8):** D1 Problem formulation, D2 Data prep (Validity 25%) · D3 Modeling rigor (15%) · D4 Validation (10%) · D5 Predictive performance (15%) · D6 Insight, D7 Decision impact (25%) · D8 Communication (10%).

**Where each criterion is earned:** D1 → slide 2 · D2 → slides 3–4 · D3 → slides 5, 7 · D4 → slides 6–8 · D5 → slides 7–8 · D6 → slides 4–6 · D7 → slides 7, 10–12 · D8 → slide 3 plus METHODS.md as a backup.

**Storyline:** *The factory catches every defect, but the repairs don't hold. Three build traits explain where the failures come from, models built only on build data predict them, and a $700K portfolio aimed at those traits saves $3.4M a year. It stays the best choice even under pessimistic assumptions.*

| # | Slide title (state the finding) | Visual (file) | Evidence to show on the slide | Time |
|---|---|---|---|---|
| 1 | Data Addicts: Data Science Track | none | Team names, track, one-line answer: "$700K → $3.4M/yr saved, best in 98% of 5,000 simulations" | 0:15 |
| 2 | We turned the business question into five testable questions | table from `EVALUATION.md` ("Problem formulation") | Q1–Q5 with target, unit, method, validation | 0:35 |
| 3 | Our workflow | diagram (redraw the ASCII one at the top of `Analysis/README.md`) | 5 workbooks → audit → inference → classifier and cost model → segmentation → portfolio | 0:25 |
| 4 | Every defect was caught in the factory, and came back in the field | `Analysis/outputs/A01_defect_journey.png` (+ `A01_leakage_screen.png` small) | 3,000/3,000 factory labels match the field failure; so we model **only on build data** (feature tiers) | 0:45 |
| 5 | Three build traits drive the failures, and the stats confirm them | `Analysis/outputs/A02_root_cause_evidence.png` | Supplier B 56% leak vs 14%; Facility 2 54% vs 0/1,570; Rev A 32% controls vs 0; backward fans only after fan-correction rework; LegacyX ×1.52; all q < 1e-75 after BH correction. Nulls: disposition, region, test station | 0:50 |
| 6 | Unsupervised: half the builds carry 83% of the warranty cost | `Analysis/outputs/A05_segments.png` | k=4 by silhouette (0.70 vs 0.12–0.18 for build-only), bootstrap ARI 0.91; segment table | 0:45 |
| 7 | Supervised: build data predicts leaks before the unit ships | `Analysis/outputs/A03_classifier_comparison.png` + `A03_confusion_and_capture.png` | Hold-out 600 units, never used for tuning; log loss 0.857 → 0.485 (90% of achievable); leak AUC 0.84; same result on a time-ordered split (0.492); "F2 or Supplier B" flags 48% of builds, catches 100% of leak cost | 0:50 |
| 8 | Cost model prices each problem within ±10% | `Analysis/outputs/A04_cost_model.png` | Two-stage model R² 0.29 vs 0.00 baseline; slice totals +3%, +10%, −2%, −5% vs family baseline −41% | 0:30 |
| 9 | Which investments? Every combination within budget | `PortfolioSim/outputs/fig3_portfolio_frontier.png` (or `fig5_program_reach.png`) | 95 portfolios scored; targeted scopes; benefits stack, not add; catalog costs | 0:35 |
| 10 | Recommendation: Braze + Poka-Yoke + Leak Detection + Supplier QA, $700K | `Analysis/outputs/A06_cumulative.png` | About 595 failures avoided/yr (20% of failures, 52% of leaks) = $3.38M/yr (44% of $7.65M); 3-yr net $4.85M mean with claim lag; 6.9× ROI; payback month 12 | 0:30 |
| 11 | The pick survives every stress test | `PortfolioSim/outputs/fig6_robustness.png` + `Analysis/outputs/A06_tornado.png` | Best in 98% of 5,000 runs; P(loss) 0%; still pays back if benefits are only 10% of catalog; Leak Detection's 40% is the key assumption | 0:40 |
| 12 | Implementation and how we'll verify each cause first | text slide | Order by go-live: Braze (mo 2) → Poka-Yoke + Supplier QA (mo 4) → Leak Detection (mo 8). Verify first: helium test on B coils, F2 braze audit, Rev A bench test, check 20 fan-corrected units. Free quick wins: F2 braze audit ($2.13M/yr exposure), fix the fan-correction instruction. Success metric: leak share in S1 builds, warranty $/unit by segment | 0:35 |
| 13 | Limits and next steps | text slide | Data has only failed units, so this is failure *mix*, not rate. Ask Airedale for production volumes by facility, supplier, revision and shift. One year of data. Associations, not proven causes (hence slide 12) | 0:20 |
| | **Total** | | | **~6:55** |

**Backup slides (Round 2 Q&A only):** `A03_calibration.png`, `A03_tree_rules.txt` (as a diagram), `A04_aggregate_knn_recheck.png` (if asked about the earlier kNN), `A06_lag_reranking.png`, `A02_root_cause_tests.csv` (full table), `PortfolioSim/outputs/fig4_top20_portfolios.png`, `fig2_program_summary.png`.

**Speaking tips:** about 150 words per minute means roughly 1,000 words of script in total. Say what each chart shows in one sentence before giving the number. Say "failure mix" and "associated with", not "rate" or "causes". Keep predictive and causal claims distinct; the handbook asks for this.

---

## Paste-ready prompt for claude.ai

> I'm building a 13-slide deck for a 7-minute recorded presentation (Virginia Tech × Airedale Applied Intelligence Challenge, **Data Science track**). Rules: static slides only (no animations), every claim must have its evidence on the slide, and include a one-slide workflow diagram. Audience: judges scoring validity, rigor, validation, predictive power vs baseline, insight/decision impact, and communication. Style: clean, white background, navy (#1d3f8f) and light blue (#9ec5f4) accents with orange (#eb6834) for leaks, so it matches the attached charts. One finding per slide, stated as the title. I'm attaching the slide plan (SLIDE_PLAN.md, which maps each slide to judging criteria D1–D8), the full analysis write-up (Analysis/README.md), the methods spec (METHODS.md), the evaluation (EVALUATION.md), the teammates' portfolio notes (PortfolioSim/NOTES.md) and the chart PNGs named in the plan. Build the deck following the plan's order, titles, visuals and evidence, and write speaker notes of about 75 words per slide (about 1,000 words total) so it fits in 7 minutes.

**Files to upload to claude.ai:**
- `Analysis/SLIDE_PLAN.md`, `Analysis/README.md`, `Analysis/METHODS.md`, `Analysis/EVALUATION.md`, `PortfolioSim/NOTES.md`
- `Analysis/outputs/`: `A01_defect_journey.png`, `A01_leakage_screen.png`, `A02_root_cause_evidence.png`, `A03_classifier_comparison.png`, `A03_confusion_and_capture.png`, `A03_calibration.png`, `A04_cost_model.png`, `A04_aggregate_knn_recheck.png`, `A05_segments.png`, `A06_cumulative.png`, `A06_tornado.png`, `A06_lag_reranking.png`
- `PortfolioSim/outputs/`: `fig3_portfolio_frontier.png`, `fig5_program_reach.png`, `fig6_robustness.png`, `fig4_top20_portfolios.png`
- The three competition PDFs (handbook, investment catalog, judging criteria D1–D8)
