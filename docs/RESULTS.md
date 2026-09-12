# 结果总览（供论文撰写使用）

所有数字来自同一次云端 run（见 `.forge/last_run` 与本文件末尾的 run 记录）；论文中的数字只能引用 `\val<Key>` 宏（各阶段 `numbers.json` 汇总）或 `tables` 阶段生成的表。下文的"数值"栏为该 run 的实际值，供撰写者核对。

## 阶段与产物

| 阶段 | 内容 | 主要文件 |
|---|---|---|
| `eda` | 问题一：KPI、集中度、日历/假日回归、注册归因、弹性 | `unit_kpis.*`, `calendar_regression.*`, `calendar_regression_summary.json`, `attribution.json`, `attribution_fit.*`, `unit_response.*`, `concentration.json`, `lorenz.*`, `day_type_profile.*`, `daily_totals.*`, `unit_daily.*` |
| `classify` | 问题二：五类分类 | `classification.*`（逐词），`classification_by_label.*`, `classification_per_unit.*`, `classification_summary.json`, `classification_check.json`（独立复核） |
| `allocate` | 问题三：分配 | `allocation.*`（逐词逐日），`allocation_summary.*`（单元-日汇总），`allocation_audit.json`, `calibration_check.json`, `bootstrap.*`, `sensitivity.*`, `window_budget_alternative.*`, `keyword_parameters.*`, `position_models.json` |
| `forecast` | 问题四：预测 + 随机分配 | `backtest.*`, `backtest_summary.*`, `unit_forecasts.*`, `budgets.*`, `plan.*`（逐词逐日，含均值与 10/50/90 分位），`plan_by_unit_day.*`, `plan_by_unit.json`, `allocation_audit.json` |
| `results` | result2/3/4.xlsx | 阶段目录根 |
| `figures` | 论文图 | `figures/*.pdf` + `*.png` |
| `tables` | 论文表 | `tables/*.tex` + `*.csv` |

## 问题一：投放策略合理性与假日效应

**方法一句话**：单元级漏斗 KPI 与关键词集中度刻画"创意质量 / 关键词管理 / 出价预算"；带周几、月份、节假日与调休哑变量的对数线性回归（HC3 稳健标准误）+ Mann–Whitney 检验量化时间规律与假日效应；非负最小二乘把日注册数归因到各推广单元的点击（MDR-0002/0003）。

**关键数值（键名）**：`TotalSpend`, `TotalClicks`, `TotalImpressions`, `TotalRegs`, `OverallCtrPct`, `OverallCpc`, `TopSharePct`, `FirstSharePct`, `TopClickSharePct`, `TopSpendSharePct`, `TopCpc`, `OtherCpc`, `BiggestUnitId`, `BiggestUnitSpendSharePct`, `ActiveKeywords`, `ZeroKeywords`, `UniqueKeywordIds`, `SharedKeywordIds`, `GiniSpendActive`, `TopTenSpendSharePct`, `TopHundredSpendSharePct`, `Holiday{Reg,Spend,Clicks,Imp}EffectPct/CiLow/CiHigh/P`, `HolidayRegGivenSpendEffectPct/CiLow/CiHigh/P`, `RegSpendElasticity(Se)`, `{Monday,Saturday,Sunday,AdjustedWorkday,PreHoliday,PostHoliday}RegEffectPct/RegP`, `MondayRegGivenSpendEffectPct`, `SundayRegGivenSpendEffectPct`, `HolidayCpcEffect/P`, `MondayCpcEffect/P`, `RegCalendarRsq`, `RegWeekdayWaldP`, `HolidayRegMedianRatio`, `HolidayRegMannWhitneyP`, `AttributionLambda`, `AttributionRsq`, `AttributionHoldoutRsq`, `AttributionHoldoutMape`, `BaselineRegsPerDay`, `PooledRegRatePerHundredClicks`, `AttributedRegs`, `CostPerAttributedReg`, `UnitsWithOwnRate`, `PooledGamma`, `PooledGammaSe`, `MinUnitGamma`, `MaxUnitGamma`, `UnitsWithOwnGamma`.

**表**：`tab_unit_kpis`（各单元年度 KPI）、`tab_calendar_effects`（日历/假日效应，注册、消费、点击）、`tab_attribution_response`（各单元注册率与来源、归因注册（按实际采用的注册率，MDR-0010）、注册成本、单元估计 $\hat\gamma_u$ 及其 SE、$R^2$、采用的 $\gamma_u$ 与来源）。

**图与建议图注**：
- `fig_daily_series`：2025 年日消费额、日点击量与日新注册数（细线为日值，粗线为 7 日滑动均值；阴影为法定节假日，虚线为调休上班日）。
- `fig_calendar_effects`：周几、法定节假日、调休、假前/假后对日注册数、消费额、点击量的效应（相对周三/普通日，%），HC3 稳健 95% 置信区间。
- `fig_unit_kpis`：各推广单元年度消费占比（左）与 CPC–CTR 散点（右，气泡面积 ∝ 消费占比，标签为单元 ID 后四位）。
- `fig_lorenz`：关键词消费额与点击量的 Lorenz 曲线（按消费降序累计）。
- `fig_response_curves`：各单元日点击量–日消费额（双对数）散点与幂函数响应曲线（γ 带 * 者使用合并弹性）。
- `fig_attribution`：日注册数与归因回归拟合值（虚线右侧为留出期）。

**验证结论**：HC3 回归与 Mann–Whitney 检验结论一致；归因模型留出期 R² 见 `AttributionHoldoutRsq`；弹性估计 R² 见 `tab_attribution_response`。

**值得写进论文的发现**：法定节假日日注册数总效应 −73.7%（95% CI [−80.0, −65.5]），其中控制当日消费额后仍有 −24.6%（[−40.0, −5.3]，p = 0.015）——约三分之一来自需求侧，其余来自公司假日主动减投（消费额 −76.7%）；周一与周日的注册下降（−53.8%、−47.4%）几乎完全由减投解释（控制消费后不显著，见 `MondayRegGivenSpendEffectPct`、`SundayRegGivenSpendEffectPct`）。CPC、CTR 无显著假日效应，上方位占比在假日略降。

**局限**：注册只有日总量，归因为回归推断而非真实追踪；假日效应包含公司主动停投的成分（用控制消费额的模型分离）。

## 问题二：关键词五分类

**方法一句话**：成本 = 年度消费额（对数尺度），效益 = 熵权复合指数（点击、浏览、注意力时长、归因注册），阈值用两类 Fisher–Jenks 自然断点（主），中位数与高斯混合为稳健性对照；无效词 = 零消费且零点击（MDR-0001/0004）。

**关键数值（键名）**：`{Gold,Key,Potential,Problem,Invalid}Count`, `{...}SpendSharePct`, `{...}Cpc`, `CostThresholdYuan`, `CostThresholdLog`, `BenefitThreshold`, `Weight{Clicks,Views,Engagement,Reg}`, `AgreementJenksMedianPct`, `AgreementGmmJenksPct`, `AgreementJenksEqualWeightsPct`, `EntropyEqualSpearman`, `MedianCostThresholdYuan`, `GmmCostThresholdYuan`.

**表**：`tab_class_summary`（各类数量、消费占比、CPC、平均效益、归因注册）、`tab_class_robustness`（不同阈值方法/权重下的类别数量）、`tab_class_per_unit`（单元 × 类别）。

**图与建议图注**：
- `fig_cost_benefit`：关键词成本–效益散点（横轴年度消费额对数，纵轴综合效益指数；虚线为自然断点阈值；颜色为类别；无效词零消费未画出）。
- `fig_class_thresholds`：对数消费额与效益指数的直方图及三种阈值方法的位置。

**验证结论**：`classification_check.json` 全部通过（标签由原始列独立重推、无效词定义、计数）。

**局限**：效益指数依赖归因注册率（单元级）；阈值为相对（数据驱动）而非业务绝对标准。

**结果文件**：`result2.xlsx`（2227 行，五个类别列 1/0 标记）。

## 问题三：关键词选择与投放模型（2025-02-01–08、08-01–08）

**方法一句话**：每个单元-日求解可分离凹规划 $\max\sum_i \rho_i c_i (x_i/s_i)^{\gamma_u}$ s.t. 预算（= 当日实际消费）与上限、最小投放额 0.5 元（选择机制）；两阶段拉格朗日 + 精确注水求解，独立校验器复核 KKT、配对转移与凸松弛上界（MDR-0005/0006/0007）。

**关键数值（键名）**：`QthreeBudget{Feb,Aug}`, `QthreeOptRegs{..}`, `QthreePropRegs{..}`, `QthreeOptClicks{..}`, `QthreePropClicks{..}`, `QthreeActualClicks{..}`, `QthreeRegGainPct{..}`, `QthreeRegGainCiLow/High{..}`, `QthreeClickGainPct{..}`, `QthreeEqualRegGainPct{..}`, `QthreeUnitDays{..}`, `QthreeUnits{..}`, `QthreeSelectedKeywords{..}`, `QthreeEligibleKeywords{..}`, `QthreeAvgCpc{..}`, `QthreePropCpc{..}`, `QthreeWindowRegs{..}`, `QthreeWindowGainPct{..}`, `QthreeAuditGroups`, `QthreeMaxCvxGap`, `QthreeAggregateGapPct`, `QthreeCalibrationMape`, `QthreeCalibrationMedianApe`, `QthreeRows`, `QthreeCapMultiplier`, `QthreeMinSpend`.

**表**：`tab_q3_summary`（各窗口各单元：预算、候选/入选词、历史比例 vs 最优注册、提升与自助区间、CPC）、`tab_q3_daily`（逐日汇总）、`tab_q3_window_budget`（窗口预算跨日调配的备选结果）、`tab_q3_sensitivity`（上限倍数、弹性、预算、问题词处理、最小投放额）、`tab_verification`（校验汇总）。

**图与建议图注**：
- `fig_q3_gain`：两个窗口逐日预期注册量（最优分配 vs 按历史比例 vs 均匀分配）与弹性自助抽样得到的提升分布。
- `fig_allocation_shift`（替换原 `fig_allocation_mix`，MDR-0010）：8 月窗口每个入选关键词的最优窗口日均投放 vs 2025 年日均消费（双对数，按类别着色，附 45° 线与上限倍数线）；右图为各类候选词数与至少入选一天的词数。位于上限线之上的一簇点是历史日均消费低于 $m/\mu$ 的长尾词（上限即最小投放额），其预算与注册份额见 `QthreeFloorSpendSharePctAug`、`QthreeFloorRegsSharePctAug`。

**验证结论**：全部单元-日问题通过独立审计（可行性、KKT、配对转移/剔除/插入不可改进、凸松弛上界）；`QthreeAggregateGapPct` 为总体最优性界。响应模型校准 MAPE 见 `QthreeCalibrationMape`。

**局限**：关键词响应曲线由年度汇总 + 单元弹性外推；2 月窗口只有 2 个单元有实际消费。

**备选解释**：窗口预算跨日调配（`tab_q3_window_budget`）；预算 ±20%（`tab_q3_sensitivity`）。

**结果文件**：`result3.xlsx`（日期、方案ID、推广单元、关键词、投入金额、预期展位、预期点击量、预期浏览量、预期注册量）。

## 问题四：2026-09-11–17 的最优策略与期望范围

**方法一句话**：单元级日历回归预测效率乘子、CPC、CTR、展位占比（滚动回测对照季节朴素与 28 日均值），预算 = 2025 同期各单元消费；两层分配（跨日 + 日内）后用 500 个蒙特卡罗情景给出竞价、展现量、展位、点击、浏览、注册的期望与 10%–90% 范围（MDR-0008）。

**关键数值（键名）**：`QfourBudget`, `QfourBudgetAnnualAlt`, `QfourUnits`, `QfourSkippedUnits`, `QfourRows`, `QfourSelectedKeywords`, `Qfour{Clicks,Imp,Views,Regs}{Mean,Low,High}`, `QfourCpc{Mean,Low,High}`, `QfourPositionMean`, `QfourScenarios`, `QfourAuditGroups`, `QfourRefClicks`, `QfourClickGainVsRefPct`, `Backtest{Calendar,Naive,Mean}Mae{Eff,Cpc,Ctr,Top,Clicks}`, `Backtest{Calendar,Naive}Mape{..}`, `BacktestCalendarCoverage{..}`, `Backtest{Calendar,Naive}Pinball{..}`.

**表**：`tab_backtest`（目标 × 模型的 MAE/RMSE/MAPE/pinball/覆盖率）、`tab_q4_daily`（逐日投入、CPC 范围、展现范围、展位、点击范围、浏览、注册范围）、`tab_q4_units`（各单元预算、弹性、候选词、点击与注册范围）。

**图与建议图注**：
- `fig_backtest`：滚动回测（2025-10-09 起每周一个起点，7 天步长）各目标的 MAE 与 80% 区间覆盖率，日历回归 vs 季节朴素 vs 28 日均值。
- `fig_q4_plan`：2026-09-11 至 17 逐日预期点击量、注册量与 CPC（点为期望，带为 10%–90% 分位）。
- `fig_q4_units`：各单元 7 天预期注册量及 10%–90% 范围（对数轴）。

**验证结论**：跨日与日内分配问题全部通过独立审计；回测指标见 `tab_backtest`。

**局限**：预测点在样本末 8.5 个月之后，只能依赖日历结构（周几、9 月效应），区间反映 2025 年的残差与月份间波动；2026 年的市场趋势无法从数据识别。

**备选解释**：预算按全年日均折算（`QfourBudgetAnnualAlt` = 27,347 元 vs 同期 23,488 元）。两种预算读法下的确定性预期注册量分别为 `QfourRegsExpectedSamePeriod`（1187）与 `QfourRegsExpectedAnnualAlt`（2327）：同期读法把 81% 的预算留给 9657930100（2025 年 9 月中旬其他单元几乎停投），而全年日均读法会把资金重新分配到效率更高但 9 月停投的单元（9811363528、9630806627 等），预期注册量约翻倍——论文应把这一点作为对公司 9 月投放节奏的建议。

**结果文件**：`result4.xlsx`（同 result3 结构，填入期望值）。

## 本次 run 的全部登记数字

run：`20260912-041501-ae773aa`（全部 11 个阶段 completed；paper/qa 门禁见 CHANGELOG）。以下为各阶段 `numbers.json` 的内容，论文中以 `\val<键>` 引用；此处仅供撰写者核对。

### `eda` 阶段登记的数字（82 个）

| 键 | 值 | 键 | 值 |
|---|---|---|---|
| `ActiveKeywords` | 1337 | `AdjustedWorkdayRegEffectPct` | -2.42 |
| `AdjustedWorkdayRegP` | 0.9419 | `AttributedRegs` | 58,929 |
| `AttributionHoldoutMape` | 30.19 | `AttributionHoldoutRsq` | 0.7375 |
| `AttributionLambda` | 0.0 | `AttributionRsq` | 0.7951 |
| `BaselineRegsPerDay` | 151.39 | `BiggestUnitId` | 9657930100 |
| `BiggestUnitSpendSharePct` | 56.67 | `CostPerAttributedReg` | 24.20 |
| `FirstSharePct` | 10.39 | `GiniSpendActive` | 0.9676 |
| `HolidayClicksCiHigh` | -68.17 | `HolidayClicksCiLow` | -81.61 |
| `HolidayClicksEffectPct` | -75.81 | `HolidayClicksP` | 0.0000 |
| `HolidayCpcEffect` | -0.0240 | `HolidayCpcP` | 0.8043 |
| `HolidayImpCiHigh` | -63.68 | `HolidayImpCiLow` | -76.49 |
| `HolidayImpEffectPct` | -70.78 | `HolidayImpP` | 0.0000 |
| `HolidayRegCiHigh` | -65.47 | `HolidayRegCiLow` | -79.99 |
| `HolidayRegEffectPct` | -73.72 | `HolidayRegGivenSpendCiHigh` | -5.32 |
| `HolidayRegGivenSpendCiLow` | -40.01 | `HolidayRegGivenSpendEffectPct` | -24.64 |
| `HolidayRegGivenSpendP` | 0.0151 | `HolidayRegMannWhitneyP` | 4.56e-14 |
| `HolidayRegMedianRatio` | 0.2140 | `HolidayRegP` | 0.0000 |
| `HolidaySpendCiHigh` | -67.61 | `HolidaySpendCiLow` | -83.21 |
| `HolidaySpendEffectPct` | -76.68 | `HolidaySpendP` | 0.0000 |
| `KeywordRows` | 2227 | `MaxUnitGamma` | 0.9301 |
| `MinUnitGamma` | 0.6502 | `MondayCpcEffect` | -0.2360 |
| `MondayCpcP` | 0.0023 | `MondayRegEffectPct` | -53.81 |
| `MondayRegGivenSpendEffectPct` | -10.85 | `MondayRegP` | 0.0000 |
| `OtherCpc` | 1.5377 | `OverallCpc` | 1.7081 |
| `OverallCtrPct` | 3.70 | `PooledGamma` | 0.8355 |
| `PooledGammaSe` | 0.0113 | `PooledRegRatePerHundredClicks` | 7.9475 |
| `PostHolidayRegEffectPct` | -17.29 | `PostHolidayRegP` | 0.4100 |
| `PreHolidayRegEffectPct` | -54.34 | `PreHolidayRegP` | 0.1482 |
| `RegCalendarRsq` | 0.4802 | `RegSpendElasticity` | 0.7235 |
| `RegSpendElasticitySe` | 0.0631 | `RegWeekdayWaldP` | 3.53e-12 |
| `SaturdayRegEffectPct` | -12.64 | `SaturdayRegP` | 0.1561 |
| `SharedKeywordIds` | 290 | `SundayRegEffectPct` | -47.36 |
| `SundayRegGivenSpendEffectPct` | -9.86 | `SundayRegP` | 0.0000 |
| `TopClickSharePct` | 67.06 | `TopCpc` | 1.7918 |
| `TopHundredSpendSharePct` | 96.40 | `TopSharePct` | 27.11 |
| `TopSpendSharePct` | 70.35 | `TopTenSpendSharePct` | 58.41 |
| `TotalClicks` | 834,815 | `TotalImpressions` | 22,580,381 |
| `TotalRegs` | 85,313 | `TotalSpend` | 1,425,949.79 |
| `TotalSpendWan` | 142.59 | `UniqueKeywordIds` | 1884 |
| `UnitDays` | 2627 | `UnitsWithOwnGamma` | 10 |
| `UnitsWithOwnRate` | 5 | `ZeroKeywords` | 890 |

### `classify` 阶段登记的数字（28 个）

| 键 | 值 | 键 | 值 |
|---|---|---|---|
| `AgreementGmmJenksPct` | 93.53 | `AgreementJenksEqualWeightsPct` | 99.19 |
| `AgreementJenksMedianPct` | 78.36 | `BenefitThreshold` | 0.2758 |
| `CostThresholdLog` | 1.6190 | `CostThresholdYuan` | 41.59 |
| `EntropyEqualSpearman` | 0.9995 | `GmmCostThresholdYuan` | 107.51 |
| `GoldCount` | 40 | `GoldCpc` | 0.7192 |
| `GoldSpendSharePct` | 0.08 | `InvalidCount` | 890 |
| `InvalidCpc` | 0.0000 | `InvalidSpendSharePct` | 0.00 |
| `KeyCount` | 271 | `KeyCpc` | 1.7143 |
| `KeySpendSharePct` | 99.11 | `MedianCostThresholdYuan` | 8.06 |
| `PotentialCount` | 977 | `PotentialCpc` | 1.0599 |
| `PotentialSpendSharePct` | 0.53 | `ProblemCount` | 49 |
| `ProblemCpc` | 2.2313 | `ProblemSpendSharePct` | 0.28 |
| `WeightClicks` | 0.1966 | `WeightEngagement` | 0.2849 |
| `WeightReg` | 0.3092 | `WeightViews` | 0.2093 |

### `allocate` 阶段登记的数字（46 个）

| 键 | 值 | 键 | 值 |
|---|---|---|---|
| `QthreeActualClicksAug` | 35,812 | `QthreeActualClicksFeb` | 5,655 |
| `QthreeAggregateGapPct` | 0.080 | `QthreeAuditGroups` | 107 |
| `QthreeAvgCpcAug` | 1.3052 | `QthreeAvgCpcFeb` | 1.9273 |
| `QthreeBudgetAug` | 37,483.78 | `QthreeBudgetFeb` | 13,681.15 |
| `QthreeCalibrationMape` | 32.92 | `QthreeCalibrationMedianApe` | 22.00 |
| `QthreeCapMultiplier` | 4.0 | `QthreeClickGainPctAug` | 15.87 |
| `QthreeClickGainPctFeb` | 12.05 | `QthreeEligibleKeywordsAug` | 1337 |
| `QthreeEligibleKeywordsFeb` | 286 | `QthreeEqualRegGainPctAug` | 77.01 |
| `QthreeEqualRegGainPctFeb` | 88.54 | `QthreeMaxCvxGap` | 2.01e-01 |
| `QthreeMinSpend` | 0.50 | `QthreeOptClicksAug` | 28,718.7 |
| `QthreeOptClicksFeb` | 7,098.4 | `QthreeOptRegsAug` | 2237.43 |
| `QthreeOptRegsFeb` | 516.42 | `QthreePropClicksAug` | 24,784.5 |
| `QthreePropClicksFeb` | 6,335.2 | `QthreePropCpcAug` | 1.5124 |
| `QthreePropCpcFeb` | 2.1595 | `QthreePropRegsAug` | 1748.04 |
| `QthreePropRegsFeb` | 459.65 | `QthreeRegGainCiHighAug` | 28.57 |
| `QthreeRegGainCiHighFeb` | 13.39 | `QthreeRegGainCiLowAug` | 27.53 |
| `QthreeRegGainCiLowFeb` | 11.67 | `QthreeRegGainPctAug` | 28.00 |
| `QthreeRegGainPctFeb` | 12.35 | `QthreeRows` | 2303 |
| `QthreeSelectedKeywordsAug` | 535 | `QthreeSelectedKeywordsFeb` | 73 |
| `QthreeUnitDaysAug` | 91 | `QthreeUnitDaysFeb` | 16 |
| `QthreeUnitsAug` | 12 | `QthreeUnitsFeb` | 2 |
| `QthreeWindowGainPctAug` | 11.25 | `QthreeWindowGainPctFeb` | 0.82 |
| `QthreeWindowRegsAug` | 2489.11 | `QthreeWindowRegsFeb` | 520.64 |

### `forecast` 阶段登记的数字（68 个）

| 键 | 值 | 键 | 值 |
|---|---|---|---|
| `BacktestCalendarCoverageClicks` | 87.26 | `BacktestCalendarCoverageCpc` | 81.83 |
| `BacktestCalendarCoverageCtr` | 78.93 | `BacktestCalendarCoverageEff` | 81.30 |
| `BacktestCalendarCoverageTop` | 78.34 | `BacktestCalendarMaeClicks` | 0.4805 |
| `BacktestCalendarMaeCpc` | 0.2932 | `BacktestCalendarMaeCtr` | 0.4368 |
| `BacktestCalendarMaeEff` | 0.2361 | `BacktestCalendarMaeTop` | 0.6403 |
| `BacktestCalendarPinballClicks` | 0.2528 | `BacktestCalendarPinballCpc` | 0.1987 |
| `BacktestCalendarPinballCtr` | 0.2720 | `BacktestCalendarPinballEff` | 0.1572 |
| `BacktestCalendarPinballTop` | 0.3939 | `BacktestMeanCoverageClicks` | 73.98 |
| `BacktestMeanCoverageCpc` | 63.63 | `BacktestMeanCoverageCtr` | 80.31 |
| `BacktestMeanCoverageEff` | 68.85 | `BacktestMeanCoverageTop` | 70.16 |
| `BacktestMeanMaeClicks` | 0.5772 | `BacktestMeanMaeCpc` | 0.2426 |
| `BacktestMeanMaeCtr` | 0.2710 | `BacktestMeanMaeEff` | 0.1874 |
| `BacktestMeanMaeTop` | 0.4521 | `BacktestNaiveCoverageClicks` | 90.20 |
| `BacktestNaiveCoverageCpc` | 88.74 | `BacktestNaiveCoverageCtr` | 91.24 |
| `BacktestNaiveCoverageEff` | 90.06 | `BacktestNaiveCoverageTop` | 90.22 |
| `BacktestNaiveMaeClicks` | 0.4796 | `BacktestNaiveMaeCpc` | 0.2525 |
| `BacktestNaiveMaeCtr` | 0.3048 | `BacktestNaiveMaeEff` | 0.2142 |
| `BacktestNaiveMaeTop` | 0.4834 | `BacktestNaivePinballClicks` | 0.3038 |
| `BacktestNaivePinballCpc` | 0.1913 | `BacktestNaivePinballCtr` | 0.1794 |
| `BacktestNaivePinballEff` | 0.1399 | `BacktestNaivePinballTop` | 0.2849 |
| `QfourAuditGroups` | 75 | `QfourBudget` | 23,488.02 |
| `QfourBudgetAnnualAlt` | 27,346.98 | `QfourClickGainVsRefPct` | 23.64 |
| `QfourClicksHigh` | 19,870.9 | `QfourClicksLow` | 12,367.2 |
| `QfourClicksMean` | 15,878.9 | `QfourCpcHigh` | 1.8992 |
| `QfourCpcLow` | 1.1820 | `QfourCpcMean` | 1.4792 |
| `QfourImpHigh` | 1,145,018.7 | `QfourImpLow` | 341,532.5 |
| `QfourImpMean` | 697,420.9 | `QfourPositionMean` | 2.6455 |
| `QfourRefClicks` | 12,843 | `QfourRegsExpectedAnnualAlt` | 2327.0 |
| `QfourRegsExpectedSamePeriod` | 1187.4 | `QfourRegsHigh` | 1,364.1 |
| `QfourRegsLow` | 854.9 | `QfourRegsMean` | 1,101.5 |
| `QfourRows` | 845 | `QfourScenarios` | 500 |
| `QfourSelectedKeywords` | 146 | `QfourSkippedUnits` | 2 |
| `QfourUnits` | 10 | `QfourViewsHigh` | 106,807.0 |
| `QfourViewsLow` | 60,552.1 | `QfourViewsMean` | 82,114.8 |

