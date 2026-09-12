# 变更记录

格式遵循 Keep a Changelog；版本号遵循语义化版本。

## [Unreleased]

### Fixed (论文撰写阶段，run `20260912-041501-ae773aa` 内重跑全部科学阶段与门禁)
- `eda`：归因注册改按下游实际采用的注册率（单元估计或合并回退）计算，原始 NNLS 估计保留为 `attributed_regs_raw`；此前使用合并回退的单元在表中显示归因注册 0、注册成本发散（MDR-0010）。
- `forecast.simulate_group`：日效率冲击按 $\exp(\mu+\sigma z)$ 抽样，使情景均值等于分配所用的对数正态期望 $\exp(\mu+\sigma^2/2)$；此前多减了 $\sigma^2/2$，蒙特卡罗期望注册（1101）系统性低于确定性目标（1187）。修正后二者一致（MDR-0008 补充）。
- `tables`：新增 `raw` 单元格规格，`tab_backtest` 的数学表头与 `tab_verification` 的 `\%` 不再被双重转义；`tab_attribution_response` 同时给出单元估计 $\hat\gamma_u$ 与采用值；`tab_verification` 报告凸松弛总体最优性界，并把预算不足 20 倍最小投放额的微小问题单列。
- `figures`：`fig_allocation_mix`（两窗口均为"重点词 ≈ 99%"，无信息量）替换为 `fig_allocation_shift`（最优窗口日均投放 vs 2025 年日均消费的双对数散点 + 各类候选/入选计数）。
- `stages`：新增论文所需的登记数字（单元/方案数、品牌型与最低 CTR 单元、科学计数法 p 值、控制消费后的周几 p 值、CTR/上方位假日效应、Mann–Whitney 中位数、最优假日减投比、上/非上方位 CTR、零消费词占比、重复词行数、凸松弛微小问题数与信息量充分的最大相对差、达上限/最小投放额词–日数、8 月窗口单元提升极值、全年日均预算口径的注册增幅等）。

### Added
- 科学阶段：`eda`（问题一：KPI、集中度、日历/假日回归、注册归因、弹性）、`classify`（问题二：熵权效益指数 + 自然断点分类与独立复核）、`allocate`（问题三：两阶段拉格朗日/注水求解的可分离凹规划、独立最优性审计、反事实对照、自助区间、灵敏度、窗口预算备选）、`forecast`（问题四：日历回归预测与滚动回测、两层随机分配、蒙特卡罗期望范围）、`results`、`figures`、`tables`。
- 建模决策记录 MDR-0001…0009；`docs/DATA_NOTES.md`；`docs/RESULTS.md`。
- 单元/性质测试：日历与解析、分类规则与校验器、分配求解器（网格穷举、SLSQP 子集枚举、性质测试）、预测与模拟。

### Verified
- run `20260912-041501-ae773aa`（Modal，medium 档）：ingest、validate、eda、classify、allocate、forecast、results、figures、tables、lint、test、paper、qa 全部 completed；44 项测试通过；qa 中 result2/3/4.xlsx 契约检查通过（`--param require_results=true`），论文用骨架编译。
- Q3：107 个单元-日分配问题全部通过独立审计，凸松弛总体最优性界 0.080%；Q4：75 个分配问题全部通过审计。
- 身份信息观察名单复核：qa 报告 5 处命中全部位于附录代码清单第 19 页对 `configs/default.toml` 中 `identity_watchlist` 定义本身的转录（"大学、学院、赛区、队号、指导教师" 字面量），不含任何真实身份信息；论文骨架、结果文件与支撑材料均不含姓名、学校、赛区、队号、邮箱。

### Changed
- `pyproject.toml`：`[tool.ruff.lint.pycodestyle] max-line-length = 150`——E501 按显示宽度计数（中文字符计 2），论文用中文字符串需要余量；格式化宽度仍为 120。
- `configs/default.toml`：`paper.sources = ["figures", "tables"]`，`package.stages` 加入四个科学阶段；新增 `[classify]`、`[allocate]`、`[forecast]` 参数节。

## [0.1.0] - 2026-09-12

### Added
- Forge 云端运行时（阶段/运行/清单模型、数据与结果契约、事件日志、确定性打包）。
- 通用流水线：ingest、validate、fmt、lint、test、paper、qa、package、release；全部在 Modal 执行。
- CUMCM 格式论文骨架（XeLaTeX + BibTeX gbt7714），自动生成数字宏、支撑材料清单、复现记录与完整程序附录。
- 工程规范、建模与写作标准、运行手册、ADR/MDR、AI 使用记录、Unlicense。

### Verified
- 三题仓库全链路冒烟通过：25 项单元/性质/集成测试，13 项论文与结果 QA 检查，发布散列核对一致。
