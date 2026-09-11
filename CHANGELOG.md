# 变更记录

格式遵循 Keep a Changelog；版本号遵循语义化版本。

## [Unreleased]

### Added
- 科学阶段：`eda`（问题一：KPI、集中度、日历/假日回归、注册归因、弹性）、`classify`（问题二：熵权效益指数 + 自然断点分类与独立复核）、`allocate`（问题三：两阶段拉格朗日/注水求解的可分离凹规划、独立最优性审计、反事实对照、自助区间、灵敏度、窗口预算备选）、`forecast`（问题四：日历回归预测与滚动回测、两层随机分配、蒙特卡罗期望范围）、`results`、`figures`、`tables`。
- 建模决策记录 MDR-0001…0009；`docs/DATA_NOTES.md`；`docs/RESULTS.md`。
- 单元/性质测试：日历与解析、分类规则与校验器、分配求解器（网格穷举、SLSQP 子集枚举、性质测试）、预测与模拟。

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
