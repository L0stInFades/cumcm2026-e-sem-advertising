# MDR-0009 验证与确认（V&V）协议

日期：2026-09-12 · 状态：已采纳 · 关联问题：全部

| 对象 | 独立校验 | 证据文件 |
|---|---|---|
| 输入数据 | 契约校验（列、类型、范围、漏斗一致性） | `validate/validation_report.json` |
| 假日效应 | HC3 稳健回归 + Mann–Whitney 非参数检验；控制消费额的对照 | `eda/calendar_regression.*`, `calendar_regression_summary.json` |
| 注册归因 | 滚动起点选择 adstock 衰减；留出期 $R^2$/MAE/MAPE | `eda/attribution.json` |
| 分类 | 校验器按原始列重推标签、无效词定义、计数；三种阈值方法一致率；熵权 vs 等权 | `classify/classification_check.json`, `classification_summary.json` |
| 响应模型 | 历史比例分配的预测点击 vs 实际点击（MAPE）；弹性 SE 与自助区间 | `allocate/calibration_check.json`, `bootstrap.*` |
| 分配最优性 | 可行性、KKT 驻点、随机配对转移、cvxpy/Clarabel 目标值对照；单元测试中的网格穷举 | `allocate/allocation_audit.json`, `forecast/allocation_audit.json`, `tests/unit/test_e_allocate.py` |
| 预测 | 滚动回测（MAE、MAPE、pinball、80% 覆盖率）与两个基线 | `forecast/backtest_summary.*` |
| 灵敏度 | 上限倍数、弹性 ±10%、预算 ±20%、问题词处理、最小投放额 | `allocate/sensitivity.*` |
| 结果文件 | 模板契约（表名、表头、行数、数值区、小数位） | `qa/qa_report.json` |

所有校验器与求解器代码分离（`allocate.verify_allocation`、`classify.verify_classification`），失败即阶段失败。
