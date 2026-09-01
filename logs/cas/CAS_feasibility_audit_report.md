# CAS第二数据集可行性审计报告

审计日期：2026-08-28（Asia/Shanghai）  
审计版本：CAS_FEASIBILITY_V1  
审计对象：NZ Transport Agency Waka Kotahi Crash Analysis System (CAS) open data  
研究用途：作为STATS19时间泛化研究的第二数据源独立复核，不与STATS19合并训练

## 1. 审计结论

**结论：有条件通过。**

CAS可以作为第二数据源，用于检验泄漏感知、时间外推评估流程在另一司法辖区数据上的可执行性，以及核心结果方向是否一致。它不能被表述为与STATS19完全同口径的外部验证集，也不能用于证明该框架已经具有跨国家普适性。

通过的前提是同时执行以下限制：

1. 仅纳入2022—2025年伤害事故（Minor、Serious、Fatal），排除Non-Injury；
2. 数据集分别训练、分别评估，不合并STATS19与CAS样本；
3. 预注册时间划分为2022—2023年训练、2024年验证、2025年测试；
4. CAS保留原生有序标签Minor < Serious < Fatal；跨数据集汇总只比较最低、中间和最高严重度等级的结果方向，不把Minor重命名为Slight，也不宣称二者定义完全相同；
5. 仅使用15个经语义审计的碰撞时可观测候选字段；
6. 2025测试集在预处理、调参和模型选择期间保持封存；
7. 只比较随机划分与时间外推差距的方向、模型增益收缩、有序误判模式和解释排序稳定性，不直接比较两国绝对Macro-F1；
8. 全文将CAS定位为“cross-jurisdiction independent replication”，不用“cross-national generalizability proven”。

## 2. 官方来源与版本

- 官方数据项：`8d684f1841fa4dbea6afaefc8a1ba0fc`
- 维护机构：NZ Transport Agency Waka Kotahi
- 统计单位：一宗警方报告的道路碰撞
- 数据许可：CC BY 4.0 International
- 数据服务：按月更新的实时ArcGIS Feature Service，不是静态年度发布包
- 冻结快照：2022—2025年伤害事故43,121宗、70个在线字段
- 快照SHA-256：`cc554366351cf4d5ccc7207f583c8ff1437036a615f1bac4dcdcccac76d1745c`
- 无损JSONL同时归档，用于区分JSON null与字面字符串`None`/`Null`

官方数据项和实时图层最近更新时间为2026-08-24；字段说明页最近更新时间为2023-07-11。说明页滞后于实时图层，因此代码必须以冻结快照中的实际字段和值为准，并把字段说明作为语义依据而非唯一模式约束。

## 3. 目标口径

CAS的`crashSeverity`由事故中最严重伤情确定。新西兰交通部官方术语表定义：

- Fatal：事故后30日内因伤死亡；
- Serious：骨折、脑震荡、内伤、挤压伤、严重割伤或裂伤、需医疗处理的严重休克，以及需要送医并留院的其他伤情；
- Minor：扭伤、瘀伤等轻伤。

CAS字段说明另写明严重度为“at time of entry”确定，开放数据页同时说明数据为实时数据、记录可能在发布后改变。因此，CAS标签与STATS19的严重度标签具有相似的有序结构，但不能视为完全相同的最终标签生成机制。

伤害事故样本分布：

| 年份 | Minor | Serious | Fatal | 合计 |
|---:|---:|---:|---:|---:|
| 2022 | 8,536 | 2,102 | 334 | 10,972 |
| 2023 | 8,571 | 2,086 | 305 | 10,962 |
| 2024 | 8,306 | 2,090 | 249 | 10,645 |
| 2025 | 8,222 | 2,061 | 259 | 10,542 |

总计Minor 33,635（78.00%）、Serious 8,339（19.34%）、Fatal 1,147（2.66%）。2025测试集Fatal为259宗，足以报告fatal recall及其Bootstrap区间，但不保证区间狭窄。

## 4. 年份准入

### 纳入

- 2022—2023：训练；
- 2024：验证；
- 2025：最终时间测试。

### 排除

- 2020—2021：官方开放数据页明确标注数据不完整；
- 2026：当前年度尚未结束，且报告处理存在时间滞后；
- 2020以前：CAS系统在2019年发生变更，`crashSHDescription`和`roadCharacter`于2019-12-17改变记录方式，且新旧CAS在2019-02-20切换；为避免额外的模式迁移，不进入第二数据集主分析。

此年份方案依据官方完整性和模式变更信息预先确定，不得在查看模型性能后更换。

## 5. 字段泄漏审计

实时图层70字段的处理结果：

- 15个候选预测字段；
- 36个事故参与者、撞击对象或其他碰撞后派生字段，排除；
- 3个伤亡计数字段，直接泄漏目标，排除；
- 6个细粒度位置字段，防止地点记忆，排除；
- 2个事件方向字段，属于碰撞事件编码，排除；
- 2个实时存在但说明页未定义字段，因语义不确定排除；
- `OBJECTID`排除；`crashYear`仅用于划分；`region`仅用于分组审计；`crashSeverity`为目标；`holiday`因缺少原始日期且不是最小匹配框架所必需而排除。

15个候选字段为：

- 数值：`advisorySpeed`、`speedLimit`、`temporarySpeedLimit`；
- 类别：`NumberOfLanes`、`crashSHDescription`、`flatHill`、`light`、`roadCharacter`、`roadLane`、`roadSurface`、`streetLight`、`trafficControl`、`urban`、`weatherA`、`weatherB`。

`NumberOfLanes`虽然以整数存储，但按类别变量处理。值0不是一律无效：299条中197条对应`Off road`。对于`speedLimit`非缺失的42,423宗事故，`urban`均可由小于80或不小于80的限速分组确定；另有698宗事故限速为空但`urban`仍有记录。主分析可保留二者以复现公开层语义，但必须做移除`urban`的消融分析，且解释重要性不得把二者当作独立机制。

### 5.1 与STATS19的特征口径

逐项对照17个STATS19特征后，仅照明、天气和限速属于概念较接近的字段；道路等级、道路类型、交通控制、城乡属性和主干道路属性只能部分对应；9个STATS19特征在CAS公开层没有可接受的对应字段。CAS另有5个候选道路属性不在STATS19冻结特征集中。完整对照见`cas_stats19_feature_crosswalk.csv`。

因此，两套数据必须采用各自的字段配置分别训练。CAS能复核的是同一套泄漏审计、冻结时间划分、有序评价和解释稳定性流程，而不是把STATS19训练出的模型直接拿到CAS测试。即使两套数据的结果方向一致，也只能称为跨司法辖区独立复核。

## 6. 数据质量与模式漂移

关键问题如下：

1. `advisorySpeed`缺失94.38%—94.82%，`temporarySpeedLimit`缺失96.07%—96.91%。这主要是结构性“不存在”，不得使用普通均值插补；保留缺失指示/类别，且做移除二者的敏感性分析。
2. `speedLimit`缺失1.45%—1.76%，`NumberOfLanes`缺失1.40%—1.90%，规模可控。
3. 候选类别在2022—2023训练中覆盖了2024验证和2025测试的全部观测类别，当前没有未见类别；代码仍须实现`__UNSEEN__`兜底。
4. 最大类别分布漂移出现在`trafficControl`：2022与2024总变差距离约0.0494；`light`、`weatherA`和`streetLight`也有较小漂移。这是时间泛化分析要报告的证据，不应清洗掉。
5. 官方说明页和实时数据值不完全一致：`crashSeverity`说明为F/S/M/N，实时值为完整英文标签；`crashSHDescription`说明为1/2，实时值为Yes/No/Unknown；`roadSurface`、`streetLight`、`weatherB`也有额外或不同标签。
6. 说明页包含5个当前图层不存在的字段，当前图层有2个未被说明页定义的业务字段。这证明必须冻结模式并逐字段审计，不能依赖网页示例代码硬编码。

## 7. CSV读取陷阱

CAS使用字面字符串`None`和`Null`作为类别。例如`streetLight="None"`有14,219宗，`weatherB="None"`有83宗，`weatherB="Null"`有41,122宗。`pandas.read_csv()`默认会把字符串`None`识别为缺失，导致静默的数据语义破坏。

强制规则：

```python
df = pd.read_csv(path, keep_default_na=False, low_memory=False)
# 仅将真正的空字符串按字段规则转换为缺失；不得把 "None" 或 "Null" 合并成 NaN。
```

无损JSONL快照为权威归档；CSV用于分析便利，但读取规则必须测试。

## 8. 与STATS19比较的边界

允许比较：

- 随机划分相对时间测试的Macro-F1/QWK差距方向；
- LightGBM相对Logistic的增益是否在时间测试中收缩；
- 严重跨级误判模式；
- SHAP全局排名在验证年与测试年之间的Spearman稳定性。

不允许比较或宣称：

- CAS与STATS19绝对Macro-F1谁更高；
- CAS Minor与STATS19 Slight完全同义；
- 两国数据已证明工具普适；
- 字段重要性代表因果作用；
- CAS结果是对英国模型的直接外部验证。

## 9. 仍需执行的准入门槛

本次是数据和方案可行性审计，不是模型结果审计。进入CAS建模前还需：

1. 用冻结JSONL生成CAS建模表，并对15个候选字段执行类型、空值和允许类别断言；
2. 完成2022—2023训练、2024验证、2025测试的固定索引和类别权重；
3. 随机参照只从2022—2024池中抽取，并使用与STATS19一致的5个种子；
4. 2025测试集仅在CAS模型及预处理被冻结后评估一次；
5. 对结构性稀疏速度字段做移除敏感性分析；
6. 在正文或补充材料中披露实时数据、字段说明滞后、标签口径和报告不足风险。

此外，CAS只有2022—2025四个连续、口径较稳定且完整的可用年份。时间跨度较短、字段漂移较弱，因此随机划分与时间测试之间未必出现明显性能差距。若未复现STATS19中的下降，只能报告“该现象在CAS短时间窗内未观察到”，不能据此否定时间外推评估的必要性，也不能在看到结果后扩展年份或改动切分。

满足以上门槛后，CAS可进入第二数据集实验；任何一项被省略，结论应降级为“不建议纳入主论文”。

## 10. 官方依据

- CAS系统说明：https://www.nzta.govt.nz/partners/data-and-tools/crash-analysis-system
- CAS开放数据项：https://opendata-nzta.opendata.arcgis.com/datasets/NZTA::crash-analysis-system-cas-data-1/about
- CAS字段说明：https://opendata-nzta.opendata.arcgis.com/pages/cas-data-field-descriptions
- 新旧CAS差异：https://www.nzta.govt.nz/partners/data-and-tools/crash-analysis-system/current-and-previous-cas-systems
- 新西兰交通部严重度术语：https://www.transport.govt.nz/statistics-and-insights/glossary-and-references
- 许可：https://creativecommons.org/licenses/by/4.0/
