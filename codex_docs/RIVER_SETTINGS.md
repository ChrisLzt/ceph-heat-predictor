# dev的River八项设置（2026-09-25）

实现用户指定的八项，并按后续要求改为普通Accuracy投票；不是全量River替换，更改尚未部署。

| 设置 | 原dev | 当前dev工作区 |
|---|---|---|
| 每叶候选特征数 | 全部7维 | round(sqrt(7))=3，叶内固定、无放回 |
| Poisson λ | 4 | 6 |
| StandardScaler | 启用 | 线上工厂与标准direct_arf回放均移除 |
| 每特征切点 | 5 | 10个min/max间等距内部点；全叶无有效原候选时补类别均值中点 |
| 分裂尝试间隔 | 100加权样本 | 50加权样本 |
| 子叶等待起点 | 0 | 继承的类别总权重 |
| 预剪枝 | 关闭，显式null收益0 | 已按用户要求删除空候选机制，不对齐River |
| Hoeffding delta | 0.001 | 0.01 |

仍为25棵树、C4七维工程特征、Accuracy投票、MC输出、ADWIN关闭，
标签窗口10秒、短窗口2秒、预热3000、快照2000样本或2秒。取消的是在线StandardScaler，
不是feature中的log2编码或C4慢历史曝光校正。10个切点不是10维特征，也不是分位数方案。

2026-09-25后续调整：删除空候选预剪枝及配置开关，只保留有效数值切点。
无有效候选时继续学习；固定最大深度上限也已删除，仅内存限制仍可停用叶子。没有引入重新激活或重置。
2026-09-26加入已测试的均值中点回退：仅原候选全无效时尝试，收益须>0、两侧估算占比须>1%；不清空历史，不启用预剪枝。
模型策略版本为6，Trace配置hash相应改变；近期历史实验并非本次修改后的验证。
历史C2H残差/扩展特征实验工厂仍保留其StandardScaler，不能作为当前线上基线；
标准make_hp_replay_model已取消StandardScaler。

依据：本机已校验的River0.26.1源码（此前官方全默认离线实验venv），
forest/adaptive_random_forest.py、tree/nodes/leaf.py、tree/utils.py、
tree/splitter/gaussian_splitter.py。公开参数也可参见
[River ARFClassifier文档](https://riverml.xyz/0.22.0/api/forest/ARFClassifier/)。
本地来源校验记录：/home/chris/ceph-tool/results/hp-river-full-20260925/official-source-audit.json。

Trace特征schema不变，config_hash加入模型策略版本、grace、lambda、切点数和delta，
新旧预测结果不能混作同一算法配置。测试入口：bash test_sh/test_hp_model_regressions.sh。
本轮验证记录：/home/chris/ceph-tool/results/hp-river-eight-dev-20260925/REPORT.md。
后续五负载离线中点对照见 /home/chris/ceph-tool/results/hp-midpoint-growth-20260926/REPORT.md；不是线上验收。dev当前总览见 DEV_ALGORITHM_STATUS.md。

投票Accuracy取每棵树训练前的累计预测正确率，逐原始训练样本更新，
不乘Poisson的k；Accuracy为0时保持单位权重回退，与River投票条件一致。
实时权重刷新和未缓存权重的快照构造均已修改；对外Balanced Accuracy统计仍保留。
