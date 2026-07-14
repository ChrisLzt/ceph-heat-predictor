在 /home/chris/ceph-heat-predictor 中修改 heat predictor。

目标：
1. 总目标：
    修改冷热识别代码，提升模型的 accuracy。
2. 性能要求：
    accuracy 要求不低于 85%，precision 和 recall 不得过低。
    pred_hot_percent 和 actual_hot_percent 误差不高于30%。
3. 关注重点：
    当前粒度是 object 级。EvaluationQueue 不能无限拉长，避免学习延迟过高；LRU/WT 可以覆盖更多 object 状态，但不能把测试设计成只靠记忆全部数据集取胜。
    可以优先查找模型的漏洞，并进行优化，也可以修改各种参数，我允许你修改代码，你有很高自由度。
4. 你可以将你认为重要的改动提交到 git，做好版本管理即可。

约束：
1. enum、perf 声明、更新、输出顺序必须一致。
2. 稳定实现变化同步更新 `codex_docs/CODEX_CEPH.md`；操作命令变化同步更新
   `codex_docs/CEPH_OPERATIONS_MANUAL.md`。
3. 代码改完后可以直接执行 install、ldconfig 和 restart，无需再次确认。

测试：
1. 先按 `AGENTS.md` 和 `codex_docs/CODEX_CEPH_TODO.md` 判定 L0-L3，不无条件运行
   更高检查级别。
2. L2 全量构建和安装：
    cd /home/chris/ceph-heat-predictor/build
    sudo env CCACHE_TEMPDIR=/tmp ninja -j64
    sudo ninja install
    sudo ldconfig
3. L2 重启：
    sudo systemctl reset-failed ceph-osd@0 ceph-osd@1
    sudo systemctl reset-failed ceph-mgr@s52
    sudo systemctl restart ceph-osd@0 ceph-osd@1
    sudo systemctl restart ceph-mgr@s52
4. L3 测试负载：
    路径：/home/chris/ceph-test/new_workload
    在每次测试前，清空 mgr 的统计信息（指令：sudo ceph osd hp reset）
    在每次测试时，观察 mgr 的统计信息（指令：sudo ceph osd hp status -f json-pretty）
    每个正式负载只运行一次；若出现巨大误差、结果异常或与历史明显冲突，只在报告中
    指出并停止。是否复测及复测次数由用户决定，不得自动追加测试。

验收标准：
- L0/L1 以文档检查、探针和目标编译结果为准。
- L2 要求 OSD/MGR 正常启动，PG `active+clean`，且只允许单副本相关 HEALTH_WARN。
- 涉及新统计字段时，字段必须同时出现在 OSD perf 和 MGR 汇总。
- L3 还要求训练队列清空、drop 为 0、计数关系成立并生成报告。
