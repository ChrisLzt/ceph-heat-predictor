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
2. 同步更新 CODEX_CEPH.md 和 CEPH_OPERATIONS_MANUAL.md。
3. 代码改完后可以直接执行 install、ldconfig 和 restart，无需再次确认。

测试：
1. 编译：
    cd /home/chris/ceph-heat-predictor/build
    sudo env CCACHE_TEMPDIR=/tmp ninja -j64
    sudo ninja install
    sudo ldconfig
2. 重启：
    sudo systemctl reset-failed ceph-osd@0 ceph-osd@1
    sudo systemctl reset-failed ceph-mgr@s52
    sudo systemctl restart ceph-osd@0 ceph-osd@1
    sudo systemctl restart ceph-mgr@s52
3. 测试负载：
    路径：/home/chris/ceph-test/new_workload
    在每次测试前，清空 mgr 的统计信息（指令：sudo ceph osd hp reset）
    在每次测试时，观察 mgr 的统计信息（指令：sudo ceph osd hp status -f json-pretty）

验收标准：
- 编译通过。
- OSD/MGR 能正常启动。
- 新字段在 OSD perf 和 MGR 汇总里都有。
- ceph -s 只允许出现单副本相关 HEALTH_WARN。
