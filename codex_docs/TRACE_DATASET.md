# Heat Predictor Trace

Trace 仅存在于 `dev`，默认关闭，用于离线复现 V2 已评估 I/O、feature 消融和错误
样本分析。

## 接口与写入路径

```bash
sudo ceph osd hp trace start PHASE DIRECTORY
sudo ceph osd hp trace stop
```

start 会先排空并关闭旧 session，再轮换到新文件。reset、enable 和 disable 在 Trace
已开启时轮换 session，避免不同实验混合。单 OSD 和 MGR status 输出开关、队列、
写入、丢弃和错误计数。

前台和 EQ 到期线程只调用非阻塞 `try_submit()`，记录进入容量65536的 MPSC ring。
独立 writer 线程每批最多写4096条定长记录；仅 ring 已满时增加 drop。

## Schema 4

文件头固定192字节，记录固定184字节。头部保存 schema、结构大小、feature 数、
OSD/session、墙钟与单调时钟锚点、配置哈希、Git 版本和 phase。

完成记录包含：

- 匿名 object key、I/O 序号和预测/标签时间；
- 五个预测时 feature、当前热度、热概率和固定预测阈值；
- `K_context`、过去10秒访问数、预测后累计访问计数和上次访问间隔；
- 未来10秒访问数、到期微批的 `K_window`、预测标签、实际标签及 cold-start 标志。

实际标签必须满足：

```text
actual_label = future_window_access_count >= K_window_at_deadline_microbatch
```

deadline 按1ms划分微批，同一微批使用最后一个实际 deadline 的访问窗口和一次 Otsu
结果。因此 Trace 中的 `K_window` 是该微批真实使用的阈值，时间近似误差小于1ms，
并非为每条 I/O 单独扫描直方图得到的阈值。

容量丢弃和预测错误记录使用 `actual_label=-1`，不进入正常混淆矩阵。

## 转换、分析与回放

```bash
python3 test_sh/convert_hp_trace.py TRACE.bin TRACE.csv

python3 test_sh/analyze_hp_trace.py \
  --run-root RUN_ROOT \
  --output-dir RUN_ROOT/offline_analysis
```

转换器严格校验 magic、schema 和结构大小。分析器校验时间顺序、概率/预测一致性及
未来访问标签，输出混淆矩阵、校准、margin、feature、object、阶段和30秒时序报告。
标签 margin 定义为：

```text
log2(1 + future_access_count) - log2(1 + K_window_at_deadline_microbatch)
```

精确回放按 OSD 独立执行，并复用当前模型与快照发布规则：

```bash
g++ -std=c++17 -O2 -pthread \
  -Ibuild/src/include -Ibuild/include -Isrc -Itest_sh \
  test_sh/hp_trace_replay.cc -Lbuild/lib -Wl,-rpath,build/lib \
  -lceph-common -o /tmp/hp_trace_replay

/tmp/hp_trace_replay TRACE.bin --output replay.tsv
```

回放只能解释 Trace 中已记录的策略；schema 4 仍未保存预测快照代次，不能保证跨线程
发布时序逐条完全复现。

## 验证

- `hp_trace_probe`：文件头、正常/异常记录、排空和 session 轮换。
- `convert_hp_trace.py --self-test`：schema 4 二进制转换。
- `test_hp_trace_replay`：V2 记录加载、替换和回放。
- `test_hp_trace_analysis.py`：V2 标签契约和离线报告。
