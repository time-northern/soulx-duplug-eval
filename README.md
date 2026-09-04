# SoulX state-only evaluation

该框架只根据 SoulX-Duplug 的逐块 state 输出评测打断与拒识，不加载对话模型，也不包含 TTS 取消链路。推理证据和指标计算完全分离：`infer/` 负责生成不可变 JSONL，`eval/` 只读取 JSONL 生成两个场景报告。

## 评测口径

| 数据 | 通过条件 |
| --- | --- |
| interruption | 标注语音窗口内至少出现一次 public `nonidle`；首次检出的 `current_time - onset` 作为时延，`< 1.0s` 只作为 on-time 诊断指标，不影响检测通过 |
| backchannel | 窗口内至少一个 raw `backchannel`，且没有 public `nonidle` |
| background speech（可选诊断） | 第二段/干扰语音开始后到音频结束没有 public `speak`；不进入默认推理和正式指标 |
| talking to other（可选诊断） | 第三段语音结束后到音频结束没有 public `speak`；不进入正式场景 |
| background noise | SID-Bench 噪音最多取前 20 秒，再追加 2 秒静音；完整输入期间没有 public `speak` |
| Easy-Turn complete/incomplete | 音频后追加 2 秒静音后，最后一个 raw `complete`/`incomplete` 终态分别判为 complete/incomplete；无终态即错误 |

raw state 是模型生成的状态 token；public state 是 `service.model.TurnModel` 应用 far-field、turn-state 等部署逻辑后对外返回的 `state`。主评测保持 `far_field_threshold: 0.02`。

事件窗口为半开区间 `[start,end)`，按 160 ms 输入块的中心点归属窗口。英文 background speech 使用 `metadata.json.timestamps[0]` 作为干扰语音起点，中文 background speech 使用同名 JSON 的 `speech_segments[1].xmin`；两者的终点都是实际音频结尾。中文 interruption 读取 `interrupt.json[0].timestamp`。Full-Duplex-Bench 只搜索文件名严格等于 `input.wav` 的音频，不读取 `clean_input.wav` 或 `context.wav`；HumDial 中文 background speech 忽略 `clean_*.wav`。SID-Bench 噪音规则检查最多 20 秒的源噪音、随后追加的 2 秒静音，以及末块右侧补零产生的全部实际 state 输出。

## current_time

对第 `i` 个块：

```text
available_i = (i + 1) * chunk_duration
start_i = max(available_i, current_time_(i-1))
current_time_i = start_i + process_time_i
```

`current_time` 是从样本开始实时播放后，该 state 最早可供上层使用的虚拟在线时间。`process_time` 使用 `time.perf_counter()` 测量；CUDA 推理在计时前后同步。音频读取和重采样不计入，模型按语言预热后再正式计时。若单块计算超过 160 ms，积压会通过下一块的 `queue_delay_sec` 继续累计。

## 数据和配置

所有路径、阈值与语言模型配置都在 `eval_config.yaml` 中。相对路径以仓库根目录为基准。主流程不会联网下载数据；远程主机需要已有：

```text
SoulX-Duplug-Eval/data/Easy-Turn-Testset-zh/complete/*.wav
SoulX-Duplug-Eval/data/Easy-Turn-Testset-zh/incomplete/*.wav
```

推理前会全量审计音频、时间戳、事件边界、空窗口、重复 ID、模型配置和 checkpoint。任何错误都会写入 `errors.jsonl`，manifest 保持 `complete: false`，正式指标不会生成。样本数量取实际通过审计的文件数，不硬编码数据卡中的名义数量。

## 运行

`run_all.sh` 是 Slurm 作业脚本，默认申请 `gpu_ai` 分区、1 个任务、12 个 CPU 和 1 张 GPU。使用 `sbatch` 提交完整评测：

```bash
sbatch SoulX-Duplug-Eval/run_all.sh \
  --config SoulX-Duplug-Eval/eval_config.yaml \
  --run-id bilingual-main-001
```

如果不传 `--run-id`，默认使用 Slurm job ID。推理开始前仍会自动执行严格的数据、时间戳、配置和 checkpoint 校验；校验失败时作业直接退出且不会生成正式指标。

`sbatch` 提交后会立即返回，Slurm 不会把计算节点输出主动推送到已经返回提示符的终端。脚本已启用 Python 无缓冲输出，并在日志中实时写入资源信息、推理阶段、逐样本进度、两类评测阶段及最终状态。提交后可以观察：

```bash
tail -F soulx_state_eval_<job_id>.out
```

也可以使用封装脚本完成“sbatch 提交 + 当前终端自动跟随日志”；它内部仍然通过 `sbatch` 提交 `run_all.sh`：

```bash
bash SoulX-Duplug-Eval/submit_and_follow.sh \
  --config SoulX-Duplug-Eval/eval_config.yaml \
  --run-id bilingual-main-001
```

按 `Ctrl-C` 只会停止终端日志跟随，不会取消 Slurm 作业。

跟随脚本提交时会显式把 Slurm 工作目录和 `PROJECT_ROOT` 设置为 SoulX-Duplug 仓库根目录，避免 `sbatch` 将脚本复制到 `/var/spool/slurmd/job.../` 后错误地从临时目录寻找配置和 Python 文件。直接使用 `sbatch SoulX-Duplug-Eval/run_all.sh` 时，应先 `cd` 到仓库根目录；也可以显式导出 `PROJECT_ROOT`。

运行目录不允许覆盖。同一个 run ID 已存在时应换一个新 ID。诊断时可传 `--limit-per-dataset N`；这种运行只做子集推理，manifest 不会标记完整，也不会生成正式指标。

如果推理已经成功，可以在不加载模型的情况下重复生成报告：

```bash
python SoulX-Duplug-Eval/eval/eval_interruption.py \
  --config SoulX-Duplug-Eval/eval_config.yaml \
  --run-id bilingual-main-001

python SoulX-Duplug-Eval/eval/eval_rejection.py \
  --config SoulX-Duplug-Eval/eval_config.yaml \
  --run-id bilingual-main-001
```

若要在已有完整、零错误的 run 中只重跑 interruption 推理，可提交专用
Slurm 作业。它使用该 run 保存的中英文 runtime config，只原子替换
`inference/interruption.jsonl`，保留其他场景证据，并重新生成三份报告：

```bash
sbatch SoulX-Duplug-Eval/rerun_interruption.sh \
  --run-id <existing-complete-run-id>
```

## 输出

```text
output/<run_id>/
├── manifest.json
├── errors.jsonl
├── configs/{en,zh,neutral}_config_used.yaml
├── inference/{interruption,backchannel,background_noise,easy_turn}.jsonl
└── evaluation/
    ├── interruption/{samples.jsonl,metrics.json,report.md}
    ├── rejection/{samples.jsonl,metrics.json,report.md}
    └── vad/{samples.jsonl,metrics.json,report.md}
```

报告分别按语言和类别列出通过率，并按实际样本数计算有效意图与无效意图的加权指标。不会把两类指标混成一个总准确率。interruption 检测通过率只统计标注语音窗口内是否出现 public `nonidle`；`on_time_interruption_rate` 单独统计首次检出时延严格小于阈值的比例。时延均值、P50、P95 使用所有已检出样本（包括晚检），漏检为 `null` 并单独计数。VAD 报告采用冻结的 Table-3 终态口径：仅 Easy-Turn 音频追加 2 秒静音，并从 raw state trace 的最后一个 `complete`/`incomplete` 终态得到类别；public `system_speak`、far-field 部署映射和 Silero endpoint 时延均不参与该指标。背景噪声使用 SID-Bench 正式评测并计入无效意图拒识率。

## 指定 checkpoint

完整推理与评测可用 `run_all.sh --checkpoint <path>` 临时覆盖 `eval_config.yaml` 中的 `model.checkpoint`，不需要修改默认配置。该路径相对于仓库根目录（也可使用绝对路径），并会记录到本次运行的 `manifest.json`。例如：

```bash
sbatch SoulX-Duplug-Eval/run_all.sh \
  --checkpoint SoulX-Duplug-Training-Official/ckpt/example/model.ckpt \
  --run-id checkpoint-eval-example
```

## Background-noise duration cap and trailing silence

正式评测使用全部 SID-Bench `background_noise/neutral` 样本。对原始时长超过 20 秒的音频，源噪音只使用其前 20 秒；随后所有样本统一追加 2 秒静音，并在完整输入期间检查 public state。源噪音上限和尾随静音分别由 `eval_config.yaml` 的 `evaluation.background_noise_max_duration_sec` 与 `evaluation.background_noise_trailing_silence_sec` 配置，并记录在运行的 `manifest.json` 及每条 inference 的 `audio` 元数据中。完整 state trace 中只要出现一次 public `speak` 就判失败，否则通过。

若已有完整运行只需按此规则替换噪声结果，可提交专用 Slurm 作业。它只覆盖 `inference/background_noise.jsonl`，保留其余场景的 inference，并自动重跑三个 `eval/` 脚本：

```bash
sbatch SoulX-Duplug-Eval/rerun_background_noise.sh \
  --run-id <existing-complete-run-id>
```

## 指标汇总

正式主指标使用分层 macro average，而不是按样本数加权：每个双语场景先对 `en` 和 `zh` 的通过率等权平均；随后 `effective intent` 与 `invalid intent` 分别对其场景等权平均。`background_noise/neutral` 是语言无关场景，在语言层只有一个 `neutral` 分量，但在无效意图的场景层与 `backchannel` 等权。`background_speech` 不进入默认推理、正式指标或技术目标检查。

报告的 `aggregates` 是主 macro 指标，含各层分量及实际样本证据数；`sample_weighted_aggregates` 保留原先按样本数计算的参考指标。Easy-Turn VAD 的 complete/incomplete accuracy 同样按语言 macro 作为主指标。每个语言或场景缺失时，只平均实际存在的分量。

## 测试

测试不需要真实 SoulX checkpoint：

```bash
python -m unittest discover \
  -s SoulX-Duplug-Eval/tests \
  -p "test_*.py" -v
```

## 单独运行或重跑 VAD

针对任意 checkpoint 新建正式 VAD-only run：

```bash
sbatch SoulX-Duplug-Eval/run_vad.sh \
  --run-id <new-run-id> \
  --checkpoint <checkpoint-path>
```

该 run 默认写入 `SoulX-Duplug-Eval/output/<new-run-id>`，直接复用正式
`infer/easy_turn/` runner 和 `eval/eval_vad.py` 的冻结协议：中英文
complete/incomplete 四个 split、音频后 2 秒静音、最后一个 upstream
`speak`/`wait` 终态分别判为 complete/incomplete，并生成语言 macro 指标。

已有完整、零错误的评测 run 可以只重跑 Easy-Turn/VAD 推理。该入口直接复用正式
协议并使用 manifest 中记录的 checkpoint：

```bash
sbatch SoulX-Duplug-Eval/rerun_vad.sh \
  --run-id <existing-complete-run-id>
```

它不会重新运行 interruption、backchannel、background speech 或 background noise
推理。由于 Easy-Turn 证据也出现在 run 级报告中，替换完成后会从已有推理 JSONL
重新生成 interruption、rejection 和 VAD 三份报告。仓库根目录旧入口
`run_easy_turn_compare.sh` 仅作为兼容转发器保留，不再包含独立评测协议。

## HumDial 中文 Talking to Other

该场景暂时不属于默认正式评测，只保留单独诊断入口。

中文 `talking_to_other` 使用：

```text
SoulX-Duplug-Eval/data/HumDial-FDBench/talk_to_others_cn
```

框架只读取该目录中不以 `clean_` 开头的三段完整 WAV，并读取同名 JSON 的
`speech_segments[2].xmax` 作为第三段语音结束点。`clean_*.wav` 是两段清洁副本，
不会进入评测；英文 HumDial 数据也不会进入评测，英文数据继续使用配置中的
Full-Duplex-Bench-en v1.5。

通过条件是从第三段结束到音频结束的半开区间内没有 public `speak`。该窗口内
出现 `idle`、`nonidle`、`backchannel` 或 `blank` 均不会单独导致失败。可单独运行：

```bash
python SoulX-Duplug-Eval/infer/infer_talking_to_other.py \
  --config SoulX-Duplug-Eval/eval_config.yaml \
  --run-id humdial-zh-talking-to-other
```

## Background speech（可选诊断）

`background_speech` 已从默认正式评测中移除，但保留数据适配与独立推理入口。
中文诊断数据使用：

```text
SoulX-Duplug-Eval/data/HumDial-FDBench/others_talk_to_user_after
```

完整输入必须恰好包含两段语音，第二段是第三方干扰语音。框架排除 `clean_*.wav`，
读取同名 JSON 的 `speech_segments[1].xmin` 作为窗口起点，并以实际音频结尾作为
窗口终点。窗口内没有 public `speak` 即通过，其他 public state 不单独导致失败。
任何 `speech_segments` 数量不等于 2 的样本都不会进入测试；当前源数据中
`0006_0035.json` 含三个语音段，因此被排除。

英文 Full-Duplex-Bench v1.5 background speech 采用相同规则：使用
`metadata.json.timestamps[0]` 作为起点，并将窗口延伸到实际音频结尾。

可单独运行诊断推理：

```bash
python SoulX-Duplug-Eval/infer/infer_background_speech.py \
  --config SoulX-Duplug-Eval/eval_config.yaml \
  --run-id background-speech-diagnostic
```

该入口生成诊断 run，不会被 `run_all.sh` 调用，也不计入 interruption 或 rejection
正式报告。

## SID-Bench 纯噪音

`background_noise` 只使用：

```text
SoulX-Duplug-Eval/data/SID-bench/silence_noise_test.jsonl
SoulX-Duplug-Eval/data/SID-bench/silence_or_noise/*.wav
```

这 500 条音频不含语音，包含低噪音、静音和环境噪音。框架校验每条标注均为
`total_nonbreak=true`、`text_with_break=null`，并检查标注时长、WAV 时长和文件集合。
每条源噪音最多取前 20 秒并追加 2 秒静音；完整推理期间任意时刻出现 public
`speak` 即判为失败，其余 public state 不单独导致失败。

SID-Bench 是无语言语境数据，只用 `neutral` 运行配置和 SenseVoice `auto` 评测一次，
不会再按英文和中文重复统计。原 FD-bench-noisy 预处理及评测流程已删除，框架不会
读取 `data/FD-bench-noisy`。

可单独运行诊断推理：

```bash
python SoulX-Duplug-Eval/infer/infer_background_noise.py \
  --config SoulX-Duplug-Eval/eval_config.yaml \
  --run-id sid-noise-diagnostic
```
