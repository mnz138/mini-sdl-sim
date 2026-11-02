# mini-sdl-sim

个人学习项目：基于 PyLabRobot 0.2.2 和 BayBE 0.15.0 的自驱动实验室仿真。
实验设备、测量曲线和目标函数均为仿真，不连接真实仪器。

测试范围、验收结果和重跑方法见 [测试说明](docs/TEST_REPORT.md)。

同一调度器支持三种实验配置：

| 配置 | 仿真测量与目标 |
| --- | --- |
| `electrolyte` | 由阻抗谱估计体电阻，电导率 `κ = 1000 K_cell / R_b`，单位 mS/cm。 |
| `tim` | 由激光闪射半升温时间计算 `α = 0.1388 d²/t_half`、`k = αρc_p`；优化带黏度惩罚的导热分数，单位 W/(m·K)。 |
| `cavity` | 由谐振曲线拟合频率和 Q，再反演 Dk/Df；优化 `−1000 Df` 加 Dk 与黏度超限惩罚后的分数。 |

## 环境与运行

本机验收环境为 Intel macOS、Python 3.12.13、PyTorch 2.2.2、NumPy 1.26.4。
`pyproject.toml` 已对 Intel macOS 限制 `numpy<2`，并固定兼容的 `pyarrow==20.0.0` 供网页表格使用。
项目搬动目录后应重建 `.venv`。
推荐使用 Python 3.12。首次下载项目后，在项目根目录创建环境并安装：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

以上使用 macOS / Linux 路径；Windows 的虚拟环境解释器路径为 `.venv\Scripts\python.exe`。
运行时直接指定项目解释器，可以避免混入 conda 或系统环境。

先用随机搜索跑小规模闭环，再运行 BayBE：

```bash
.venv/bin/python run_campaign.py --profile electrolyte --optimizer random --rounds 2 --batch 2 --p-timeout 0
.venv/bin/python run_campaign.py --profile electrolyte --optimizer bayes --rounds 2 --batch 2 --p-timeout 0
```

把 `--profile` 改为 `tim` 或 `cavity` 可切换实验。`--mode sequential` 顺序执行，默认 `pipeline`。
`--p-timeout 1` 强制每次测量失败，用于检查首次尝试加最多两次重试。
失败样品记录为 `failed`，结果为 `null`，不作为低分交给优化器。
配方本身不可行则记为 `done`、`feasible=false`，用配置中规定的惩罚值回流。

命令行日志默认写入 `data/` 下带时间戳的 JSONL，每个样品一行；同名 `.summary.json` 保存配置、版本、时间线和异常事件。
网页运行保存 JSONL，当前结果和时间线保存在页面会话中。
也可用 `--output data/my-run.jsonl` 指定位置；已有文件不会被覆盖。
`requested` 是推荐配方，`actual` 是配制后的配方。BayBE 学习实际值，即使称量误差让它略超出推荐范围，也不截断修改。
其下一轮推荐仍受搜索空间边界约束。

## 整体验收

```bash
# 三种配置各跑随机和 BayBE 两轮，再比较同一批 8 个配方的调度耗时
.venv/bin/python scripts/validate_project.py

# 额外跑电解液 6 轮 × 8 个样品，覆盖补液和换枪头
.venv/bin/python scripts/validate_project.py --full
```

验收输出保存在独立的 `data/validation-*/` 目录。这里比较的是合成模型下的程序行为；单个随机种子的最优值不能证明优化算法优劣。
模拟时间由压缩后的真实计时器计算，包含 Python 调度开销；不同机器的时间不应直接作为设备吞吐率比较。

## 网页

```bash
.venv/bin/python -m streamlit run app.py
```

网页包含收敛曲线、参数空间、96 孔板、调度时间线、样品记录和异常事件。
只在点击运行按钮后开始实验。TIM 和 cavity 不使用移液板，因此没有对应的孔板数据。
在 VS Code 的“运行和调试”中选择“网页前端”，点击绿色箭头或按 F5。
该启动项会跳过首次启动的 `Email:` 提示，并自动打开浏览器。
修改启动配置后，先点红色停止按钮，再重新启动，正在运行的进程不会自动应用新配置。
如果浏览器没有弹出，点击终端输出中的网址；已有服务占用默认端口时，新服务可能使用 8502 等端口，以实际输出为准。
普通 Python 文件可用“运行当前文件”或 Code Runner；测试文件应由 pytest 执行。

电解液单次运行最多 96 个样品；当前没有自动换样品板。单次移液最多 300 µL，样品孔容量 360 µL。
缺液测试应让槽内剩余量小于一次合法移液量，例如剩 20 µL 时取 50 µL，且补满后的储液量不少于 50 µL。

## 建议阅读顺序

`models/clock`、`lab`、`profiles`、`scheduler`、`optimizer/campaign`、命令行、网页。
先读对应测试中的输入、预期输出和失败条件，再解释实现如何满足这些条件。
