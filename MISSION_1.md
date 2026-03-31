你现在是这个仓库的主开发代理。请从零开始为一个基于 REASSEMBLE 数据集的“最小化接触状态 observer 实验项目”搭建代码骨架、数据管线、训练脚本和基础文档。

## 总目标
构建一个可运行、可扩展、可复现实验的 Python 项目，用于在 REASSEMBLE 数据集上完成以下最小实验：

1. insert 子集上的阶段辨识（phase recognition）
2. insert 子集上的接触模式辨识（contact mode recognition）
3. 一个最小化 observer，用历史 visual + F/T + pose + structure token 恢复低维 latent state，并用该 latent state 支持多任务预测：
   - 当前 phase 分类
   - 当前 contact 分类
   - success/failure 预测
   - next-step pose delta 或 wrench 预测

## 已知数据前提
请按以下真实数据假设设计接口，不要自行发明不必要的数据格式：
- 数据集由 HDF5 `.h5` 文件和同名 `*_poses.json` 文件组成
- 不同传感器的时间戳是分开存储的，因此必须实现统一时间轴对齐模块
- action segments 位于 `segments_info`
- 每个高层 segment 下有 `low_level` 技能段，因此 phase 任务第一版优先复用官方 low-level 标注，而不是重新发明完整阶段标签

## 当前开发原则
- 先做最小可运行版本，不做过度设计
- 先支持 F/T + pose；vision 接口先留好，但实现优先级低于非视觉 baseline
- 所有模块都需要可测试、可替换
- 优先保证清晰的数据流和实验可复现性
- 如遇不确定设计，优先写成可扩展接口，并在文档中记录 decision note
- 不要编造数据字段；如果字段名无法确认，请把读取逻辑设计成可配置，并在 TODO 中显式标出

## 目录结构要求
请创建并填充如下结构，允许你根据实现需要补充少量文件，但不要大幅偏离：

reassemble_minexp/
  AGENTS.md
  README.md
  pyproject.toml
  requirements.txt
  .gitignore

  configs/
    dataset.yaml
    train_phase.yaml
    train_contact.yaml
    train_observer.yaml

  scripts/
    00_scan_files.py
    01_build_index.py
    02_extract_insert_segments.py
    03_align_modalities.py
    04_make_phase_labels.py
    05_make_contact_labels.py
    06_export_windows.py
    07_train_baseline.py
    08_train_observer.py
    09_eval.py
    10_plot_cases.py

  src/
    io/
      h5_reader.py
      pose_json_reader.py
      segment_parser.py
      timestamp_aligner.py
    dataset/
      trial_index.py
      insert_index.py
      window_dataset.py
    labels/
      phase_mapper.py
      contact_rule_labeler.py
    models/
      mlp.py
      gru.py
      tcn.py
      observer.py
      heads.py
    train/
      trainer.py
      losses.py
      metrics.py
    eval/
      evaluator.py
      visualizer.py
    utils/
      paths.py
      logging_utils.py
      seed.py
      config.py

  notebooks/
    inspect_one_trial.ipynb
    check_labels.ipynb

  tests/
    test_config.py
    test_trial_index.py
    test_timestamp_aligner.py
    test_phase_mapper.py

  outputs/
    .gitkeep

## 每个阶段必须完成的事情

### Milestone 1: 项目骨架与配置系统
完成以下内容：
- 初始化 Python 项目
- 选择一个简单稳妥的技术栈（建议 Python + PyTorch + h5py + pydantic/omegaconf）
- 建立统一配置加载方式
- 写 README 的项目简介、目录说明、运行顺序
- 写 AGENTS.md，约束后续开发行为

AGENTS.md 至少包含：
- 先读 README 再改代码
- 每次完成一个 milestone 后必须更新 README 或 docs note
- 修改数据流程代码后，优先运行相关 tests
- 不引入重量级依赖，除非明确必要
- 所有新增脚本必须能通过 `python scripts/xxx.py --help`

### Milestone 2: 数据扫描与索引
实现：
- `00_scan_files.py`
- `01_build_index.py`

要求：
- 扫描原始数据目录
- 建立 file manifest
- 建立 trial index
- 解析 high-level / low-level segment 元数据
- 记录 success/failure、对象名、模态可用性、时间范围
- 输出 CSV 到 `data/processed/` 或配置指定目录

要求代码结构：
- 脚本层只做参数解析和调度
- 具体逻辑放到 `src/io` 和 `src/dataset`

### Milestone 3: insert 子集提取
实现：
- `02_extract_insert_segments.py`

要求：
- 从 trial index 中抽取所有 insert 片段
- 为每个 insert 片段分配唯一 ID
- 保留其对应 low-level skill sequence
- 生成 `insert_index.csv`

### Milestone 4: 时间对齐模块
实现：
- `timestamp_aligner.py`
- `03_align_modalities.py`

要求：
- 支持以配置指定的 target frequency 生成统一时间轴
- 支持 pose/F/T 的最近邻和线性插值
- 支持 RGB 的最近帧索引映射
- vision 先只做“帧索引对齐”，不做重型预处理
- 输出对齐后的 sample 索引，而不是复制整段原始大数据

这是最核心模块，请优先保证接口干净、文档完整、单元测试齐全

### Milestone 5: 标签系统
实现：
- `04_make_phase_labels.py`
- `05_make_contact_labels.py`
- `phase_mapper.py`
- `contact_rule_labeler.py`

要求：
- phase 第一版直接基于 low-level skills 做 remap
- contact 标签先实现规则弱标签，至少支持以下类：
  - free
  - touch
  - search_contact
  - insertion_contact
  - jam_or_abnormal
- contact 规则阈值必须由 config 控制，不要硬编码
- 输出标签质量检查所需的可视化辅助函数接口

### Milestone 6: 训练样本导出
实现：
- `06_export_windows.py`
- `window_dataset.py`

要求：
- 支持固定长度历史窗
- 支持 stride
- 支持导出训练/验证/测试 split
- 样本至少包含：
  - ft history
  - pose history
  - optional rgb frame refs
  - structure token
  - y_phase
  - y_contact
  - y_success
  - y_next_delta

如果 structure token 暂时无法从原始数据自动得到，请先实现可从外部 YAML/CSV 注入的接口

### Milestone 7: baseline
实现：
- `07_train_baseline.py`
- `mlp.py`
- `gru.py`
- `metrics.py`
- `losses.py`
- `trainer.py`

要求先支持三组 baseline：
1. F/T only
2. Pose only
3. F/T + Pose

要求：
- phase 和 contact 分类至少支持 macro-F1
- success/failure 支持 accuracy 和 F1
- 有最小 checkpoint 保存与日志输出
- 配置驱动运行，不要把实验参数散落在代码里

### Milestone 8: observer
实现：
- `08_train_observer.py`
- `observer.py`
- `heads.py`

要求：
- 一个最小 observer encoder（优先 GRU 或 TCN）
- 输出低维 latent state z_t
- 多头预测：
  - phase
  - contact
  - success/failure
  - next-step dynamics
- 支持 ablation 开关：
  - no_history
  - no_ft
  - no_pose
  - no_vision
  - no_structure

### Milestone 9: 评估与可视化
实现：
- `09_eval.py`
- `10_plot_cases.py`
- `evaluator.py`
- `visualizer.py`

要求：
- 支持按 split 输出结果
- 支持保存 confusion matrix、PR/F1 summary、若干 case plots
- case plot 至少叠加：
  - phase 标签
  - contact 标签
  - 关键 wrench 曲线
  - 关键 pose/位移曲线

## 工程约束
- 不要直接假定 GPU 必须存在；应兼容 CPU
- 不要把数据路径写死
- 不要把 notebook 当主流程；脚本才是主流程
- 每个脚本都要可单独运行
- 所有重要模块要加 docstring 和类型注解
- 不要为了“聪明”而过度抽象
- 不要静默吞掉异常，报错要可定位
- 不要引入数据库、消息队列、服务端框架这类无关复杂度

## 你工作的方式
严格按下面循环工作：
1. 先读仓库现状
2. 写一个简短计划
3. 每完成一个 milestone：
   - 修改代码
   - 运行最小必要测试/检查
   - 修复失败
   - 更新 README/状态说明
4. 再进入下一个 milestone

## 输出要求
现在请不要一次性写完整大而全解释。请直接开始干活，并遵守以下输出节奏：

- 先输出：
  1. 你理解的目标
  2. 你准备采取的 milestone 顺序
  3. 你将先检查哪些文件/目录
- 然后开始创建项目骨架和核心文件
- 在每个 milestone 结束时，输出：
  - 已完成文件
  - 关键设计决定
  - 运行过的命令
  - 当前未解决问题
  - 下一步

## Done when
只有在以下条件都满足时，才能认为第一轮搭建完成：
- 项目目录完整
- README 足够让新开发者按顺序运行
- 数据扫描、索引、insert 提取、时间对齐、标签生成、窗口导出、baseline 训练脚本全部存在并可运行到合理程度
- observer 训练脚本和评估脚本存在
- 关键模块有最小测试
- 所有 TODO 都被集中记录，而不是散落在代码各处

现在开始。