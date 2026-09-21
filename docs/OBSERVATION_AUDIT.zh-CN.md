# 三组实验的视觉与 privileged state 审计

更新：后续第四组已完成用户确认的 **RGB-D + 本体感知** 实验，策略不接收物体真值或在线结构 oracle。空任务技能库学习 2 轮达到 3/3，冻结后额外 2/2；见 [视觉学习报告](VISION_FROM_SCRATCH.zh-CN.md)。下面保留前三组的审计，不将新实验结果追溯套用于旧策略。

审计对象为本对话实际执行的三组实验：双块基线、初始三层塔、当前助手四轮反馈学习后的三层塔。结论：**三组都使用 privileged state，均不是纯视觉控制。**

| 实验 | 策略实际使用的输入 | 图像用途 | 分类 |
|---|---|---|---|
| 双块基线 | 模拟器目标块和底座的真实世界坐标；底层 IK 使用模型和当前关节状态 | 关键帧、录像、浏览器展示 | Privileged-state policy |
| 初始三层塔 | 物体真实位置、世界坐标系半尺寸、末端位置、关节角；已知机器人运动学与关节限位 | 参考视频/开发诊断、录像与回放 | Privileged-state feedback policy |
| LLM 反馈学习三层塔 | 继承上一组全部输入，并用真实关节角做到位反馈；外层助手还读了物体真实速度、位置和结构检查结果 | 助手辅助诊断失败与核对结果、保存证据、录像 | 外层图像+真值轨迹反馈；内层 privileged-state policy |

## 数据来源的代码证据

- `src/aspire_so100/stacking_env.py::_object_state`：物体位置来自 `data.xpos`，姿态来自 `data.xquat`，速度来自 `data.qvel`。这些不是从 RGB 或 depth 推断出的状态。
- 同文件 `observation`：末端位置来自 `data.site_xpos`，关节角来自 `data.qpos`。
- `src/aspire_so100/tower.py::extent`：用实际几何旋转 `geom_xmat` 和模型 `geom_size` 计算世界坐标半尺寸；`TowerAPI.get_observation` 将其作为 `half_extent_world` 暴露给策略。
- `policies/stack_small_red_on_blue.py` 第 7–9 行读取物体真值位置。
- `policies/stack_three_blocks.py` 的 `run` 读取目标/支撑块位置、半尺寸与末端偏移，并依据关节状态求解 IK。
- `policies/stack_three_blocks_learned.py` 继承这些读数，增加关节到位判断。四轮实际候选均调用 `get_observation`，均未调用 `get_overhead_rgbd`。
- 三个当前策略文件的 SHA-256 均与对应已录制 Demo 的 `recording.json` 一致，审计对象就是生成视频时使用的代码。

## 评估真值和策略真值要分开

评估器用真值计算成功率是常见做法，本身不能证明策略用了 privileged state；这里的关键证据是**执行策略直接读取了物体真值并据此规划、修正动作**。

另外，两种三层塔策略都会调用 `api.check_structure`，并根据其返回结果决定继续或终止。该结果使用真实接触、姿态、速度、质心和漂移，因此真值结构判定也进入了策略决策，而非只在运行结束后做离线评分。

关节角通常可由真实机器人编码器获得，是本体感知，不应单独等同于不现实的特权信息；已知运动学模型也不必然违规。但本项目的物体真值位置、几何范围和接触/稳定性 oracle 足以确定这三组不是 vision-only。

## 图像没有参与在线动作选择

`scripts/record_demo.py` 先运行策略并记录物理状态，再恢复状态渲染 RGB、depth 和腕部画面。`scripts/evaluate_tower.py --visuals` 保存初末图像。**视频里有多路相机不等于控制器以图像为输入。**

在直接学习实验中，助手确实看过第 1、3 轮末态相机图，但修复同时依赖真值轨迹中约 54 mm 的抬升量、约 0.133 m/s 的速度与关节信息，因此外层学习过程也不能标为纯 vision。

## 对已有结果应采用的表述

已有成功率、耗时改善与稳定性检查结果仍然有效，但都应标注为：**MuJoCo 仿真、privileged-state-assisted control、已知模型、有限布局扰动**。这些结果未验证纯视觉感知，也不能作为 vision-only 的性能报告。

若后续开展视觉实验，应先明确是否允许本体感知和标定信息，再从控制 API 中移除物体真值、真实几何范围和结构 oracle，用 RGB/RGB-D 感知结果替代；成功判定保留在策略不可访问的评估器中。训练/修复阶段接触到的真值反馈也应单独披露。这是后续实验设计建议，本次审计没有修改或重跑策略。
