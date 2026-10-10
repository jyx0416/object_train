# ATEC 2026 西瓜感知仿真

当前阶段提供一个不修改 Conda 环境的基础 USD 场景生成器。场景依据 `atec.pdf`
第 9--13 页构建，目标物引用（而非复制或修改）根目录下的 `watermelon.usdz`。

## 已落实的比赛约束

- 目标类别：`watermelon`，参考尺寸为长 20 cm、上宽 5 cm、下宽 6 cm。
- 户外赛道地形：泥地、鹅卵石、草地、台阶和坡道。
- 台阶单级高度使用 16 cm（规则上限 17 cm）。
- 坡道使用 8°（规则要求小于 10°）。
- 泥块、草叶和石子不带目标语义，作为背景干扰物。
- USDZ 实测为 Y-up、米制，包围盒约 `5.70 x 8.69 x 19.21 cm`；导入 Z-up
  场景时做坐标系旋转，但不做额外缩放。

## 生成场景

```bash
conda run --no-capture-output -n rl_mani_lab \
  python scripts/build_atec_scene.py
```

输出为 `assets/scenes/atec_perception_scene.usda`。脚本只使用已有的 `pxr`，不会
安装/升级依赖，也不会写入 `rl_mani_lab` 环境。`--seed` 可改变静态预览中的杂物
分布和目标姿态。

在可正常执行 `nvidia-smi` 的图形会话中，可渲染五种地形预览：

```bash
XDG_CACHE_HOME=/tmp/atec_isaac_cache \
XDG_CONFIG_HOME=/tmp/atec_isaac_config \
XDG_DATA_HOME=/tmp/atec_isaac_data \
conda run --no-capture-output -n rl_mani_lab \
  python scripts/render_scene_preview.py
```

输出目录为 `outputs/scene_preview`。Isaac Sim 6.0.1 需要 NVIDIA 驱动及支持 RTX 的
GPU；脚本在没有真正写出图片时会明确失败，不会把空目录误报为成功。

下一阶段将在此基础上增加 Isaac Sim Replicator 数据采集：每帧只保留一枚目标，
随机选择地形、落稳姿态、相机距离/俯仰、太阳方向与强度、曝光和干扰物，并同步
导出 RGB、实例分割及由可见区域计算的 YOLO 框。训练集、验证集和测试集会按随机
种子拆分，避免同一背景布局泄漏到不同集合。

## 生成 YOLO 数据

### 多类别 YOLO 数据

多类别采集池包含 `assets/objects` 下全部模型：

- `watermelon.usdz`、`watermelon02.usdz` -> `watermelon`
- `can01.usdz`、`can02.usdz` -> `can`
- `lunch_box01.usdz`、`lunch_box02.usdz` -> `lunch_box`
- `red_bag.usdz` -> `red_bag`
- `purple_bag.usdz`、`sand_bottle.usdz` 作为不带标签的干扰物

目标池保留原来的 12 个目标 Prim 配置，干扰物池保留原来的 4 个 Prim 配置。
每张图随机放置 0--6 个物体（干扰物计入总数），允许空背景。每个环境单独
启动一次 Isaac Sim，异常退出时按已写出的 PNG/label 对断点续采：

```bash
scripts/collect_yolo_dataset.sh all 1000 data/data01 4100 --fresh
```

也可一次只收集一个环境，例如 `mud`：

```bash
scripts/collect_yolo_dataset.sh mud 1000 data/data01 4100 --fresh
```

先确认当前终端的 `nvidia-smi` 正常，然后生成 10 张检查样本：

```bash
XDG_CACHE_HOME=/tmp/atec_isaac_cache \
XDG_CONFIG_HOME=/tmp/atec_isaac_config \
XDG_DATA_HOME=/tmp/atec_isaac_data \
conda run --no-capture-output -n rl_mani_lab \
  python scripts/generate_yolo_data.py --num-images 10
```

数据会追加写入 `data/Images` 和 `data/labels`，类别文件为 `data/classes.txt`。
每个标签采用标准 YOLO 检测格式：`class_id x_center y_center width height`，坐标均
归一化到 0--1。程序默认从已有最大编号继续追加，不覆盖原数据；只有明确传入
`--overwrite` 才会使用相同编号覆盖。

### 近距离拍摄修正

- 采集相机显式使用 `clippingRange=(0.01, 100)`，避免 Replicator 相机默认近裁剪面
  裁掉近距离目标/地面而在图片底部和目标上出现黑块。
- 目标最低点按地形表面直接校正；不使用刚体、碰撞体或物理下落。
- 泥地、鹅卵石和草地整体铺设更密集的背景杂物，不在目标旁专门堆积。台阶只保留
  三级台阶本体；目标可放在下方台阶边缘，由上一级台阶自然形成遮挡。语义分割会
  估算可见比例，低于 50% 的候选图被拒绝。
- 相机距离、姿态和取景偏移随机化，目标在画面中有大有小，也允许偏离中心或靠
  近边缘。每张候选图会拒绝目标框内或底边的纯黑渲染伪影；渲染子帧为 32。

### 垃圾桶单独采集（data06_bin）

新增 `blue_bin`、`green_bin` 两个目标类；`red_bin` 只作为不带标签的干扰物。
类别文件保持最终六类顺序：

```text
watermelon
can
lunch_box
red_bag
blue_bin
green_bin
```

先生成五个独立 USD 采集场景（泥土、草地、鹅卵石、台阶、warehouse）：

```bash
conda run --no-capture-output -n rl_mani_lab \
  python scripts/build_bin_scenes.py
```

垃圾桶使用模型原始尺寸，底部贴地、只绕垂直轴随机旋转。green/blue 模型保持
导入姿态；red 模型额外做 180 度翻转，达到和其他垃圾桶一致的正放效果。每张图
有 1--3 个垃圾桶，但同一种颜色/类型不重复；至少一个为带标签的 blue/green，
red 只作干扰物。多个垃圾桶按相机视野的横向轴近似并排，中心间距足以保证
标签框零重叠，仅加入小幅前后错落，避免完整前后遮挡。相机固定在世界高度
0.5 m，`look_at` 的 Z 同样为 0.5 m，因此光轴平视、无俯仰；拍摄距离按 1--3
个垃圾桶分别约在 1.55--3.20、2.00--3.50、2.40--3.80 m 中随机（取景约束会
进一步推远），保证整组数据有近有远。

按用户要求每个环境 30 张、总计 150 张，输出到 `data/data06_bin`：

```bash
XDG_CACHE_HOME=/tmp/atec_isaac_cache \
XDG_CONFIG_HOME=/tmp/atec_isaac_config \
XDG_DATA_HOME=/tmp/atec_isaac_data \
conda run --no-capture-output -n rl_mani_lab \
  python scripts/generate_bin_yolo_data.py --num-images 150 --overwrite
```

单个 Isaac Sim 进程会依次打开五个场景并平均分配 150 张图。采集前请先确认
`nvidia-smi` 能看到可用 RTX GPU。

### 垃圾桶 + 旧物体混合采集（data07_bin&obj）

`scripts/build_bin_object_scenes.py` 生成五个独立的混合采集场景，输出在
`assets/scenes/bin_object/`。每个场景包含 blue/green/red 三个垃圾桶 prim，以及
旧四类物体的 watermelon、can、lunch_box、red_bag 模型池。red 垃圾桶仍翻转
180° 后正放，并且只作为不带标签的干扰物。

`scripts/generate_bin_object_yolo_data.py` 的布局分为两层：

- 远处：blue/green/red 三个垃圾桶每张图都出现，近似横向并排，带小幅深度错落；
- 近处：地面上放 1--3 个旧物体，物体类型不重复，姿态、尺度和贴地方式参考旧
  四类仿真采集逻辑。

相机仍在世界高度 0.5 m，`look_at` 的 Z 也为 0.5 m，保持平视。近处物体距离约
1.50--1.75 m，远处垃圾桶约 3.23--3.82 m。候选图要求 blue/green 垃圾桶和全部
旧物体都有有效实例框，red 垃圾桶也必须可见；标签框重叠超过 0.30 的候选会被
拒绝，避免完全前后遮挡。

生成五个环境各 20 张、总计 100 张，输出到 `data/data07_bin&obj`：

```bash
conda run --no-capture-output -n rl_mani_lab \
  python scripts/build_bin_object_scenes.py

XDG_CACHE_HOME=/tmp/atec_isaac_cache \
XDG_CONFIG_HOME=/tmp/atec_isaac_config \
XDG_DATA_HOME=/tmp/atec_isaac_data \
conda run --no-capture-output -n rl_mani_lab \
  python scripts/generate_bin_object_yolo_data.py \
    --output 'data/data07_bin&obj' --num-images 100 --overwrite --seed 7107
```

### 合并数据集 data08

`scripts/build_data08.py` 按顺序合并：

1. `data03` 五个仿真环境各自的前 60 对：共 300 对；
2. `data04` 按原编号取前 152 对；
3. `data06_bin` 全部 150 对；
4. `data07_bin&obj` 全部 100 对。

合计 702 对。输出统一重命名为 `data08_000000.png` / `data08_000000.txt` 到
`data08_000701.png` / `data08_000701.txt`，并写入六类 `classes.txt` 与逐张来源
`metadata.json`。源数据只复制、不移动、不修改：

```bash
python3 scripts/build_data08.py --overwrite
```
