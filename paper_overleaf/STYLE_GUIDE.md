# Alphabrain Platform — 视觉与配图风格指南

本文件固化 Alphabrain Platform Technical Report 的**配色**与**配图风格**，
供后续画系统总览图 / 架构图 / 能力网格图时保持统一。
风格参考 GR-3 Technical Report 的图示语言，配色替换为 AI² ROBOTICS（智平方）品牌色。

---

## 1. 配色（Color Palette）

### 1.1 主品牌色（正文已使用，定义于 `alphabrain.sty`）
| 角色 | 名称 | HEX | RGB | 用途 |
|---|---|---|---|---|
| 主色 | Brand Navy | `#143C5C` | 20, 60, 92 | 标题、链接、图中主块/主箭头 |
| 深色 | Deep Navy | `#0D2A40` | 13, 42, 64 | 横线、强调、深色描边 |
| 浅底 | Mist | `#E6EDF4` | 230, 237, 244 | 面板填充、表格底色 |
| 描边 | Mist Edge | `#9DBAD3` | 157, 186, 211 | 面板/盒子边框 |
| 正文 | Ink | `#1F2328` | 31, 35, 40 | 正文与图中正文字 |
| 次要 | Gray | `#6B7280` | 107, 114, 128 | 次要文字、占位、坐标轴 |

### 1.2 品牌辅助色（取自智平方 logo，用于图表分类/数据类型标签）
| 名称 | HEX | 用途建议 |
|---|---|---|
| Logo Teal | `#16A89B` | 第 2 类数据 / 第 2 序列 |
| Logo Green | `#5DBB46` | 第 3 类数据 / 正向指标 |
| Logo Lime | `#8BC53F` | 强调高亮（少量） |
| Logo Sky | `#1CA0DB` | 与主蓝区分的浅蓝标签 |
| Accent Amber | `#E8A33D` | 对比色（baseline / 警示，少量使用） |

> **分类配色顺序**（柱状/折线/散点）：
> `#143C5C → #16A89B → #5DBB46 → #1CA0DB → #E8A33D → #6B7280`
> 单色渐变（heatmap/进度）：`#E6EDF4 → #9DBAD3 → #143C5C → #0D2A40`

---

## 2. 配图风格（Figure Style，参考 GR-3）

### 2.1 通用原则
- **圆角面板**：所有图块用圆角矩形（圆角半径 ~6–8pt），不要直角硬边。
- **浅色填充 + 细描边**：面板填充 `Mist (#E7F0F7)`，描边 `Mist Edge (#A9CCE3)` 0.8–1pt。
- **留白充足**：面板之间留明显间距，整体干净通透，不要塞满。
- **彩色标签**：图块标题/数据类型标签用 `Brand Blue` 着色（如 "Vision–Language Data"）。
- **连接箭头**：流程箭头用 `Brand Blue`，圆头、适当加粗（1.5–2pt），可带浅色发光底。
- **中心主块（模型名 pill）**：圆角胶囊形，`Brand Blue → Deep Blue` 渐变填充，白色粗体模型名（如 "Alphabrain"）。
- **编号圆徽**：序列图右上角小圆徽，`Brand Blue` 实心圆 + 白色数字。
- **字体**：图内文字尽量与正文一致用 **Palatino Linotype**；若用矢量工具画，正文字 Palatino，标签可用无衬线（Fira Sans / Helvetica）。
- **截图边框**：示例照片/截图统一加 4–6pt 圆角，描边 `Mist Edge`。

### 2.2 三类典型图（对应 GR-3 Fig.1/2/3）
1. **系统总览图（Overview）**：顶部三类数据面板（VL 数据 / 机器人轨迹 / 人类轨迹），
   箭头汇聚到中心模型 pill，下方展开能力示例面板。标签全部 `Brand Blue`。
2. **能力网格图（Capabilities）**：等距图片网格，圆角 + 浅蓝描边，
   每格下方居中浅蓝指令文字；长程任务序列加蓝色编号圆徽。
3. **架构图（Model）**：圆角矩形模块（主模块 `Mist` 填充 + 蓝描边，
   子模块可用 logo green 浅色区分），token 序列用小圆角色块阵列，箭头 `Brand Blue`。

### 2.3 工具与导出
- **矢量优先**：图用 PDF/SVG 矢量导出（TikZ / Figma / Illustrator / PPT 导 PDF），不要位图截图当主图。
- **命名**：`figures/overview.pdf`、`figures/architecture.pdf`、`figures/capabilities.pdf`。
- **DPI**：必须用位图时 ≥ 300 DPI。
- **配色取值**：作图软件直接用本文 §1 的 HEX。

---

## 3. LaTeX/TikZ 复用样式
画 LaTeX 矢量图时 `\input{figures/figstyle.tex}`，已预置品牌色与节点样式：
`abpanel`（圆角面板）、`abpill`（模型胶囊）、`abbadge`（编号圆徽）、`abarrow`（箭头）。
示例见 `figures/figstyle.tex` 文件末尾注释。
