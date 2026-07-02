# Alphabrain Platform — Technical Report (LaTeX 模板)

仿 GR-3 Technical Report 排版风格，使用 AI² ROBOTICS（智平方）品牌配色。

## 文件
| 文件 | 作用 |
|---|---|
| `main.tex` | 正文（标题、Abstract、各章节、表格、参考文献） |
| `alphabrain.sty` | 样式包：品牌配色、章节标题、Abstract 盒子、页眉页脚、标题宏 |
| `references.bib` | 参考文献示例 |
| `STYLE_GUIDE.md` | **视觉与配图风格指南**（配色 + GR-3 风格画图规范） |
| `figures/figstyle.tex` | 可复用的品牌 TikZ 样式（圆角面板/模型胶囊/编号徽/箭头） |
| `figures/logo_aifang.png` | 公司 logo（已放好） |
| `fonts/*.ttf` | 真正的 Palatino Linotype 字体文件（pala/palab/palai/palabi） |
| `main.pdf` | 示例编译产物 |

## 在 Overleaf 使用
1. 把整个 `alphabrain-tech-report` 文件夹打包成 zip（**务必包含 `fonts/` 文件夹**）。
2. Overleaf → New Project → **Upload Project** → 选 zip。
3. 编译器设为 **XeLaTeX**（Menu → Compiler），主文件为 `main.tex`。
4. 直接编译即可。

> ⚠️ 必须用 **XeLaTeX**（不是 pdfLaTeX）：正文用的是真实 Palatino Linotype 字体文件，
> 通过 fontspec 从 `fonts/` 目录加载。Palatino Linotype 为微软专有字体，随项目内部使用；
> 对外公开发布请留意字体授权。

## 品牌配色（定义在 alphabrain.sty）
| 名称 | HEX | 用途 |
|---|---|---|
| `abTeal` | `#143C5C` | 主品牌深蓝（章节标题、内部链接） |
| `abBlue` | `#143C5C` | 引用、URL 链接 |
| `abDeep` | `#0D2A40` | 最深蓝（横线、强调） |
| `abMist` | `#E6EDF4` | 表格底色（Abstract 盒子现为白底） |
| `abMistEdge` | `#9DBAD3` | 盒子/面板边框 |
| `abInk` `abGray` | 正文 / 次要文字 |

改色只需编辑 `alphabrain.sty` 顶部 `BRAND PALETTE` 区块。

## 常用宏（main.tex 中调用）
- `\abheader{figures/logo_aifang.png}` — 左上角 logo + 横线
- `\abreporttitle{...}` — 横线包裹的居中标题
- `\aborg{...}` / `\abnote{...}` — 机构名 / 居中说明行（GitHub、官网、作者列表）
- `ababstract` 环境 + `\abstractheading` — 圆角品牌色 Abstract 盒子
- `\abmeta{Date:}{...}` — 元信息行
- `\abrole{Core Contributors}{...}` — 贡献者列表

## 替换内容清单
- [ ] 把各章节占位文字换成真实内容
- [ ] 用真实图替换 `figures/overview.pdf`（取消 `\includegraphics` 注释、删占位框）
- [ ] 更新 `references.bib` 并在正文 `\citep{...}` 引用
- [ ] 填写 Contributions and Acknowledgements
- [ ] 核对 Date / Correspondence / Project Page / GitHub / 官网链接
