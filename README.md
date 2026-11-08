# local-tts-suite

> **把一段参考音频，变成能说任意话的专属音色。**
> 全离线 · 不联网 · 不上传 · 纯本地推理

一个 WorkBuddy / Claude Code 技能，封装了本地声音克隆全流程的 **21 个实战坑**。

⭐ **它解决什么问题**：训练一个专属音色，完整成本约 6 小时——
但真正花在"训练"上的只有 40 分钟，其余 5 小时 20 分钟全在踩坑。
换台电脑、换个模型版本，这些坑**一个都不会少**。

**封成技能之后：照着做 1 小时。**

---

## 它讲了什么

- [为什么需要它](#为什么需要它)
- [怎么用](#怎么用)
- [五步流水线](#五步流水线)
- [核心知识](#核心知识)
- [技术决策](#技术决策)
- [合规边界](#合规边界)
- [安装依赖](#安装依赖)
- [许可与致谢](#许可与致谢)

---

## 为什么需要它

克隆一个音色，技术上不复杂，但**坑密集且隐蔽**：

| 症状 | 真因 |
|---|---|
| `size mismatch 512 vs 732` | v2Pro 用了 512 词表的权重 |
| 报「ffmpeg error」 | ⚠️ 真因是 `list.txt` 字段顺序错 |
| 生成成功但音色是别人的 | ⚠️ 路径写错 ⇒ **静默回退**到官方默认权重 |
| 双击 `.bat` 没反应 | 文件是 LF 换行，Windows 只认 CRLF |
| 训练崩在保存那一刻 | 漏建 `logs_s2_v2Pro` 目录 |

⭐ **注意第 2、3 行**——它们**不报错**。
"能生成出声音"不等于"配置是对的"。这就是为什么本技能的验收步骤
**故意制造失败**，用反证法确认链路真的可信。

---

## 怎么用

### 第 0 步：装技能

把本目录放进技能目录：

```bash
# WorkBuddy / Claude Code
cp -r local-tts-suite ~/.workbuddy/skills/
# 或 Windows
xcopy /E /I local-tts-suite "%USERPROFILE%\.workbuddy\skills\local-tts-suite"
```

### 第 1 步：体检

```bash
python scripts/env_check.py --root <GPT-SoVITS 仓库根> \
       --py <仓库根>/runtime/python.exe
```

输出显卡/CUDA/底模/数据/换行的完整体检报告，告诉你**缺什么、怎么补**。

### 第 2 步：备料

```bash
python scripts/prepare_data.py --src <你的录音目录> --out <仓库根>/data/my_voice \
       --texts 台词.txt --ffmpeg <ffmpeg 路径>
```

切片 + 繁简统一 + 生成合规的 `list.txt`。

### 第 3 步：训练

```bash
python scripts/train.py --root <仓库根> --exp my_voice --stage s1 --epochs 15
python scripts/train.py --root <仓库根> --exp my_voice --stage s2 --epochs 15
```

⭐ **训练前会自动预检**，把低级错拦在几小时的训练之外。

### 第 4 步：验收

```bash
python scripts/verify.py --root <仓库根> \
       --gpt <GPT_weights_v2Pro/my_voice-e15.ckpt> \
       --sovits <SoVITS_weights_v2Pro/my_voice_e15_s150.pth> \
       --ref <参考音频.wav> --ref-text "参考音频对应的原文"
```

正向生成 + **故意制造失败**的反向验证。

### 第 5 步：出片

```bash
python scripts/make_srt.py --outputs-dir <outputs> --latest 3 \
       --texts 台词.txt -o 旁白.srt
```

---

## 五步流水线

```
① 环境体检 env_check.py     → 显卡/显存/CUDA/底模/数据/换行 〔只读〕
② 备料     prepare_data.py  → 切片 + 繁简统一 + list.txt
③ 训练     train.py          → ⭐预检拦截 → s1/s2 两阶段
④ 验收     verify.py         → 正向生成 + ⭐故意制造失败
⑤ 出片     make_srt.py       → 批量配音 + SRT 字幕
```

⭐ **两个反直觉的设计**：

- **③ 训练前先预检** —— 训练一次几小时，不能因低级错白跑
- **④ 验收含故意制造失败** —— 静默的错误必须用反证法抓

---

## 核心知识

### 两条路线怎么选

| | **GPT-SoVITS v2Pro** | **IndexTTS-2.5** |
|---|---|---|
| 适合 | ⭐ **训自己的音色** | 零样本克隆、情感解耦 |
| 能否微调 | ✅ 能（s1+s2） | ❌ 官方没有微调脚本 |
| 协议 | **MIT** | bilibili Model Use License |
| 上手成本 | 高（有坑） | 低（开箱即用） |

### 关键参数速查

| 项 | 值 | 说明 |
|---|---|---|
| **GPT 底模** | `s1v3.ckpt`（**732 词表**） | v2Pro 专用 |
| **SoVITS 底模** | `s2G488k.pth` | |
| **推理配置** | 必须包在 `"custom"` 键下 | 否则静默回退 |
| **输入键名** | `text_lang` / `prompt_lang` | 不是 `language` |
| **参考音频** | 3~10 秒（**切 8 秒**最稳） | 源码硬限制 |
| **list.txt** | `{相对路径}\|{输出名}\|{语言}\|{文本}` | ⭐ 第 0 位是路径 |
| **训练目录** | `logs_s2_v2Pro` 必先建 | 漏建崩在保存时 |
| **s1 配方** | 15 epoch，loss 2.55e+3 → 1.56e+3 | 已验证 |

### Windows 专属

```python
# Git Bash 的 PATH 用 ':'，Windows Python 要 ';'
os.environ['PATH'] = runtime_dir + ';' + p.replace(':', ';')

# sys.path 要同时加仓库根与 GPT_SoVITS/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'GPT_SoVITS'))
```

---

## 技术决策

**为什么 IndexTTS 没有微调脚本？**（架构原因，不是偷懒）
它的说话人信息来自参考音频的 **speaker embedding**，不在主干权重里
⇒ 换音色天然不需要训练 ⇒ 官方索性没出训练脚本。

**为什么训练前要预检？**
训练一次几小时。#13 坑（漏建目录）会让它在第 4 个 epoch 保存时崩——
**跑了两小时才崩**，是最不值当的失败。

**为什么验收要故意制造失败？**
因为**很多错是静默的**。#14 坑的表现是"配置错了但不报错"，
只有用反证法（故意用错路径加载，看能否发现回退）才能确认检查真的有效。

---

## 合规边界

| 项 | 结论 |
|---|---|
| **GPT-SoVITS** | **MIT**，无商用限制 |
| **IndexTTS-2.5** | 4.1(c) 禁侵犯隐私（克隆他人声音须授权）｜2.2 大厂须单独授权 |
| **克隆自己的声音** | ✅ 无问题 |
| **克隆他人声音** | 🚨 声音权类比肖像权保护，**须书面授权** |
| **发布/传播/商用** | 🚨 法律 + 模型协议 + 平台三层同时生效 |

⭐ **本仓库只含方法与脚本，不含任何模型权重、参考音频或训练语料。**
请用**你自己的**素材训练你自己的音色。

---

## 安装依赖

**上游项目**（本仓库不附带，需自行 clone）

| 项目 | 用途 | 协议 |
|---|---|---|
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 训练专属音色 | MIT |
| [IndexTTS-2.5](https://github.com/index-tts/index-tts) | 零样本克隆 | bilibili Model Use License |

**环境**：Python 3.10+ · PyTorch 2.7+（cu128）· NVIDIA GPU 8GB+ · ffmpeg

⭐ **黑卡（算力 12.0 / sm_120）必须用 cu128 构建的 torch**，否则推理直接报错。

---

## 许可与致谢

**本仓库**（方法、脚本、坑位清单）：**MIT**。

**上游项目**版权归各自作者，本仓库只是把它们的使用经验整理成可执行的知识。

⭐ **坑位清单是本仓库最有价值的部分**——它来自真实踩坑，不是从文档抄的。
如果你也踩出了新坑，欢迎提 issue 补充。

---

**一句话**：踩过的坑不该只留在自己的机器上。
