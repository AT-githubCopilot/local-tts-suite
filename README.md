# local-tts-suite

> **把一段参考音频，变成能说任意话的专属音色。**
> 全离线 · 不联网 · 不上传 · 纯本地推理

一个 WorkBuddy / Claude Code 技能，封装了本地声音克隆全流程的 **21 个实战坑**。

---

## 它讲了什么

- [缘起](#缘起)
- [用法](#用法)
- [流水线](#流水线)
- [知识](#知识)
- [决断](#决断)
- [边界](#边界)
- [依赖](#依赖)
- [致谢](#致谢)

---

## 缘起

---代价 时间的真相---

训练一个专属音色，完整成本约 6 小时。

但真正花在"训练"上的只有 40 分钟。
其余 5 小时 20 分钟，全在踩坑。

---坑密 隐蔽是常态---

坑不只多，而且**不报警**。

| 症状 | 真因 |
|---|---|
| `size mismatch 512 vs 732` | v2Pro 用了 512 词表的权重 |
| 报「ffmpeg error」 | ⚠️ 真因是 `list.txt` 字段顺序错 |
| 生成成功但音色是别人的 | ⚠️ 路径写错 ⇒ **静默回退**到官方默认权重 |
| 双击 `.bat` 没反应 | 文件是 LF 换行，Windows 只认 CRLF |
| 训练崩在保存那一刻 | 漏建 `logs_s2_v2Pro` 目录 |

第三行最值得停一停——它**不报错**。
声音出来了，只是变成了别人的。

---免疫 一次踩坑---

换台电脑、换个模型版本，这些坑**一个都不会少**。

所以它们被写成了文件，而不是留在脑子里。
**封成技能之后：照着做 1 小时。**

---

## 用法

### 装技能

```bash
# WorkBuddy / Claude Code（Windows）
xcopy /E /I local-tts-suite "%USERPROFILE%\.workbuddy\skills\local-tts-suite"

# macOS / Linux
cp -r local-tts-suite ~/.workbuddy/skills/
```

### 五步

```bash
# ① 体检 —— 显卡/底模/数据/换行，一次说清缺什么
python scripts/env_check.py --root <GPT-SoVITS 仓库根> \
       --py <仓库根>/runtime/python.exe

# ② 备料 —— 切片 + 繁简统一 + 生成合规的 list.txt
python scripts/prepare_data.py --src <你的录音目录> --out <仓库根>/data/my_voice \
       --texts 台词.txt --ffmpeg <ffmpeg 路径>

# ③ 训练 —— ⭐ 先预检，再开跑
python scripts/train.py --root <仓库根> --exp my_voice --stage s1 --epochs 15
python scripts/train.py --root <仓库根> --exp my_voice --stage s2 --epochs 15

# ④ 验收 —— 正向生成 + ⭐故意制造失败
python scripts/verify.py --root <仓库根> \
       --gpt <GPT_weights_v2Pro/my_voice-e15.ckpt> \
       --sovits <SoVITS_weights_v2Pro/my_voice_e15_s150.pth> \
       --ref <参考音频.wav> --ref-text "参考音频对应的原文"

# ⑤ 出片 —— 批量配音 + SRT 字幕
python scripts/make_srt.py --outputs-dir <outputs> --latest 3 \
       --texts 台词.txt -o 旁白.srt
```

---

## 流水线

```
① 环境体检 env_check.py     → 显卡/显存/CUDA/底模/路径合规性
② 备料     prepare_data.py  → 切片 + 繁简统一 + list.txt
③ 训练     train.py          → ⭐预检拦截 → s1/s2 两阶段
④ 验收     verify.py         → 正向生成 + ⭐故意制造失败
⑤ 出片     make_srt.py       → 批量配音 + SRT 字幕
```

---预检 训练前先拦一道---

训练一次要几小时。
漏建一个目录，可能让它在第 4 个 epoch 保存时崩——**跑了两小时才崩**，是最不值当的失败。

⇒ `train.py` 会在开跑前把已知低级错全拦下来。

---反证 不制造失败就不知道对不对---

"能生成出声音"不等于"配置是对的"。
很多错是静默的。

⇒ `verify.py` 的反向用例会**故意用错路径加载**，看模型会不会偷偷回退到官方默认权重。
能发现，才算验收通过。

---

## 知识

### 选路线

| | **GPT-SoVITS v2Pro** | **IndexTTS-2.5** |
|---|---|---|
| 适合 | ⭐ **训自己的音色** | 零样本克隆、情感解耦 |
| 能否微调 | ✅ 能（s1+s2） | ❌ 官方没有微调脚本 |
| 协议 | **MIT** | bilibili Model Use License |
| 上手成本 | 高（有坑） | 低（开箱即用） |

### 参数速查

| 项 | 值 | 说明 |
|---|---|---|
| **GPT 底模** | `s1v3.ckpt`（**732 词表**） | v2Pro 专用 |
| **SoVITS 底模** | `s2G488k.pth` | |
| **推理配置** | 必须包在 `"custom"` 键下 | 否则静默回退 |
| **输入键名** | `text_lang` / `prompt_lang` | 不是 `language` |
| **参考音频** | 3–10 秒（**切 8 秒**最稳） | 源码硬限制 |
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

### 参考音频规范

| 项 | 要求 | 原因 |
|---|---|---|
| **时长** | **3–10 秒**（切 8 秒最稳） | 推理硬限制 |
| **内容** | 干净人声，无音乐无混响 | 噪声会被一起学进去 |
| **条数** | 2–3 条不同录音 | 只录一条容易过拟合 |
| **总量** | ≥ 60 秒有效 | 36 切片 / 130 秒是已验证起点 |

⭐ **多样本要同一个人、同一种语气** —— 两种音色混在一起会训出"平均音"。

---

## 决断

---为何 IndexTTS 没有微调脚本---

不是偷懒，是架构决定的。

它的说话人信息来自参考音频的 **speaker embedding**，不在主干权重里。
换音色天然不需要训练 ⇒ 官方索性没出训练脚本。

---为何预检要放在训练前---

训练一次几小时。第 4 个 epoch 才崩，是最贵的教训。

---为何验收要反着做---

配置错了但声音能出来——这才是最危险的失败。
只有用反证法（故意用错路径，看能否发现回退），才知道检查本身有没有效。

---

## 边界

| 项 | 结论 |
|---|---|
| **GPT-SoVITS** | **MIT**，无商用限制 |
| **IndexTTS-2.5** | 4.1(c) 禁侵犯隐私（克隆他人声音须授权）｜2.2 大厂须单独授权 |
| **克隆自己的声音** | ✅ 无问题 |
| **克隆他人** | 🚨 声音权类比肖像权保护，**须书面授权** |
| **发布 / 传播 / 商用** | 🚨 法律 + 模型协议 + 平台三层同时生效 |

⭐ **本仓库只含方法与脚本，不含任何模型权重、参考音频或训练语料。**
请用**你自己的**素材训练**你自己的**音色。

---

## 依赖

| 项目 | 用途 | 协议 |
|---|---|---|
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 训练专属音色 | MIT |
| [IndexTTS-2.5](https://github.com/index-tts/index-tts) | 零样本克隆 | bilibili Model Use License |

**环境**：Python 3.10+ · PyTorch 2.7+（cu128）· NVIDIA GPU 8GB+ · ffmpeg

⭐ **黑卡（算力 12.0 / sm_120）必须用 cu128 构建的 torch**，否则推理直接报错。

---

## 致谢

**本仓库**（方法、脚本、坑位清单）：**MIT**。

**上游项目**版权归各自作者，本仓库只是把它们的使用经验整理成可执行的知识。

⭐ **坑位清单是本仓库最有价值的部分**——它来自真实踩坑，不是从文档抄的。
如果你也踩出了新坑，欢迎提 issue 补充。

---

**一句话**：踩过的坑不该只留在自己的机器上。
