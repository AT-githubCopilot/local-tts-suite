---
name: local-tts-suite
description: 本地声音克隆全套流水线——把一段参考音频训成专属音色，封装 21 个实战坑（词表版本、list.txt 格式、custom 配置键、路径分隔符、CRLF 换行、静默回退陷阱等），并提供推理验收与字幕生成。当用户说「声音克隆」「音色微调」「TTS」「配音」「训一个我的音色」「克隆声音」「GPT-SoVITS」「IndexTTS」「怎么把这段声音变成能说任意话的工具」「换机器重装声音克隆」，或任何「本地部署 + 训练专属音色 + 批量配音」的场景时使用。
agent_created: true
version: 1.0.0
---

# 本地声音克隆全套流水线

**一句话结论：一条五步流水线，把「一段参考音频」变成「能说任意话的专属音色」，且 21 个坑全部预埋。**

---

## 为什么要有这个技能

⚠️ **一次踩坑，终身免疫。**

训练一个专属音色的完整成本约 **6 小时**——但真正花在"训练"上的只有 40 分钟，
其余 5 小时 20 分钟全在踩坑：

- 词表版本不对 → `size mismatch 512 vs 732`
- `list.txt` 字段顺序错 → 报「ffmpeg error」，真因完全无关
- 推理配置没包在 `"custom"` 下 → **静默**加载官方默认权重
- `.bat` 文件是 LF 换行 → Windows cmd 双击**静默失败**
- 并行下载和并行装互相绊倒 → 目录嵌套一层

⇒ 这些坑**不会因为你换台电脑就消失**。封成技能，它们就变成可携带的免疫。

⭐ **换机器重装**：照着做 1 小时，不看技能 6 小时。

---

## 五步流水线

```
① 环境探测 env_check.py     → 显卡/显存/CUDA/底模/路径合规性
② 备料 prepare_data.py      → 切片 + 转写 + 繁简转换 + list.txt 生成
③ 训练 train.py              → ⭐预检拦截 → s1/s2 两阶段 + loss 监控
④ 推理验收 verify.py         → 正向生成 + ⭐故意制造失败
⑤ 出片 make_srt.py          → 批量配音 + SRT 字幕
```

⭐ **两个反直觉的设计，是本技能真正的价值**：

- **③ 训练前先预检** —— 训练一次几小时，不能因为低级错白跑
- **④ 验收含故意制造失败** —— "能生成出声音"不等于"配置是对的"，很多错是**静默**的

---

## 两条路线：先选对，别走错

| | **GPT-SoVITS v2Pro** | **IndexTTS-2.5** |
|---|---|---|
| **适合** | ⭐ **训自己的音色** | 零样本克隆、情感解耦 |
| **能否微调** | ✅ 能（s1+s2 两阶段） | ❌ **官方没有微调脚本** |
| **协议** | **MIT**（无商用限制） | `bilibili Model Use License Agreement` |
| **上手成本** | 高（有坑） | 低（开箱即用） |

⭐ **为什么 IndexTTS 没有微调脚本（架构原因，不是偷懒）**：
它的说话人信息来自参考音频的 **speaker embedding**，不在主干权重里
⇒ 换音色天然不需要训练 ⇒ 官方索性没出训练脚本。

---

## 三条硬规矩

### ① 词表版本必须对上
**v2Pro 必须用 732 词表的 `s1v3.ckpt`**，不是 512 词表权重。

### ② `list.txt` 字段顺序是 `{相对路径}|{输出名}|{语言}|{文本}`
⚠️ **第 0 位是路径**（相对 list.txt 所在目录，如 `wavs/xxx.wav`）。
照 `tools/asr/funasr_asr.py` 实锤的格式。

### ③ 推理配置必须包在 `"custom"` 键下
```python
cfg = {"custom": {"version": "v2Pro", "device": "cuda:0", ...}}
```

⭐ **这个坑比想象中危险**——GPT-SoVITS 源码 `TTS.py` 有两处静默回退：

```python
# 第 318 行：custom 键不存在 → 回退 v2 默认配置
self.configs = configs_.get("custom", configs_["v2"])

# 第 344 行：路径不存在 → 回退官方默认权重
if (self.t2s_weights_path in [None, ""]) or (not os.path.exists(self.t2s_weights_path)):
    self.t2s_weights_path = self.default_configs[version]["t2s_weights_path"]
    print(f"fall back to default t2s_weights_path: {self.t2s_weights_path}")
```

⇒ **配置错了不报错，只是音色悄悄变成别人的。**
`verify.py` 的反向用例就是专门抓这个的。

---

## 五个最容易踩的坑（完整 21 条见 `references/坑位清单.md`）

| # | 症状 | 真因 | 对策 |
|---|---|---|---|
| ⭐ 1 | `size mismatch 512 vs 732` | v2Pro 用了 512 词表权重 | 换 `s1v3.ckpt` |
| ⭐ 2 | 报「ffmpeg error」 | ⚠️ **真因是 `list.txt` 字段顺序错** | 照 `{路径}\|{输出名}\|{语言}\|{文本}` |
| ⭐ 3 | 加载了别人的音色 | 路径错 ⇒ **静默回退**到默认权重 | `verify.py` 反证 |
| ⭐ 4 | `.bat` 双击**静默失败** | LF 换行，cmd 只认 CRLF | 转 CRLF |
| ⭐ 5 | 报错指向 A，真因在 B | 「构建失败」要看日志尾部 | 看 stderr 末行 |

---

## 路径与换行（Windows 专属）

```python
# ⭐ Git Bash 的 PATH 用 ':'，Windows Python 要 ';'
os.environ['PATH'] = runtime_dir + ';' + p.replace(':', ';')

# ⭐ sys.path 要同时加仓库根和 GPT_SoVITS/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'GPT_SoVITS'))

# ⭐ .bat 必须是 CRLF
# Python: open(path,'rb').read().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
```

---

## 参考音频规范

| 项 | 要求 | 原因 |
|---|---|---|
| **时长** | **3–10 秒**（切 8 秒最稳） | GPT-SoVITS 推理硬限制（源码 `>160000 或 <48000`） |
| **内容** | 干净人声，无音乐无混响 | 噪声会被一起学进去 |
| **条数** | 2–3 条不同录音 | 只录一条容易过拟合到那一句的语调 |
| **总量** | ≥ 60 秒有效 | 36 切片 / 130 秒是已验证的起点 |

⭐ **多样本要同一个人、同一种语气** —— 两种音色混在一起会训出"平均音"。

---

## 训练配方（已验证）

| 阶段 | 参数 | 结果 |
|---|---|---|
| **s1**（GPT） | 15 epoch | loss **2.55e+3 → 1.56e+3** |
| **s2**（SoVITS） | 15 epoch（step 150） | 收敛正常 |
| **产出** | `GPT_weights_v2Pro/xxx-e{5,10,15}.ckpt`<br>`SoVITS_weights_v2Pro/xxx_e{5}_s50.pth` 等 | 每阶段留 3 个点便于对比 |
| **推理用** | 取**最后一个**（e15 + s150） | 前面的是对照 |

⭐ **训前必做**：`mkdir -p logs_s2_v2Pro`（官方 `webui.py` 会建，直接调脚本会绕过）。

---

## 推理接口要点

```python
res = tts.run({
    "ref_audio_path": r"...8s.wav",
    "prompt_text": "参考音频对应的原文",
    "prompt_lang": "zh",       # ⭐ 不是 language
    "text": "要念的话",
    "text_lang": "zh",         # ⭐ 不是 language
    "speed_factor": 1.0,
    "top_k": 15, "top_p": 0.8, "temperature": 0.7,
    "text_split_method": "cut0",
    "seed": 1234,
})
# ⭐ run() 返回 generator，要迭代且元素是 (sr, audio) 元组
```

⭐ **语气三件套**（`top_k`/`top_p`/`temperature`）是"语气随机性"——
值越大念法越发散、越有情绪起伏；越小越平越稳。
把它们包装成"沉静/温和/昂扬"三档，比暴露三个滑块好用。

---

## 脚本

| 脚本 | 干什么 |
|---|---|
| `scripts/env_check.py` | ① 探测显卡/显存/CUDA/底模/数据/换行，出可执行性报告（只读） |
| `scripts/prepare_data.py` | ② 切片 + 繁简统一 + 生成合规的 `list.txt` |
| `scripts/train.py` | ③ 预检拦截 + 两阶段训练启动 |
| `scripts/verify.py` | ④ 验收（正向 + **故意制造失败**） |
| `scripts/make_srt.py` | ⑤ 批量配音 + SRT 字幕（时间轴按实际音频对齐） |

---

## 合规边界

| 项 | 结论 |
|---|---|
| **GPT-SoVITS** | **MIT**，无商用限制 |
| **IndexTTS-2.5** | 4.1(c) 禁侵犯隐私（克隆他人声音须授权）｜2.2 大厂须单独授权 |
| **自己的声音** | ✅ 无问题 |
| **克隆他人** | 🚨 声音权类比肖像权保护 + 模型协议，**须书面授权** |

⚠️ **一条线不能跨**：**发布 / 传播 / 商用** —— 三层法律同时生效。

---

## 附 · 本技能的原理框图

⭐ 随封装一并产出（技能封装完成 = 必须同时有原理框图）：

| 文件 | 内容 |
|---|---|
| `references/架构.md` | mermaid 源码 + 结构说明 + 坑位表 |
| `references/架构.svg` | ⭐ **双击即看**的成品图 |

⭐ **调用方式**：「看看 local-tts-suite 的原理」→ 读 `references/架构.md` 或直接 `present_files` 那张 svg。
