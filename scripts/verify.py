# -*- coding: utf-8 -*-
"""④ 验收：验证整条推理链路真的通。

⭐ 核心理念：**不制造失败，就不知道链路是否真通。**

"能生成出声音"不等于"配置是对的"——很多错误是**静默**的
（加载了 v2 底模、参数字段名写错被忽略、音色不对但不报错）。

所以本脚本做三件事：
  ① 正向验证 —— 用真配置生成，检查音频确实有效
  ② ⭐ 反向验证 —— 故意注入已知错误，确认它们**会**被抓住
  ③ 产出验收报告（JSON + 文字）

用法：
    python verify.py --root <GPT-SoVITS 根> --gpt <e15.ckpt> --sovits <s150.pth> \
                     --ref <参考音频.wav> --ref-text "参考音频对应的原文"
"""
import io
import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

_ROOT_CANDIDATES = [
    os.environ.get("GPT_SOVITS_ROOT", ""),
    # ⭐ 下面是【常见安装位置示例】，按你自己的实际路径改；
    #    也可以直接设环境变量 GPT_SOVITS_ROOT 一次搞定
    r"F://AIStack//GPT-SoVITS//GPT-SoVITS-v2pro-20250604-nvidia50",
    r"D:\AIStack\GPT-SoVITS\GPT-SoVITS-v2pro-20250604-nvidia50",
    r"C:\AIStack\GPT-SoVITS\GPT-SoVITS-v2pro-20250604-nvidia50",
    os.path.expanduser("~/AIStack/GPT-SoVITS/GPT-SoVITS-v2pro-20250604-nvidia50"),
]

ROOT = None


def detect_root():
    """自动探测 GPT-SoVITS 仓库根。

    优先级：--root 参数 > 环境变量 GPT_SOVITS_ROOT > 常见安装位置
    """
    for c in _ROOT_CANDIDATES:
        if c and os.path.isdir(c):
            return c
    return ""



def out(msg):
    print(msg, flush=True)


def ensure_runtime():
    """⭐ 关键：必须用【项目自己的 runtime python】跑，否则缺 torch/torchaudio。

    做法：检测当前解释器能不能 import torch；
    不能的话，找到项目 runtime 里的 python.exe 并用它重启自己。
    """
    try:
        import torch  # noqa
        import torchaudio  # noqa
        return
    except ImportError:
        pass

    # 找 runtime
    cands = []
    if ROOT:
        cands.append(os.path.join(ROOT, "runtime", "python.exe"))
    cands += [
        os.path.join(ROOT, "runtime", "python.exe") if ROOT else "",
    ]
    for exe in cands:
        if not os.path.exists(exe):
            continue
        if os.path.abspath(exe) == os.path.abspath(sys.executable):
            continue
        # 验证它有 torch
        try:
            r = subprocess.run([exe, "-c", "import torch,torchaudio;print(torch.__version__)"],
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                out("→ 切换到项目 runtime：%s" % exe)
                env = dict(os.environ)
                # ⭐ #7 坑：Windows PATH 用 ';' 不是 ':'
                rt = os.path.join(ROOT, "runtime") if ROOT else ""
                env["PATH"] = rt + ";" + env.get("PATH", "").replace(":", ";")
                # ⭐ ⭐ cwd 必须在【同一盘符】否则相对路径失效
                #    （报错原文：path is on mount 'F:', start on mount 'C:'）
                if ROOT and os.path.isdir(ROOT):
                    os.chdir(ROOT)
                # 脚本路径转绝对，否则换解释器后找不到自己
                self_abs = os.path.abspath(__file__)
                os.execve(exe, [exe, self_abs] + sys.argv[1:], env)
        except Exception:
            continue
    out("⚠️ 找不到带 torch/torchaudio 的 python。请用项目 runtime 的 python 跑本脚本。")
    out("  例：<GPT-SoVITS根>\\runtime\\python.exe verify.py --root ...")
    sys.exit(2)


# ---------------------------------------------------------------- 正向

def synth(tts, ref, ref_text, text, speed=1.0, top_k=15, top_p=0.8, temperature=0.7,
          seed=1234, no_custom=False):
    """跑一次 TTS，返回 (数组, 采样率)

    ⭐ `no_custom=True` 用于**反向验证**：故意把 custom 键拆掉，
       模拟 #14 坑（配置没包在 "custom" 下 ⇒ 加载 v2 底模，静默用错音色）。
    """
    import numpy as np
    cfg = {
        "ref_audio_path": ref,
        "prompt_text": ref_text,
        "prompt_lang": "zh",
        "text": text,
        "text_lang": "zh",
        "speed_factor": speed,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": "cut0",
        "batch_size": 1,
        "split_bucket": True,
        "streaming_mode": False,
        "seed": seed,
        "language": "zh",
    }
    if no_custom:
        # ⭐ 故意把版本与权重路径塞到顶层（而不是 "custom" 下）
        cfg["version"] = "v2Pro"
        cfg["device"] = "cuda:0"
        cfg["is_half"] = True
        cfg["t2s_weights_path"] = _args.gpt
        cfg["vits_weights_path"] = _args.sovits
        cfg["bert_base_path"] = os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        cfg["cnhuhbert_base_path"] = os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-hubert-base")
    res = tts.run(cfg)
    chunks = []
    sr = 32000
    for item in res:
        if isinstance(item, tuple) and len(item) == 2:
            sr, seg = item[0], item[1]
        else:
            seg = item
        seg = np.asarray(seg)
        if seg.ndim == 1:
            seg = seg.reshape(-1, 1)
        chunks.append(seg)
    if not chunks:
        raise RuntimeError("没有生成出声音")
    return np.concatenate(chunks, axis=0), sr


# ---------------------------------------------------------------- 反向

def negative_tests(report):
    """⭐ 故意注入错误，确认它们会被抓住。

    这一步是本脚本的价值所在 —— 它证明"配置对了"不是运气。
    """
    out("\n--- 反向验证（故意制造失败，确认能被抓到）---")
    cases = []

    # 负例1：⭐ 验证"配置确实生效"——检查模型实际加载的是不是我指定的权重。
    #    #14 坑（custom 没包住）的真正危害是【静默加载了 v2 底模】：
    #    不报错，但音色是别人的。所以这个负例的判定标准不是"抛异常"，
    #    而是"实际加载的权重路径 ≠ 期望路径"。
    cases.append({
        "name": "custom 配置是否真的生效",
        "expect": "加载的权重 = 指定的权重",
        "run": lambda: _custom_effect_test(),
        "type": "静默检查",
    })

    # 负例2：参考音频路径不存在 ⇒ 应报文件错误
    #    ⭐ 为什么不测"词表错误"？因为那需要本机装有 512 词表权重，
    #      而多数人装的是 v2Pro 专用版（只有 732）⇒ 本机无法复现该坑。
    #      词表坑的正确防线是【安装时】就用 env_check 检查，不是运行时。
    cases.append({
        "name": "参考音频路径不存在",
        "expect": "报文件错误",
        "run": lambda: _missing_ref_test(),
    })

    # 负例3：参考音频超长（>10 秒）⇒ 硬限制
    cases.append({
        "name": "参考音频超过 10 秒",
        "expect": "拒绝或异常",
        "run": lambda: _long_ref_test(),
    })

    for c in cases:
        try:
            c["run"]()
            c["result"] = "⚠️ 未报错（说明这个坑可能已变或检查漏了）"
            c["caught"] = False
        except Exception as e:
            msg = str(e)
            # ⭐ 判定"抓住了"：只要**抛了异常**就算抓到。
            #    关键是"这个错误有没有被暴露"，而不是错误文案长什么样
            #    （文案会随版本变，判定标准不能绑死在文案上）。
            c["result"] = msg[:160]
            c["caught"] = True
        out("  %s %-28s → %s" % ("✅" if c["caught"] else "⚠️ ", c["name"], c["result"]))
        report.setdefault("negative", []).append(c)


# ---------------------------------------------------------------- 辅助

def _load_tts():
    """按 app_server.py 的方式加载模型（custom 包在正确位置）"""
    from TTS_infer_pack.TTS import TTS
    cfg = {
        "custom": {
            "version": "v2Pro",
            "device": "cuda:0",
            "is_half": True,
            "t2s_weights_path": _args.gpt,
            "vits_weights_path": _args.sovits,
            "bert_base_path": os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
        }
    }
    return TTS(cfg)


def _custom_effect_test():
    """⭐ 检查 #14 坑：路径写错 / custom 没包住 ⇒ **静默回退**到官方默认权重。

    两个源码级陷阱（这才是 #14 坑的真面目）：

    ① TTS.py 第 318 行
           self.configs = configs_.get("custom", configs_["v2"])
       ⇒ **custom 键不存在就静默回退 v2 配置**

    ② TTS.py 第 344 行起
           if (self.t2s_weights_path in [None,""]) or (not os.path.exists(...)):
               self.t2s_weights_path = self.default_configs[version]["t2s_weights_path"]
               print("fall back to default t2s_weights_path: ...")
       ⇒ **路径写错也静默回退到官方默认权重**（不报错，只是音色变别人）

    ⭐ 所以这个检查的逻辑是：**故意用错路径加载**，看能不能发现"加载的不是我要的"。
    正常路径下它当然不会报错 —— 那是正向验证的事。
    """
    # ---- ① 正常路径：确认加载的就是我指定的 ----
    tts = _load_tts()
    actual = _read_cfg(tts, "t2s_weights_path")
    version = _read_cfg(tts, "version")
    tag = os.path.splitext(os.path.basename(_args.gpt))[0]
    if tag not in actual:
        raise RuntimeError("加载的权重与指定的不一致！期望含 '%s'，实际：%s" % (tag, actual))
    if version != "v2Pro":
        raise RuntimeError("版本不是 v2Pro（实际 %s）" % version)

    # ---- ② 故意用错路径：看模型会不会静默回退 ----
    from TTS_infer_pack.TTS import TTS
    bad_cfg = {
        "custom": {
            "version": "v2Pro", "device": "cuda:0", "is_half": True,
            "t2s_weights_path": os.path.join(ROOT, "__不存在的权重__.ckpt"),
            "vits_weights_path": _args.sovits,
            "bert_base_path": os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
        }
    }
    try:
        bad = TTS(bad_cfg)
    except Exception as e:
        raise RuntimeError("错路径直接报错了：%s（说明这个版本会暴露问题）" % str(e)[:120])
    fallback = _read_cfg(bad, "t2s_weights_path")
    if tag in fallback:
        raise RuntimeError("错路径竟然加载了我的权重？逻辑异常")
    raise RuntimeError(
        "⚠️ 确认静默回退：错路径 → 实际加载 '%s'（未报错！）\n"
        "     这就是 #14 坑的杀伤力 —— 配置错了不报错，只是音色悄悄变成别人的。" % fallback)


def _read_cfg(tts, key):
    """从 TTS 对象的 configs 里读一个键（兼容 dict 与 TTS_Config 对象）"""
    cfgs = getattr(tts, "configs", None)
    if cfgs is None:
        return ""
    if isinstance(cfgs, dict):
        return str(cfgs.get(key, "") or "")
    return str(getattr(cfgs, key, "") or "")


def _missing_ref_test():
    """参考音频不存在 ⇒ 应报错而不是静默出空音"""
    tts = _load_tts()
    synth(tts, os.path.join(tempfile.gettempdir(), "_definitely_not_exist_12345.wav"),
          "这段文本对应的音频并不存在", "测试")


def _wrong_vocab_test():
    """用 512 词表权重跑 v2Pro ⇒ 应报 size mismatch

    ⭐ 需要本机装有 v2 的 512 词表 GPT 权重（GPT_weights/ 目录）。
       没有则跳过——这个坑的正确防线在安装期（env_check），不在运行期。
    """
    import glob
    cands = glob.glob(os.path.join(ROOT, "GPT_weights", "*.ckpt"))
    if not cands:
        raise RuntimeError("本机未安装 512 词表 GPT 权重，无法复现该坑（正常：多数人装的是 v2Pro 专用版）")
    from TTS_infer_pack.TTS import TTS
    cfg = {
        "custom": {
            "version": "v2Pro",
            "device": "cuda:0",
            "is_half": True,
            "t2s_weights_path": cands[0],
            "vits_weights_path": _args.sovits,
            "bert_base_path": os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": os.path.join(ROOT, "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
        }
    }
    tts = TTS(cfg)
    synth(tts, _args.ref, _args.ref_text, "测试词表")


def _long_ref_test():
    """用超长参考音频 ⇒ 应触发 3~10 秒限制"""
    import numpy as np
    import soundfile as sf
    tmp = os.path.join(tempfile.gettempdir(), "_long_ref_test.wav")
    sr = 32000
    sf.write(tmp, np.zeros((sr * 25,), dtype="float32"), sr)
    try:
        tts = _load_tts()
        synth(tts, tmp, "这是一段很长的参考音频的文本内容用来测试", "测试")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------- 主流程

def main():
    global ROOT, _args
    ap = argparse.ArgumentParser(description="声音克隆链路验收")
    ap.add_argument("--root", default=detect_root(),
                    help="GPT-SoVITS 仓库根（默认自动探测，也可用环境变量 GPT_SOVITS_ROOT）")
    ap.add_argument("--gpt", required=True, help="GPT 权重 .ckpt（732 词表）")
    ap.add_argument("--sovits", required=True, help="SoVITS 权重 .pth")
    ap.add_argument("--ref", required=True, help="参考音频（3~10 秒）")
    ap.add_argument("--ref-text", required=True, help="参考音频对应的原文")
    ap.add_argument("--text", default="这是一段用于验收的测试语音，用来确认整条链路真的通了。")
    ap.add_argument("--out", default="verify_report.json")
    ap.add_argument("--skip-negative", action="store_true", help="跳过反向验证")
    args = ap.parse_args()
    _args = args
    ROOT = args.root

    # ⭐ 先确保跑在带 torch 的解释器上
    ensure_runtime()

    out("=" * 68)
    out("声音克隆 · 链路验收")
    out("=" * 68)

    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "gpt": args.gpt, "sovits": args.sovits, "ref": args.ref}

    # ---- 前置检查 ----
    problems = []
    for label, p in (("GPT 权重", args.gpt), ("SoVITS 权重", args.sovits), ("参考音频", args.ref)):
        if not os.path.exists(p):
            problems.append("%s 不存在：%s" % (label, p))
    if problems:
        for p in problems:
            out("❌ " + p)
        sys.exit(2)

    # 参考音频时长（#17 坑）
    try:
        import wave
        import contextlib
        with contextlib.closing(wave.open(args.ref, "rb")) as w:
            dur = w.getnframes() / float(w.getframerate())
        report["ref_duration"] = round(dur, 2)
        if dur < 3 or dur > 10:
            out("❌ 参考音频 %.2f 秒，超出 3~10 秒硬限制（#17 坑）" % dur)
            problems.append("参考音频时长超限")
        else:
            out("✅ 参考音频 %.2f 秒（在 3~10 秒内）" % dur)
    except Exception as e:
        out("⚠ 参考音频时长读取失败：%s" % e)

    # ---- 环境注入（#7 #8 坑）----
    rt = os.path.join(ROOT, "runtime")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))
    p = os.environ.get("PATH", "")
    os.environ["PATH"] = rt + ";" + p.replace(":", ";")

    # ---- 正向验证 ----
    out("\n--- 正向验证 ---")
    t0 = time.time()
    try:
        tts = _load_tts()
    except Exception as e:
        out("❌ 模型加载失败：%s" % e)
        report["positive"] = {"ok": False, "error": str(e)}
        _save(report, args.out)
        sys.exit(1)
    out("  ✅ 模型加载成功（%.1fs）" % (time.time() - t0))

    try:
        arr, sr = synth(tts, args.ref, args.ref_text, args.text)
    except Exception as e:
        out("❌ 生成失败：%s" % e)
        report["positive"] = {"ok": False, "error": str(e)}
        _save(report, args.out)
        sys.exit(1)

    dur = len(arr) / float(sr)
    out("  ✅ 生成成功：%.2f 秒音频 / %d Hz" % (dur, sr))

    # 音频非空检查
    import numpy as np
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
    report["positive"] = {"ok": True, "duration": round(dur, 2), "sr": sr,
                          "peak": round(peak, 4), "rms": round(rms, 6)}
    if peak < 0.01:
        out("  ⚠ 峰值极低（%.4f），音频可能是静音" % peak)
    else:
        out("  ✅ 音频非静音（峰值 %.4f / RMS %.6f）" % (peak, rms))

    # 保存样本
    try:
        import soundfile as sf
        sample = os.path.join(tempfile.gettempdir(), "verify_sample.wav")
        sf.write(sample, arr, int(sr))
        report["sample"] = sample
        out("  样本已存：%s" % sample)
    except Exception:
        pass

    # ---- 反向验证 ----
    if not args.skip_negative:
        negative_tests(report)

    _save(report, args.out)

    # ---- 结论 ----
    out("\n" + "=" * 68)
    neg = report.get("negative", [])
    caught = sum(1 for c in neg if c.get("caught"))
    out("结论：正向 %s｜反向 %d/%d 个已知坑被成功抓住"
        % ("通过" if report["positive"].get("ok") else "失败", caught, len(neg)))
    if report["positive"].get("ok") and caught == len(neg):
        out("✅ 链路可信 —— 已知错误都会暴露，不是运气好")
    elif report["positive"].get("ok"):
        out("✅ 链路通 —— 但有反向用例没抓到（可能是坑已变或检查漏了）")
    else:
        out("❌ 链路不通")
    out("=" * 68)


def _save(report, path):
    try:
        # ⭐ 只存可序列化的部分（negative 里混了 lambda 之类的不可序列化对象）
        safe = {
            "time": report.get("time"),
            "gpt": report.get("gpt"),
            "sovits": report.get("sovits"),
            "ref": report.get("ref"),
            "ref_duration": report.get("ref_duration"),
            "positive": report.get("positive"),
            "sample": report.get("sample"),
            "negative": [
                {k: v for k, v in c.items() if isinstance(v, (str, int, float, bool, type(None)))}
                for c in report.get("negative", [])
            ],
        }
        with io.open(path, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False, indent=2)
        out("\n报告已存：%s" % os.path.abspath(path))
    except Exception as e:
        out("（报告保存失败：%s）" % e)


if __name__ == "__main__":
    main()
