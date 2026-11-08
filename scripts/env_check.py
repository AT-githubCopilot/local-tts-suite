# -*- coding: utf-8 -*-
"""① 环境探测：本地声音克隆的可执行性体检。

只读，不装不改任何东西。

用法：
    python env_check.py [--root <GPT-SoVITS 仓库根>]

输出：一张"能不能干、缺什么、怎么补"的报告。
"""
import io
import os
import sys
import json
import shutil
import argparse
import platform
import subprocess

sys.stdout.reconfigure(encoding="utf-8")

OK = "✅"
WARN = "⚠️"
BAD = "❌"

# ---- 自动探测 GPT-SoVITS 仓库根 ----
# ⭐ 优先级：--root 参数 > 环境变量 GPT_SOVITS_ROOT > 常见安装位置
#    下面的路径是【示例】，请改成你自己的实际路径
_ROOT_CANDIDATES = [
    os.environ.get("GPT_SOVITS_ROOT", ""),
    # ⭐ 下面是【常见安装位置示例】，按你自己的实际路径改；
    #    也可以直接设环境变量 GPT_SOVITS_ROOT 一次搞定
    r"F://AIStack//GPT-SoVITS//GPT-SoVITS-v2pro-20250604-nvidia50",
    r"D:\AIStack\GPT-SoVITS\GPT-SoVITS-v2pro-20250604-nvidia50",
    r"C:\AIStack\GPT-SoVITS\GPT-SoVITS-v2pro-20250604-nvidia50",
    os.path.expanduser("~/AIStack/GPT-SoVITS/GPT-SoVITS-v2pro-20250604-nvidia50"),
]


def detect_root():
    """自动探测 GPT-SoVITS 仓库根。

    优先级：--root 参数 > 环境变量 GPT_SOVITS_ROOT > 常见安装位置
    """
    for c in _ROOT_CANDIDATES:
        if c and os.path.isdir(c):
            return c
    return ""


# ---- 检查项定义：(名称, 检查函数) ----
results = []


def add(level, name, detail, fix=""):
    results.append({"level": level, "name": name, "detail": detail, "fix": fix})


def check_gpu():
    """GPU / 驱动 / CUDA 能力"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.strip().split("\n"):
                name, mem, drv = [x.strip() for x in line.split(",")]
                add(OK, "GPU", "%s | 显存 %s | 驱动 %s" % (name, mem, drv))
            return
    except Exception:
        pass
    add(WARN, "GPU", "nvidia-smi 不可用或无 NVIDIA 卡",
        "纯 CPU 也能跑，但速度慢 10~50 倍，不推荐")


def check_torch_cuda(py=None):
    """torch + CUDA 实际可用性

    ⭐ 用【项目自己的 runtime python】检查，而不是当前解释器 ——
    当前 shell 的 python 常常没有 torch，但项目 runtime 里有。
    """
    code = (
        "import torch;"
        "print(torch.__version__ + '|' + str(torch.cuda.is_available()) + '|' + "
        "(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '') + '|' + "
        "(str(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else '0,0'))"
    )
    for exe in [py] if py else []:
        try:
            out = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=90)
            if out.returncode == 0 and out.stdout.strip():
                parts = out.stdout.strip().split("|")
                if len(parts) >= 4:
                    v, cuda, dev, cap = parts[0], parts[1] == "True", parts[2], parts[3]
                    add(OK, "PyTorch", "%s | CUDA 可用 | %s" % (v, dev))
                    try:
                        major = int(cap.strip("() ").split(",")[0])
                        # ⭐ sm_120 (Blackwell, 算力 12.0) 需要 cu128；
                        #    torch 版本字符串里带 "+cuXXX" 才是真实构建
                        build = ""
                        if "+cu" in v:
                            build = v.split("+cu")[1]
                        if major >= 12:
                            if build and int("".join(ch for ch in build if ch.isdigit()) or 0) < 128:
                                add(BAD, "CUDA 版本",
                                    "算力 %s 需 cu128+，当前构建是 cu%s" % (cap, build),
                                    "装 torch cu128：pip install torch --index-url .../cu128")
                            else:
                                add(OK, "CUDA 版本",
                                    "算力 %s | 构建 cu%s ✓" % (cap, build or "?"))
                    except Exception:
                        pass
                    return
        except Exception:
            pass
    # 退路：当前解释器
    try:
        import torch
        v = torch.__version__
        if torch.cuda.is_available():
            add(OK, "PyTorch", "%s | CUDA 可用 | %s" % (v, torch.cuda.get_device_name(0)))
        else:
            add(WARN, "PyTorch", "%s | CUDA 不可用" % v, "确认 NVIDIA 驱动")
    except ImportError:
        add(WARN, "PyTorch", "当前解释器未装 torch",
            "若项目 runtime 里已装则无碍；否则 pip install torch --index-url .../cu128")
    except Exception as e:
        add(WARN, "PyTorch", "检测异常：%s" % e)


def check_paths(root):
    """仓库结构完整性"""
    if not root or not os.path.isdir(root):
        add(BAD, "仓库路径", "%s 不存在" % root, "clone 仓库或改 --root")
        return
    add(OK, "仓库路径", root)

    need = [
        ("推理核心", "GPT_SoVITS/TTS_infer_pack"),
        ("推理脚本", "GPT_SoVITS/inference_webui.py"),
        ("语义提取", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"),
        ("ASR", "GPT_SoVITS/tools/asr/funasr_asr.py"),
        ("BERT", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
    ]
    for label, rel in need:
        p = os.path.join(root, rel.replace("/", os.sep))
        if os.path.exists(p):
            add(OK, label, "✓")
        else:
            add(WARN, label, "缺 %s" % rel, "确认 clone 完整")

    # ⭐ 权重与词表
    pm = os.path.join(root, "GPT_SoVITS", "pretrained_models")
    s1v3 = os.path.join(pm, "s1v3.ckpt")
    if os.path.exists(s1v3):
        add(OK, "GPT 底模", "s1v3.ckpt 在位（732 词表，v2Pro 正确底模）")
    else:
        add(BAD, "GPT 底模", "缺 s1v3.ckpt",
            "⭐ v2Pro 必须用它；用 512 词表权重会报 size mismatch 512 vs 732")

    # ⭐ s2 底模：v2Pro 用 s2G488k.pth（不是 s2G.pth）
    s2 = os.path.join(pm, "s2G488k.pth")
    if os.path.exists(s2):
        add(OK, "SoVITS 底模", "s2G488k.pth 在位")
    else:
        alt = os.path.join(pm, "s2Gv3.pth")
        if os.path.exists(alt):
            add(WARN, "SoVITS 底模", "有 s2Gv3.pth 但缺 s2G488k.pth",
                "v2Pro 走 48k 路线，确认你的版本要哪个")
        else:
            add(WARN, "SoVITS 底模", "缺 s2G488k.pth", "训练需要")

    hu = os.path.join(pm, "chinese-hubert-base")
    if os.path.isdir(hu):
        add(OK, "HuBERT", "chinese-hubert-base 在位")
    else:
        add(WARN, "HuBERT", "缺 chinese-hubert-base", "中文版必需")

    # ⭐ 语义 token 与 SSL 特征（训练必需）
    for sub, label in (("5-wav32k", "重采样音频"),
                       ("6-name2semantic.tsv", "语义 token"),
                       ("7-cnhubert", "HuBERT 特征")):
        if any(os.path.exists(os.path.join(root, "logs", d, sub))
               for d in (os.listdir(os.path.join(root, "logs"))
                         if os.path.isdir(os.path.join(root, "logs")) else [])):
            add(OK, "训练产物 %s" % label, "在位")
        else:
            add(WARN, "训练产物 %s" % label, "缺 %s" % sub,
                "若要重训需重跑 prepare_datasets 阶段 5~7")


def check_logs_dir(root):
    """⭐ 训练目录（#13 坑）"""
    d = os.path.join(root, "logs_s2_v2Pro")
    if os.path.isdir(d):
        add(OK, "训练输出目录", "logs_s2_v2Pro 已存在")
    else:
        add(WARN, "训练输出目录", "缺 logs_s2_v2Pro",
            "⭐ 官方 webui.py 会自动建；直接调脚本会绕过 ⇒ 训到保存时崩。先 mkdir -p")


def check_ffmpeg(root=None):
    """ffmpeg 可用性（#7 坑：PATH 分隔符）"""
    exe = shutil.which("ffmpeg")
    if exe:
        add(OK, "ffmpeg", exe)
        return
    # ⭐ 项目 runtime 里通常自带 ffmpeg
    if root:
        for cand in (os.path.join(root, "runtime", "ffmpeg.exe"),
                     os.path.join(root, "ffmpeg.exe")):
            if os.path.exists(cand):
                add(WARN, "ffmpeg", "项目 runtime 里有，但不在 PATH",
                    "⭐ Git Bash 的 PATH 用 ':'，Windows Python 要 ';'；"
                    "在 runner 里修：os.environ['PATH']=rt+';'+p.replace(':',';')")
                return
    add(WARN, "ffmpeg", "找不到",
        "装 ffmpeg 或用项目 runtime 自带的；注意 Windows PATH 分隔符是 ';' 不是 ':'")


def check_ascii_path(root):
    """⭐ 路径纯 ASCII（Blender 那次栽过：中文路径 ⇒ OCIO 读不到）"""
    if not root:
        return
    non_ascii = [c for c in root if ord(c) > 127]
    if non_ascii:
        add(WARN, "路径字符", "路径含非 ASCII 字符：%s" % "".join(non_ascii),
            "⚠️ 某些库（如 Blender 的 OCIO）读不到 ⇒ 静默降级。尽量用纯 ASCII 路径")
    else:
        add(OK, "路径字符", "纯 ASCII")


def check_disk():
    """磁盘余量"""
    try:
        usage = shutil.disk_usage(os.path.abspath(os.sep))
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 20:
            add(BAD, "磁盘余量", "仅 %.1f GB 可用" % free_gb,
                "完整安装需 40+ GB（模型 19 GB + 训练产物 1 GB + 缓存）")
        elif free_gb < 50:
            add(WARN, "磁盘余量", "%.1f GB 可用" % free_gb, "够用但不宽裕")
        else:
            add(OK, "磁盘余量", "%.1f GB 可用" % free_gb)
    except Exception as e:
        add(WARN, "磁盘余量", "检测失败：%s" % e)


def check_data(root):
    """训练数据"""
    d = os.path.join(root, "data")
    if not os.path.isdir(d):
        add(WARN, "训练数据", "无 data 目录", "先跑 prepare_data.py 备料")
        return
    subs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    if not subs:
        add(WARN, "训练数据", "data 下无子目录", "先跑 prepare_data.py")
        return
    for s in subs:
        wavs = []
        for dp, dns, fns in os.walk(os.path.join(d, s)):
            wavs += [f for f in fns if f.lower().endswith(".wav")]
        lst = os.path.join(d, s, "list.txt")
        total_sec = 0.0
        try:
            import wave
            import contextlib
            for w in wavs:
                p = os.path.join(d, s, "wavs", w)
                if not os.path.exists(p):
                    p = os.path.join(dp, w)
                try:
                    with contextlib.closing(wave.open(p, "rb")) as f:
                        total_sec += f.getnframes() / float(f.getframerate())
                except Exception:
                    pass
        except Exception:
            pass
        if os.path.exists(lst):
            add(OK, "训练数据 %s" % s,
                "%d 个 wav / %.1f 秒有效 | list.txt 在位" % (len(wavs), total_sec))
        else:
            add(WARN, "训练数据 %s" % s,
                "%d 个 wav 但缺 list.txt" % len(wavs),
                "list.txt 格式：{音频路径}|{输出名}|{语言}|{文本}（⭐ 第 0 位是路径）")


def check_crlf(paths):
    """⭐ #18 坑：bat 文件必须是 CRLF"""
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            raw = open(p, "rb").read()
        except Exception:
            continue
        has_crlf = b"\r\n" in raw
        has_lf = b"\n" in raw
        if has_lf and not has_crlf:
            add(BAD, "换行格式 %s" % os.path.basename(p),
                "⚠️ LF 换行 —— Windows cmd 双击会静默失败",
                "转成 CRLF")
        else:
            add(OK, "换行格式 %s" % os.path.basename(p), "CRLF ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=detect_root(),
                    help="GPT-SoVITS 仓库根（默认自动探测，也可用环境变量 GPT_SOVITS_ROOT）")
    ap.add_argument("--py", help="项目 runtime 的 python.exe（用于检测 torch）")
    ap.add_argument("--bats", nargs="*", default=[], help="要检查换行格式的 .bat")
    args = ap.parse_args()

    print("=" * 68)
    print("本地声音克隆 · 环境体检")
    print("=" * 68)
    print("系统：%s %s" % (platform.platform(), platform.python_version()))

    check_gpu()
    check_torch_cuda(args.py)
    check_paths(args.root)
    check_logs_dir(args.root)
    check_ffmpeg(args.root)
    check_ascii_path(args.root)
    check_disk()
    check_data(args.root)
    if args.bats:
        check_crlf(args.bats)

    # ---- 输出 ----
    print()
    for r in results:
        print("%s %-16s %s" % (r["level"], r["name"], r["detail"]))
        if r["fix"]:
            print("   %s → %s" % ("　　", r["fix"]))

    bad = sum(1 for r in results if r["level"] == BAD)
    warn = sum(1 for r in results if r["level"] == WARN)
    print("\n" + "=" * 68)
    print("结论：%d 项必须修，%d 项建议修" % (bad, warn))
    if bad == 0:
        print("✅ 可以开工（建议把 WARN 项也处理掉，能省后面几小时）")
    else:
        print("❌ 先修上面标 ❌ 的项")
    print("=" * 68)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "env_report.json")
    try:
        with io.open(os.path.abspath(out), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("（报告已存：%s）" % os.path.abspath(out))
    except Exception:
        pass


if __name__ == "__main__":
    main()
