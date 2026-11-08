# -*- coding: utf-8 -*-
"""② 备料：切片 + 转写 + 繁简转换 + 生成合规的 list.txt。

⭐ 这条流水线对应 GPT-SoVITS 的 prepare_datasets 阶段 1~4，
   但**只做"文本侧"**（切片 + list.txt）——音频重采样与特征提取
   交给官方脚本（它们依赖特定的 ffmpeg 与环境）。

用法：
    python prepare_data.py --src <你的原始录音目录> --out <GPT-SoVITS 根>/data/<名字> \
                            [--min-len 4] [--max-len 10] [--lang zh]

产出：
    <out>/wavs/*.wav        切好的片段
    <out>/list.txt          ⭐ 格式：{音频路径}|{输出名}|{语言}|{文本}

⭐ 三个关键点：
1. **list.txt 第 0 位是音频路径**（不是输出名）——这是 #9 坑
2. **切片 3~10 秒**——推理硬限制（#17 坑）
3. **繁简统一**——训练与推理都用同一套字，否则会出现念不出的字
"""
import io
import os
import sys
import json
import shutil
import argparse
import subprocess
import wave
import contextlib

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- 音频工具

def wav_info(path):
    """返回 (采样率, 帧数, 声道)"""
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getframerate(), w.getnframes(), w.getnchannels()


def slice_wav(src, dst, start_sec, dur_sec, sr_target=32000):
    """切一段 wav（只处理 wav，不转码）"""
    sr, frames, ch = wav_info(src)
    start = int(start_sec * sr)
    n = int(dur_sec * sr)
    if start + n > frames:
        n = frames - start
    if n <= 0:
        return False
    with contextlib.closing(wave.open(src, "rb")) as w:
        w.setpos(start)
        data = w.readframes(n)
    with contextlib.closing(wave.open(dst, "wb")) as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)
    return True


def to_mono16k(src, dst, ffmpeg):
    """转成 16k 单声道 16bit（GPT-SoVITS 训练输入标准）"""
    cmd = [ffmpeg, "-y", "-i", src, "-ac", "1", "-ar", "16000",
           "-sample_fmt", "s16", "-acodec", "pcm_s16le", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(dst)


# ---------------------------------------------------------------- 文本工具

def to_simplified(text, zhconv_available=True):
    """繁 → 简。用 opencc 或 zhconv，都没有就原样返回。"""
    if not zhconv_available or not text:
        return text
    try:
        import zhconv
        return zhconv.convert(text, "zh-cn")
    except Exception:
        pass
    try:
        import opencc
        return opencc.OpenCC("t2s").convert(text)
    except Exception:
        return text


# ---------------------------------------------------------------- 主流程

def segment_by_energy(src, ffmpeg, min_len=4.0, max_len=9.0, sr=16000):
    """用静音检测切分（ffmpeg silencedetect）。

    返回 [(起点秒, 时长秒), ...]
    """
    cmd = [ffmpeg, "-i", src, "-af",
           "silencedetect=noise=-35dB:d=0.35", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    sil = []
    for line in (r.stderr or "").splitlines():
        if "silence_start" in line:
            try:
                sil.append(("s", float(line.split("silence_start:")[1].split()[0])))
            except Exception:
                pass
        elif "silence_end" in line:
            try:
                sil.append(("e", float(line.split("silence_end:")[1].split()[0])))
            except Exception:
                pass

    total = wav_info(src)[1] / float(wav_info(src)[0])

    # 无静音信息 ⇒ 按时长等分
    if not sil:
        step = max_len
        out = []
        t = 0.0
        while t < total - 0.5:
            out.append((t, min(step, total - t)))
            t += step
        return out

    # 由静音区间推出有声段
    segs = []
    t = 0.0
    for kind, v in sil:
        if kind == "s":
            if v - t > 0.6:
                segs.append((t, v - t))
            t = v
        else:
            t = v
    if total - t > 0.6:
        segs.append((t, total - t))

    # 合并过短的、切断过长的
    out = []
    for s, d in segs:
        if d < min_len and out:
            out[-1] = (out[-1][0], out[-1][1] + d + 0.05)
            continue
        while d > max_len:
            out.append((s, max_len))
            s += max_len
            d -= max_len
        if d >= min_len:
            out.append((s, d))
    return out


def main():
    ap = argparse.ArgumentParser(description="声音克隆备料：切片 + list.txt")
    ap.add_argument("--src", required=True, help="原始录音目录")
    ap.add_argument("--out", required=True, help="输出目录（data/<名字>）")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 可执行文件")
    ap.add_argument("--min-len", type=float, default=4.0, help="最短切片秒")
    ap.add_argument("--max-len", type=float, default=9.0, help="最长切片秒（推理硬限 10s）")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--texts", help="文本文件（每行一句，与切片一一对应）；不给则留空待填")
    ap.add_argument("--no-slice", action="store_true", help="不切片，直接用整段（仍需 3~10s）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wavdir = os.path.join(args.out, "wavs")
    os.makedirs(wavdir, exist_ok=True)

    # ---- 文本 ----
    texts = []
    if args.texts and os.path.exists(args.texts):
        with io.open(args.texts, "r", encoding="utf-8-sig") as f:
            texts = [l.strip() for l in f.read().replace("\r\n", "\n").split("\n") if l.strip()]

    # ---- 找源文件 ----
    srcs = []
    for fn in sorted(os.listdir(args.src)):
        if fn.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
            srcs.append(os.path.join(args.src, fn))
    if not srcs:
        print("!! %s 里没有音频文件" % args.src, file=sys.stderr)
        sys.exit(2)
    print("找到 %d 个源文件" % len(srcs))

    rows = []
    idx = 0
    for src in srcs:
        base = os.path.splitext(os.path.basename(src))[0]
        # 非 wav 先转
        work = src
        tmp = None
        if not src.lower().endswith(".wav"):
            tmp = os.path.join(wavdir, "_tmp_%s.wav" % base)
            if not to_mono16k(src, tmp, args.ffmpeg):
                print("  ⚠ 转码失败：%s" % os.path.basename(src))
                continue
            work = tmp

        if args.no_slice:
            cuts = [(0.0, min(wav_info(work)[1] / float(wav_info(work)[0]), 9.0))]
        else:
            cuts = segment_by_energy(work, args.ffmpeg, args.min_len, args.max_len)

        for k, (start, dur) in enumerate(cuts):
            name = "%s_%02d.wav" % (base, k)
            dst = os.path.join(wavdir, name)
            if not slice_wav(work, dst, start, dur):
                continue
            idx += 1
            rows.append({
                # ⭐ 真实 list.txt 用【相对路径】wavs/xxx.wav（相对 list.txt 所在目录）
                "path": "wavs/" + name,
                "abs": os.path.join(wavdir, name).replace("\\", "/"),
                "name": name[:-4],
                "dur": round(dur, 2),
                "text": texts[idx - 1] if idx - 1 < len(texts) else "",
            })

        if tmp and os.path.exists(tmp):
            os.remove(tmp)

    # ---- 转繁为简（统一字表） ----
    for r in rows:
        if r["text"]:
            r["text"] = to_simplified(r["text"])

    # ---- 写 list.txt ⭐ 格式：{相对路径}|{输出名}|{语言}|{文本} ----
    #    路径是【相对 list.txt 所在目录】的，如 wavs/nv_001.wav
    #    ⭐ 第 0 位是路径，第 1 位是输出名（可含 .wav），语言 zh
    lst = os.path.join(args.out, "list.txt")
    with io.open(lst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("%s|%s|%s|%s\n" % (r["path"], r["name"] + ".wav", args.lang, r["text"]))

    total = sum(r["dur"] for r in rows)
    print("\n=== 备料完成 ===")
    print("切片 %d 个 / 总时长 %.1f 秒 → %s" % (len(rows), total, wavdir))
    print("list.txt → %s" % lst)
    print("\n⭐ 下一步（官方脚本，注意 PATH 分隔符是 ';' 不是 ':'）：")
    print("  1. 语义提取  GPT_SoVITS/prepare_datasets/3-get-semantic.py")
    print("  2. HuBERT    GPT_SoVITS/prepare_datasets/7-get-hubert-wav32k.py")
    print("  3. 训练前    mkdir -p logs_s2_v2Pro   ⭐（#13 坑，漏了会崩在保存时）")

    if not texts:
        print("\n⚠ 没有 --texts ⇒ list.txt 的文本列为空。")
        print("  请给每行一句的文本文件重跑，或手工填写后转繁为简。")

    # ---- 报告 ----
    rep = os.path.join(args.out, "prepare_report.json")
    try:
        with io.open(rep, "w", encoding="utf-8") as f:
            json.dump({"count": len(rows), "total_sec": round(total, 1),
                       "rows": rows}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
