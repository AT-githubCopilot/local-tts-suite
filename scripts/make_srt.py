# -*- coding: utf-8 -*-
"""
本地 TTS · 字幕生成器
======================
把「配音台生成的多个 wav」+「对应的文字」→ SRT 字幕文件。

⭐ 为什么字幕要跟"实际音频"对齐，而不是跟原稿对齐
------------------------------------------------
TTS 念出来的语速跟原稿不一样（还受 speed_factor / 语气参数影响），
所以拿原始 ASR 的时间戳去套 TTS 音频，字幕会飘。

本工具的做法：**按实际 wav 的时长顺序累加**，得到每句的起止时间。
⇒ 字幕与配音**天然对齐**。

用法
----
    # ① 批量生成的 wav + 同名 txt（每行一句）
    python make_srt.py --wavs out1.wav out2.wav --texts 台词.txt -o 旁白.srt

    # ② 从outputs 目录按修改时间取最近的 N 个
    python make_srt.py --outputs-dir outputs --latest 3 --texts 台词.txt -o 旁白.srt

    # ③ 直接给「句:时长」对照表
    python make_srt.py --segments "第一句|3.2" "第二句|5.8" -o 旁白.srt

    # ④ 批量接口的 items JSON（最省事）
    python make_srt.py --from-api-items items.json -o 旁白.srt
"""
import io
import os
import sys
import json
import glob
import argparse
import wave
import contextlib

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- 工具

def wav_duration(path):
    """读 wav 时长（秒）。非 wav 或读不到时返回 None。"""
    try:
        with contextlib.closing(wave.open(path, "rb")) as w:
            frames = w.getnframes()
            rate = w.getframerate() or 1
            return frames / float(rate)
    except Exception:
        pass
    # 兜底：用 soundfile（若可用）
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def fmt_srt_time(seconds):
    """秒 → SRT 时间戳 HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def write_srt(segments, out_path, gap=0.12):
    """写 SRT。segments = [(text, start, end), ...]"""
    lines = []
    for i, (text, start, end) in enumerate(segments, 1):
        # SRT 里换行用 \n 即可，但先去掉文本首尾空白
        text = (text or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append("%s --> %s" % (fmt_srt_time(start), fmt_srt_time(end)))
        lines.append(text)
        lines.append("")
    content = "\n".join(lines)
    with io.open(out_path, "w", encoding="utf-8-sig") as f:   # BOM：剪映/Pr 更友好
        f.write(content)
    return content.count("-->")


def split_text_lines(path):
    """读文本文件，按行拆句（去掉空行）"""
    with io.open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    out = []
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if s:
            out.append(s)
    return out


# ---------------------------------------------------------------- 主流程

def build_from_wavs(wavs, texts, gap=0.12):
    """按 wav 顺序累加时长，配 texts 里的句子"""
    segs = []
    t = 0.0
    for i, w in enumerate(wavs):
        dur = wav_duration(w) or 0.0
        txt = texts[i] if i < len(texts) else ""
        start = t
        end = t + dur
        segs.append((txt, start, end))
        t = end + gap
    return segs


def main():
    ap = argparse.ArgumentParser(description="配音台字幕生成器")
    ap.add_argument("--wavs", nargs="*", help="wav 文件列表（按播放顺序）")
    ap.add_argument("--outputs-dir", help="outputs 目录")
    ap.add_argument("--latest", type=int, help="取 outputs 里最近的 N 个（新的在前）")
    ap.add_argument("--texts", help="台词文件（每行一句）")
    ap.add_argument("--segments", nargs="*", help='直接给 "句子|时长" 对照')
    ap.add_argument("--from-api-items", help="配音台批量返回的 items JSON")
    ap.add_argument("-o", "--out", default="旁白.srt", help="输出 srt 路径")
    ap.add_argument("--gap", type=float, default=0.12, help="句间间隔秒数")
    ap.add_argument("--dry", action="store_true", help="只打印不写文件")
    args = ap.parse_args()

    texts = []
    if args.texts:
        texts = split_text_lines(args.texts)

    segments = None

    # ---- 路径①：直接给 wav + 台词 ----
    if args.wavs:
        wavs = list(args.wavs)
        if not texts:
            print("!! 没给 --texts，无法配文", file=sys.stderr)
            sys.exit(2)
        if len(texts) != len(wavs):
            print("⚠ 台词 %d 句 vs 音频 %d 条，数量对不上（按序号配）"
                  % (len(texts), len(wavs)), file=sys.stderr)
        segments = build_from_wavs(wavs, texts, args.gap)

    # ---- 路径②：从 outputs 取最近的 N 个 ----
    elif args.outputs_dir and args.latest:
        files = sorted(glob.glob(os.path.join(args.outputs_dir, "*.wav")),
                       key=os.path.getmtime, reverse=True)[: args.latest]
        # 最新的在最后播放 ⇒ 反转成播放顺序
        wavs = list(reversed(files))
        if not texts:
            print("!! 没给 --texts", file=sys.stderr)
            sys.exit(2)
        print("取到 %d 个 wav（按时间正序）" % len(wavs))
        segments = build_from_wavs(wavs, texts, args.gap)

    # ---- 路径③：直接给 句子|时长 ----
    elif args.segments:
        segments = []
        t = 0.0
        for s in args.segments:
            if "|" not in s:
                continue
            txt, dur = s.rsplit("|", 1)
            try:
                dur = float(dur)
            except ValueError:
                dur = 0.0
            segments.append((txt, t, t + dur))
            t += dur + args.gap

    # ---- 路径④：配音台 API 的 items ----
    elif args.from_api_items:
        with io.open(args.from_api_items, "r", encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict):
            items = items.get("items", [])
        segments = []
        t = 0.0
        for it in items:
            if not it.get("ok"):
                continue
            dur = float(it.get("duration") or 0.0)
            segments.append((it.get("text", ""), t, t + dur))
            t += dur + args.gap

    if segments is None:
        print(__doc__)
        sys.exit(2)

    # ---- 输出 ----
    print("\n=== 字幕预览 ===")
    for i, (txt, s, e) in enumerate(segments, 1):
        print("%2d  %s --> %s  %s" % (i, fmt_srt_time(s), fmt_srt_time(e),
                                      (txt or "")[:40]))

    if args.dry:
        print("\n(--dry，未写文件)")
        return

    n = write_srt(segments, args.out, args.gap)
    print("\n✅ 已写入 %s（%d 条字幕）" % (args.out, n))


if __name__ == "__main__":
    main()
