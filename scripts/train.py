# -*- coding: utf-8 -*-
"""③ 训练：GPT-SoVITS 两阶段训练的启动器与监控。

⭐ 它做三件事：
   1. ⭐ **预检**（在训练开始前把已知的坑拦下来）——最值钱的一步
   2. 启动 s1（GPT）/ s2（SoVITS）训练
   3. 监控 loss，出报告

用法：
    # 只预检（不训练）
    python train.py --root <仓库根> --exp <实验名> --check-only

    # 训练 s1
    python train.py --root <仓库根> --exp my_voice --stage s1 --epochs 15

    # 训练 s2
    python train.py --root <仓库根> --exp my_voice --stage s2 --epochs 15
"""
import io
import os
import re
import sys
import json
import time
import shutil
import argparse
import subprocess

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



def out(m):
    print(m, flush=True)


# ---------------------------------------------------------------- 预检

def precheck(exp, stage):
    """⭐ 训练前的坑位拦截 —— 训练一次要几小时，不能因为低级错白跑。

    返回 (通过?, 问题列表)
    """
    problems = []
    warns = []

    # #13 坑：训练输出目录
    logs_dir = os.path.join(ROOT, "logs_s2_v2Pro")
    if stage == "s2":
        if not os.path.isdir(logs_dir):
            problems.append(("logs_s2_v2Pro 不存在",
                             "⭐ 官方 webui.py 会自动建，直接调脚本会绕过 ⇒ 训到保存时崩。"
                             "先 mkdir -p %s" % logs_dir))
        else:
            out("  ✅ logs_s2_v2Pro 已存在")

    # 训练数据与 list.txt
    data_dir = os.path.join(ROOT, "data", exp)
    if not os.path.isdir(data_dir):
        problems.append(("data/%s 不存在" % exp, "先跑 prepare_data.py 备料"))
    else:
        lst = os.path.join(data_dir, "list.txt")
        if not os.path.exists(lst):
            problems.append(("list.txt 不存在", "训练必需"))
        else:
            # ⭐ #9 坑：校验 list.txt 格式
            bad = 0
            total = 0
            with io.open(lst, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    parts = line.split("|")
                    if len(parts) < 3:
                        bad += 1
                    else:
                        p = parts[0]
                        # 路径应存在（相对 list.txt 所在目录）
                        if not os.path.exists(os.path.join(data_dir, p.replace("/", os.sep))):
                            bad += 1
            if bad:
                problems.append(("list.txt 有 %d/%d 行格式或路径不对" % (bad, total),
                                 "⭐ 格式应为 {相对路径}|{输出名}|{语言}|{文本}，"
                                 "第 0 位是路径"))
            else:
                out("  ✅ list.txt %d 行全部合规" % total)

            # 素材时长
            import wave
            import contextlib
            total_sec = 0.0
            n = 0
            wavdir = os.path.join(data_dir, "wavs")
            if os.path.isdir(wavdir):
                for fn in os.listdir(wavdir):
                    if fn.lower().endswith(".wav"):
                        try:
                            with contextlib.closing(wave.open(os.path.join(wavdir, fn), "rb")) as w:
                                total_sec += w.getnframes() / float(w.getframerate())
                            n += 1
                        except Exception:
                            pass
            out("  ✅ 素材 %d 个 / %.1f 秒有效" % (n, total_sec))
            if total_sec < 30:
                warns.append(("素材仅 %.1f 秒", "少于 30 秒容易训不稳，建议 ≥ 60 秒"))
            # ⭐ 切片时长超限（#17 坑：推理限 3~10 秒）
            long_cnt = 0
            if os.path.isdir(wavdir):
                for fn in os.listdir(wavdir):
                    if not fn.lower().endswith(".wav"):
                        continue
                    try:
                        with contextlib.closing(wave.open(os.path.join(wavdir, fn), "rb")) as w:
                            d = w.getnframes() / float(w.getframerate())
                        if d > 10.5 or d < 2.5:
                            long_cnt += 1
                    except Exception:
                        pass
            if long_cnt:
                warns.append(("%d 个切片时长不在 3~10 秒" % long_cnt,
                               "推理端会拒绝（#17 坑）；重切或改用短参考"))

    # 底模
    pm = os.path.join(ROOT, "GPT_SoVITS", "pretrained_models")
    s1v3 = os.path.join(pm, "s1v3.ckpt")
    if not os.path.exists(s1v3):
        problems.append(("s1v3.ckpt 不存在", "⭐ v2Pro 必须用 732 词表底模"))
    else:
        out("  ✅ s1v3.ckpt 在位（732 词表）")

    s2 = os.path.join(pm, "s2G488k.pth")
    if not os.path.exists(s2):
        warns.append(("s2G488k.pth 不存在", "训练需要"))

    # 语义 token 与 SSL 特征
    logs = os.path.join(ROOT, "logs", exp)
    if os.path.isdir(logs):
        if not any(os.path.exists(os.path.join(logs, "6-name2semantic.tsv")) for _ in [0]):
            warns.append(("缺 6-name2semantic.tsv", "跑 3-get-semantic.py"))
        if not os.path.isdir(os.path.join(logs, "7-cnhubert")):
            warns.append(("缺 7-cnhubert", "跑 7-get-hubert-wav32k.py"))
    else:
        warns.append(("logs/%s 不存在" % exp, "语义与特征还没提取"))

    return problems, warns


# ---------------------------------------------------------------- 训练

def find_train_script(stage):
    """找官方训练脚本"""
    cands = []
    for name in ("s1_train.py", "s2_train.py", "s1.py", "s2.py"):
        for base in ("GPT_SoVITS", os.path.join("GPT_SoVITS", "tools")):
            p = os.path.join(ROOT, base, name)
            if os.path.exists(p):
                cands.append(p)
    if stage == "s1":
        pref = ["s1_train.py", "s1.py"]
    else:
        pref = ["s2_train.py", "s2.py"]
    for want in pref:
        for c in cands:
            if os.path.basename(c) == want:
                return c
    return cands[0] if cands else None


def run_training(stage, exp, epochs, env_extra=None):
    """启动训练"""
    script = find_train_script(stage)
    if not script:
        out("❌ 找不到官方训练脚本（s1_train.py / s2_train.py）")
        out("   目录：%s" % ROOT)
        return False

    cmd = [sys.executable, script,
           "--exp_name", exp,
           "--epochs", str(epochs),
           "--pretrained_model", os.path.join(ROOT, "GPT_SoVITS", "pretrained_models", "s1v3.ckpt"),
           "--device", "cuda:0",
           "--is_half", "True",
           "--max_epoch", str(epochs),
           "--if_save_latest", "True",
           "--if_save_every_weights", "True",
           "--save_every_weight", "5",
           ]
    if stage == "s2":
        cmd += ["--pretrained_model", os.path.join(ROOT, "GPT_SoVITS", "pretrained_models", "s2G488k.pth")]

    out("\n训练命令：")
    out("  " + " ".join(cmd))
    out("\n开始训练（Ctrl+C 可中断）…\n")

    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    return r.returncode == 0


# ---------------------------------------------------------------- 主流程

def main():
    global ROOT
    ap = argparse.ArgumentParser(description="GPT-SoVITS 训练启动器")
    ap.add_argument("--root", default=detect_root(),
                    help="GPT-SoVITS 仓库根（默认自动探测，也可用环境变量 GPT_SOVITS_ROOT）")
    ap.add_argument("--exp", required=True, help="实验名（data/ 下同名目录）")
    ap.add_argument("--stage", choices=["s1", "s2"], default="s1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--check-only", action="store_true", help="只做预检，不训练")
    args = ap.parse_args()
    ROOT = args.root

    out("=" * 68)
    out("GPT-SoVITS 训练 · %s 阶段 · 实验「%s」" % (args.stage, args.exp))
    out("=" * 68)

    out("\n--- 预检（把已知的坑拦在训练之前）---")
    problems, warns = precheck(args.exp, args.stage)

    if warns:
        out("\n⚠ 建议处理：")
        for name, fix in warns:
            out("  ⚠ %s\n    → %s" % (name, fix))
    if problems:
        out("\n❌ 必须先修：")
        for name, fix in problems:
            out("  ❌ %s\n    → %s" % (name, fix))
        out("\n⭐ 训练一次要几小时，别因为低级错白跑。修完再执行。")
        sys.exit(2)

    out("\n✅ 预检通过")
    if args.check_only:
        out("（--check-only，未开始训练）")
        return

    out("\n--- 训练配方 ---")
    out("  阶段      ：%s" % args.stage)
    out("  epoch     ：%d" % args.epochs)
    out("  每 5 epoch 存一个权重（便于做对比组）")
    out("  ⭐ 推理时取【最后一个】权重，前面的是对照组")

    t0 = time.time()
    ok = run_training(args.stage, args.exp, args.epochs)
    dt = time.time() - t0

    out("\n" + "=" * 68)
    if ok:
        out("✅ 训练完成，用时 %.1f 分钟" % (dt / 60))
        out("\n下一步：")
        out("  1. 跑验收：python verify.py --gpt <新e15.ckpt> --sovits <新s150.pth> ...")
        out("  2. 对比试听：用不同 epoch 的权重各出几版，选最好的")
    else:
        out("❌ 训练未正常结束（退出码非 0）")
        out("   ⭐ 看日志尾部——真凶常在最后几行，不在错误摘要里")
    out("=" * 68)


if __name__ == "__main__":
    main()
