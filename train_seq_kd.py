"""
train_seq_kd.py
Train a HuBERT student whose output is the SAME as the NeMo teacher's
(slu_conformer_transformer_large_slurp): same tokenizer, same Transformer decoder,
same semantics string, e.g.

    {'scenario': 'alarm'| 'action': 'set'| 'entities': [{'type': 'time'| 'filler': 'five am'}]}

Student sees NOISY audio (default 5 dB SNR); teacher sees CLEAN audio.

Loss per batch (all token-level terms are teacher-forced on the gold target):
  ce      : cross-entropy of student tokens vs gold semantics
  kd_tok  : T^2 * KL(teacher token distribution || student)  -- same vocab, so
            the student matches the teacher's full distribution at every token
  kd_dec  : 1 - cos(student decoder states, teacher decoder states)
  kd_enc  : 1 - cos(student adapter output, teacher Conformer output),
            time-aligned (student 20 ms -> teacher 40 ms frames)

  L = w_ce*ce + w_tok*kd_tok + w_dec*kd_dec + w_enc*kd_enc

Example:
    python train_seq_kd.py \
        --slurp_jsonl_dir /data/slurp/dataset/slurp --audio_root /data/slurp/audio \
        --noise_dir /data/musan/noise --snr_db 5 \
        --output_dir ./ckpt_seq_kd --epochs 20 --batch_size 8
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import intent_accuracy, slu_f1
from model_seq import HubertSemanticSLU
from seq_dataset import SemanticsSlurpDataset, TargetFormatter, collate_semantics
from teacher_nemo import (DEFAULT_NEMO_PATH, NemoSLUTeacher, download_teacher,
                          entities_to_tagged_text, parse_semantics, semantics_intent)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slurp_jsonl_dir", required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--output_dir", default="./ckpt_seq_kd")
    p.add_argument("--noise_dir", default=None)
    p.add_argument("--dev_noise_dir", default=None)
    p.add_argument("--noisy_audio_root", default=None)
    p.add_argument("--snr_db", type=float, default=5.0)
    p.add_argument("--max_audio_seconds", type=float, default=15.0)
    # student
    p.add_argument("--hubert_name", default="facebook/hubert-base-ls960")
    p.add_argument("--layer_fusion", choices=["single", "weighted_sum"], default="weighted_sum")
    p.add_argument("--semantic_layer", type=int, default=8)
    p.add_argument("--fusion_layers", default=None)
    p.add_argument("--freeze_encoder_layers", type=int, default=0)
    p.add_argument("--adapter_layers", type=int, default=2)
    p.add_argument("--random_decoder", action="store_true",
                   help="Random-init the teacher-shaped decoder instead of copying teacher weights")
    # teacher
    p.add_argument("--teacher_nemo", default=DEFAULT_NEMO_PATH)
    p.add_argument("--teacher_device", default=None)
    p.add_argument("--teacher_input", choices=["clean", "noisy"], default="clean")
    # optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5, help="HuBERT encoder")
    p.add_argument("--bridge_lr", type=float, default=5e-4, help="adapter + layer fusion")
    p.add_argument("--decoder_lr", type=float, default=1e-4, help="teacher-shaped decoder")
    p.add_argument("--freeze_decoder_epochs", type=int, default=1,
                   help="Keep the copied teacher decoder frozen for the first N epochs so the "
                        "encoder+adapter first learn to produce teacher-like features")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # loss weights
    p.add_argument("--ce_weight", type=float, default=1.0)
    p.add_argument("--kd_token_weight", type=float, default=1.0)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--kd_dec_weight", type=float, default=0.5)
    p.add_argument("--kd_enc_weight", type=float, default=1.0)
    # eval
    p.add_argument("--max_decode_len", type=int, default=128)
    p.add_argument("--eval_teacher", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def intents_from_jsonl(path):
    pairs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("scenario") is not None and r.get("action") is not None:
                    pairs.setdefault(f"{r['scenario']}_{r['action']}", (r["scenario"], r["action"]))
    return sorted(pairs), pairs


def pad_audio(items, key):
    n = max(it[key].shape[0] for it in items)
    wav = torch.zeros(len(items), n)
    lens = torch.zeros(len(items), dtype=torch.long)
    for i, it in enumerate(items):
        L = it[key].shape[0]
        wav[i, :L], lens[i] = it[key], L
    return wav, lens


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def enc_kd_loss(s_enc, t_enc, t_lens):
    Tt = t_enc.shape[1]
    s = F.interpolate(s_enc.transpose(1, 2), size=Tt, mode="linear", align_corners=False).transpose(1, 2)
    mask = (torch.arange(Tt, device=s.device)[None, :] < t_lens[:, None]).float()
    return masked_mean(1.0 - F.cosine_similarity(s, t_enc, dim=-1), mask)


def gold_tagged(entities):
    return entities_to_tagged_text({"entities": [{"type": t, "filler": f} for t, f in entities]})


def score(texts, batch_intents, batch_entities):
    sems = [parse_semantics(t) for t in texts]
    pi = [semantics_intent(s) or "<invalid>" for s in sems]
    acc = intent_accuracy(pi, batch_intents)
    f1 = slu_f1([entities_to_tagged_text(s) for s in sems], [gold_tagged(e) for e in batch_entities])
    return {"intent_accuracy": acc, **f1, "invalid": sum(s is None for s in sems)}


def fmt_m(m):
    return (f"intent_acc={m['intent_accuracy']:.4f} slu_f1={m['f1']:.4f} "
            f"(P={m['precision']:.4f} R={m['recall']:.4f}) unparseable={m['invalid']}")


# --------------------------------------------------------------------------- #
@torch.no_grad()
def teacher_outputs(teacher, loader, key):
    out = {}
    for b in tqdm(loader, desc=f"teacher decode ({key})"):
        enc, lens = teacher.encode(b[key].to(teacher.device), b["attention_mask"].sum(-1).to(teacher.device))
        for f, t in zip(b["audio_files"], teacher.decode(enc, lens)):
            out[f] = t
    return out


@torch.no_grad()
def evaluate(student, teacher, loader, device, max_len, teacher_ref=None, show=3):
    student.eval()
    texts, intents, ents, files = [], [], [], []
    for b in loader:
        ids = student.greedy(b["noisy"].to(device), b["attention_mask"].to(device),
                             teacher.bos, teacher.tok.eos_id, max_len=max_len)
        texts += [teacher.ids_to_text(x) for x in ids]
        intents += b["gold_intents"]
        ents += b["gold_entities"]
        files += b["audio_files"]
    student.train()
    m = score(texts, intents, ents)
    if teacher_ref:
        tt = [teacher_ref.get(f, "") for f in files]
        m["same_as_teacher"] = sum(a.strip() == b.strip() for a, b in zip(texts, tt)) / max(1, len(tt))
        m["same_intent_as_teacher"] = sum(
            semantics_intent(parse_semantics(a)) == semantics_intent(parse_semantics(b))
            for a, b in zip(texts, tt)) / max(1, len(tt))
    for i in range(min(show, len(texts))):
        print(f"    student: {texts[i]}")
        if teacher_ref:
            print(f"    teacher: {teacher_ref.get(files[i], '')}")
    return m


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if (args.noise_dir is None) == (args.noisy_audio_root is None):
        raise SystemExit("Give exactly one of --noise_dir or --noisy_audio_root")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seq_kd_args.json").write_text(json.dumps(vars(args), indent=2))
    jdir = Path(args.slurp_jsonl_dir)
    tdev = args.teacher_device or args.device

    # ---- data ----
    common = dict(max_audio_seconds=args.max_audio_seconds, snr_db=args.snr_db,
                  noisy_audio_root=args.noisy_audio_root)
    train_ds = SemanticsSlurpDataset(jdir / "train.jsonl", args.audio_root,
                                     noise_dir=args.noise_dir, deterministic=False, **common)
    dev_ds = SemanticsSlurpDataset(jdir / "devel.jsonl", args.audio_root,
                                   noise_dir=(args.dev_noise_dir or args.noise_dir) if args.noise_dir else None,
                                   deterministic=True, **common)
    print(f"[data] train={len(train_ds)} dev={len(dev_ds)} snr={args.snr_db} dB")

    # ---- teacher + calibration of its exact output format ----
    intent_list, pairs = intents_from_jsonl(jdir / "train.jsonl")
    teacher = NemoSLUTeacher(download_teacher(args.teacher_nemo), intent_list, pairs, tdev)
    cal_items = [dev_ds[i] for i in range(min(8, len(dev_ds)))]
    wav, lens = pad_audio(cal_items, "clean")
    teacher.calibrate(wav, lens)
    fmt = TargetFormatter(teacher)
    ex = cal_items[0]
    print(f"[format] gold target example: {fmt.render(ex['scenario'], ex['action'], ex['entities'])}")

    for ds in (train_ds, dev_ds):
        ds.attach_targets(fmt)
    print("[format] gold targets tokenised")
    collate = collate_semantics
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    teacher_ref = teacher_outputs(teacher, dev_loader, "clean")
    if args.eval_teacher:
        tn = teacher_outputs(teacher, dev_loader, "noisy")
        for key, ref in (("clean", teacher_ref), (f"noisy {args.snr_db} dB", tn)):
            texts = [ref[e["audio_file"]] for e in dev_ds.examples]
            m = score(texts, [f"{e['scenario']}_{e['action']}" for e in dev_ds.examples],
                      [e["entities"] for e in dev_ds.examples])
            print(f"[teacher] dev {key}: {fmt_m(m)}")

    # ---- student ----
    student = HubertSemanticSLU(
        teacher.model, hubert_name=args.hubert_name, layer_fusion=args.layer_fusion,
        semantic_layer=args.semantic_layer,
        fusion_layers=[int(x) for x in args.fusion_layers.split(",")] if args.fusion_layers else None,
        freeze_encoder_layers=args.freeze_encoder_layers, adapter_layers=args.adapter_layers,
        init_decoder_from_teacher=not args.random_decoder,
    ).to(args.device)

    groups = [
        {"params": [p for p in student.hubert.parameters() if p.requires_grad], "lr": args.lr},
        {"params": student.bridge_parameters(), "lr": args.bridge_lr},
        {"params": student.decoder_parameters(), "lr": args.decoder_lr},
    ]
    optimizer = torch.optim.AdamW(groups)
    total = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.bridge_lr, args.decoder_lr], total_steps=total)
    for name, g in zip(["hubert", "bridge", "decoder"], groups):
        print(f"[optim] {name}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']}")

    T = args.kd_temperature
    best, step = -1.0, 0
    for epoch in range(args.epochs):
        freeze_dec = epoch < args.freeze_decoder_epochs and not args.random_decoder
        student.set_decoder_trainable(not freeze_dec)
        student.train()
        running = {}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}{' (decoder frozen)' if freeze_dec else ''}")
        for b in pbar:
            att = b["attention_mask"]
            t_wav = b["clean"] if args.teacher_input == "clean" else b["noisy"]
            with torch.no_grad():
                t_enc, t_lens = teacher.encode(t_wav.to(tdev), att.sum(-1).to(tdev))
                t_logp, t_h = teacher.forced(t_enc, t_lens, b["dec_inp"].to(tdev), b["dec_mask"].to(tdev))
            t_enc, t_lens, t_logp, t_h = (x.to(args.device) for x in (t_enc, t_lens, t_logp, t_h))

            dec_inp, labels, mask = (b[k].to(args.device) for k in ("dec_inp", "labels", "dec_mask"))
            out = student(b["noisy"].to(args.device), att.to(args.device), dec_inp, mask)
            s_logp = out["log_probs"]

            ce = masked_mean(-s_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1), mask)
            t_logT = F.log_softmax(t_logp / T, -1)
            s_logT = F.log_softmax(s_logp / T, -1)
            kd_tok = masked_mean((t_logT.exp() * (t_logT - s_logT)).sum(-1), mask) * T * T
            kd_dec = masked_mean(1.0 - F.cosine_similarity(out["dec_states"], t_h.float(), dim=-1), mask)
            kd_enc = enc_kd_loss(out["enc"], t_enc.float(), t_lens)

            loss = (args.ce_weight * ce + args.kd_token_weight * kd_tok
                    + args.kd_dec_weight * kd_dec + args.kd_enc_weight * kd_enc)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            for k, v in (("loss", loss), ("ce", ce), ("kd_tok", kd_tok), ("kd_dec", kd_dec), ("kd_enc", kd_enc)):
                running[k] = 0.98 * running.get(k, float(v)) + 0.02 * float(v)
            if step % args.log_every == 0:
                pbar.set_postfix({k: f"{v:.3f}" for k, v in running.items()})

        m = evaluate(student, teacher, dev_loader, args.device, args.max_decode_len, teacher_ref)
        print(f"[epoch {epoch}] student dev noisy: {fmt_m(m)}")
        print(f"[epoch {epoch}] identical to teacher(clean): {m['same_as_teacher']:.4f}  "
              f"same intent: {m['same_intent_as_teacher']:.4f}")

        ckpt = {"state_dict": student.state_dict(), "prefix_parts": teacher.prefix_parts,
                "config": {k: getattr(args, k) for k in ("hubert_name", "layer_fusion", "semantic_layer",
                                                          "fusion_layers", "adapter_layers")}}
        torch.save(ckpt, out_dir / "last_student.pt")
        if m["f1"] > best:
            best = m["f1"]
            torch.save(ckpt, out_dir / "best_student.pt")
            print(f"[epoch {epoch}] new best (noisy dev slu_f1={best:.4f}) -> best_student.pt")


if __name__ == "__main__":
    main()
