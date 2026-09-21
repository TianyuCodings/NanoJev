#!/usr/bin/env python3
"""Small, auditable non-generative decision learning experiment.

One backbone forward scores all candidate paths in a batch. `encode_leaves` is the
reference encoder: it writes the shared state/question prefix into every candidate
path, so a question with K candidates evaluates its prefix K times. Passing a
`shared_prefix.SharedPrefixEncoder` as `forward(..., prefix_sharing=encoder)`
computes each prefix once and reads only candidate suffixes against it; see
`scripts/shared_prefix.py` for the equivalence argument and `docs/SHARED_PREFIX.md`
for measured token counts. No chain of thought, vocabulary decoding loop, or
candidate-wise generation.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_decision_encoding():
    """Import the canonical encoder as a top-level module.

    `scripts/` is on `sys.path` when these files run as scripts, but not when the
    directory is imported as a package. The repository already relies on top-level names
    (`calibrated_objectives`, `predict_toy_decisions`), so load the same way and register
    under the real name, which keeps one implementation shared by both import styles.
    """
    name = "decision_encoding"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


decision_encoding = _load_decision_encoding()
build_candidate_paths = decision_encoding.build_candidate_paths


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def load_examples(path, tokenizer, max_length):
    examples, audit = [], []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        for qid, q in row['questions'].items():
            typ = q['type']
            # One canonical encoder for training and serving; see decision_encoding.py.
            ids, texts, prefix, leaves = build_candidate_paths(
                tokenizer.encode, row['state'], q, tokenizer.eos_token_id)
            if typ == 'choice':
                gold = ids.index(row['gold'][qid])
            else:
                gold = int(row['gold'][qid])
            if max(map(len, leaves)) > max_length:
                raise ValueError(f"No silent truncation: {row['id']}:{qid}")
            observed = row['teacher']['native_probs'][qid]
            if set(observed) != set(ids):
                raise ValueError(f"Teacher candidate mapping mismatch: {row['id']}:{qid}")
            raw = [float(observed[k]) for k in ids]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in raw):
                raise ValueError('Teacher probabilities must be finite and in [0,1]')
            decimals = row['teacher'].get('rounding', {}).get('probabilityDecimals')
            scale = 10 ** decimals if isinstance(decimals, int) else None
            if scale and any(abs(p*scale-round(p*scale)) > 1e-6 for p in raw):
                raise ValueError('Teacher probabilities violate declared decimal precision')
            unit_sum = sum(round(p * scale) for p in raw) == scale if scale else abs(sum(raw)-1) < 1e-6
            teacher_ok = unit_sum and sum(raw) > 0
            # Identity proxy at the provider's declared decimal precision; no fabricated labels.
            target = raw if teacher_ok else None
            ex = dict(id=f"{row['id']}:{qid}", state_id=row['state_id'], family_id=row['family_id'],
                      split=row['split'], qid=qid, type=typ, candidate_ids=ids, gold_index=gold,
                      leaf_tokens=leaves, prefix_length=len(prefix),
                      teacher_raw_probs=raw, teacher_probs=target,
                      teacher_rounding=row['teacher'].get('rounding'), source=row, candidate_texts=texts)
            examples.append(ex)
            audit.append(dict(id=ex['id'], split=ex['split'], type=typ, k=len(ids),
                              teacher_raw_sum=sum(raw), teacher_proxy_usable=teacher_ok,
                              teacher_argmax_matches_gold=max(range(len(ids)), key=raw.__getitem__) == gold,
                              target_transform='identity_rounded_proxy' if teacher_ok else 'quarantined'))
    return examples, audit


class DecisionModel(nn.Module):
    def __init__(self, backbone, set_head):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.scalar = nn.Linear(hidden, 1)  # Nonzero random initialization avoids a dead first step.
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        self.set_head = set_head
        if set_head == 'attention':
            self.set_project = nn.Linear(hidden + 1, 128)
            self.set_attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
            self.set_output = nn.Linear(128, 1)
            # Only the final residual projection starts at zero; its upstream layers are nonzero.
            nn.init.zeros_(self.set_output.weight)
            nn.init.zeros_(self.set_output.bias)

    def encode_leaves(self, examples, pad_token):
        """Reference candidate encoding: every path carries its own copy of the prefix.

        Returns `(leaves, valid)` with `leaves` of shape `(questions, kmax, hidden)`
        and `valid` a boolean mask over real candidate slots. Positions beyond a
        question's candidate count are left at zero and marked invalid, which is
        exactly what the scalar head and the set head ignore downstream.
        """
        paths = [ids for ex in examples for ids in ex['leaf_tokens']]
        device = self.scalar.weight.device
        lengths = torch.tensor([len(ids) for ids in paths], device=device)
        width = int(lengths.max())
        tokens = torch.full((len(paths), width), pad_token, dtype=torch.long, device=device)
        for i, ids in enumerate(paths):
            tokens[i, :len(ids)] = torch.tensor(ids, device=device)
        attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
        hidden = self.backbone(input_ids=tokens, attention_mask=attention,
                               use_cache=False).last_hidden_state
        leaves = hidden[torch.arange(len(paths), device=device), lengths-1]
        return self.pack_leaves(examples, leaves)

    def pack_leaves(self, examples, leaves):
        """Reshape flat per-candidate leaf states into `(questions, kmax, hidden)`.

        A Boolean question has one semantic path but reports two candidate slots, so its
        single leaf is copied into both slots; `decision_head` maps a Boolean question to
        `[0, z]` regardless, so the copied value is never read. Every other question needs
        exactly one leaf per candidate. The remaining padding slots keep the zero fill the
        reference encoder always produced.
        """
        device = leaves.device
        expected = [len(ex['leaf_tokens']) for ex in examples]
        if leaves.shape[0] != sum(expected):
            raise ValueError('Leaf states must cover every question path exactly once')
        kmax = max(len(ex['candidate_ids']) for ex in examples)
        h = leaves.new_zeros((len(examples), kmax, leaves.shape[-1]))
        valid = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
        offset = 0
        for i, ex in enumerate(examples):
            paths, slots = expected[i], len(ex['candidate_ids'])
            if paths != slots and not (ex['type'] == 'boolean' and paths == 1):
                raise ValueError(f"{ex['id']}: {paths} leaf paths cannot fill {slots} candidate slots")
            h[i, :paths] = leaves[offset:offset+paths]
            if paths == 1 and slots > 1:
                h[i, 1:slots] = leaves[offset]
            valid[i, :slots] = True
            offset += paths
        return h, valid

    def decision_head(self, examples, h, valid, return_logits=False):
        """Score packed leaf states: shared scalar head plus the optional set head."""
        device = h.device
        kmax = valid.shape[1]
        h = self.norm(h)
        z = self.scalar(h).squeeze(-1).float()
        choice = torch.tensor([i for i, ex in enumerate(examples) if ex['type'] == 'choice'], device=device)
        if self.set_head == 'attention' and len(choice):
            log_k = valid[choice].sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
            u = self.set_project(torch.cat([h[choice], log_k.to(h.dtype)], dim=-1))
            mixed, _ = self.set_attention(u, u, u, key_padding_mask=~valid[choice], need_weights=False)
            delta = self.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
            z = z.index_add(0, choice, delta)
        # Boolean has one semantic path and one scalar, representing logits [0,z].
        out = []
        for i, ex in enumerate(examples):
            if ex['type'] == 'boolean':
                out.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax-2)))
            else:
                out.append(z[i])
        logits = torch.stack(out).masked_fill(~valid, -1e9)
        return (logits, valid) if return_logits else logits

    def forward(self, examples, pad_token, prefix_sharing=None):
        """Reference forward. `prefix_sharing` selects the shared-prefix encoder.

        `None` and `False` keep the historical path, which repeats the prefix in
        every candidate path. A `SharedPrefixEncoder` (or any callable with the
        same `encode` signature) computes each question's prefix once and
        evaluates only candidate suffixes against its cached keys and values.
        """
        if prefix_sharing:
            rows = prefix_sharing.encode_examples(self.backbone, examples, pad_token)
            if rows is None:
                # No question in this batch has a shareable prefix; the reference
                # encoding stays authoritative for the whole batch.
                h, valid = self.encode_leaves(examples, pad_token)
                return self.decision_head(examples, h, valid, return_logits=True)
            if any(not isinstance(row, torch.Tensor) or row.dim() != 1 for row in rows):
                raise ValueError('Shared-prefix encoder must return one rank-1 state per candidate')
            leaves = torch.stack(rows) if len(rows) > 1 else rows[0].unsqueeze(0)
            h, valid = self.pack_leaves(examples, leaves)
            return self.decision_head(examples, h, valid, return_logits=True)
        h, valid = self.encode_leaves(examples, pad_token)
        return self.decision_head(examples, h, valid, return_logits=True)


def loss_for(logits, examples, objective):
    target = torch.zeros_like(logits)
    for i, ex in enumerate(examples):
        if objective == 'gold':
            target[i, ex['gold_index']] = 1
        else:
            if ex['teacher_probs'] is None:
                raise ValueError('Quarantined teacher target entered training')
            target[i, :len(ex['candidate_ids'])] = torch.tensor(ex['teacher_probs'], device=logits.device)
    return -(target * logits.float().log_softmax(-1)).sum(-1)


def prediction_record(ex, logits):
    k = len(ex['candidate_ids'])
    z = logits[:k].float().cpu()
    keep = ('id', 'state_id', 'family_id', 'split', 'qid', 'type', 'candidate_ids',
            'gold_index', 'teacher_raw_probs', 'teacher_probs', 'teacher_rounding')
    record = {**{key:ex.get(key) for key in keep},
              'teacher_target_kind':'rounded_proxy_distribution' if ex.get('teacher_probs') is not None else None,
              'student_logits':z.tolist(), 'student_probs':z.softmax(-1).tolist()}
    # New pipeline may carry independent soft gold. A compatibility argmax is not
    # converted into an observed outcome; the exact target kind remains explicit.
    if 'gold_distribution_probs' in ex:
        record.update(gold_probs=ex.get('gold_probs'),
                      gold_distribution_probs=ex['gold_distribution_probs'],
                      gold_probs_kind=ex.get('gold_probs_kind'),
                      gold_label_kind=ex.get('gold_label_kind'),
                      teacher_target_error=ex.get('teacher_target_error'))
    return record


@torch.no_grad()
def evaluate(model, examples, pad_token, batch, path=None):
    model.eval()
    records = []
    for start in range(0, len(examples), batch):
        group = examples[start:start+batch]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z, _ = model(group, pad_token)
        records.extend(prediction_record(ex, logits) for ex, logits in zip(group, z))
    if path:
        Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    eligible = [r for r in records if r['teacher_probs'] is not None]
    return dict(
        accuracy=sum(max(range(len(r['student_probs'])),key=r['student_probs'].__getitem__)==r['gold_index'] for r in records)/len(records),
        gold_nll=sum(-math.log(max(r['student_probs'][r['gold_index']], 1e-30)) for r in records)/len(records),
        teacher_ce=sum(-sum(t*math.log(max(p,1e-30)) for t,p in zip(r['teacher_probs'],r['student_probs'])) for r in eligible)/len(eligible) if eligible else None,
        questions=len(records), teacher_eligible=len(eligible))


@torch.no_grad()
def frozen_native_baseline(lm, tokenizer, examples, batch, path, use_chat):
    """Native, untrained single-label logits baseline; conditional on valid labels only."""
    label_ids = []
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError('Baseline requires single-token label names')
        label_ids.append(ids[0])
    all_tokens = []
    for ex in examples:
        q = ex['source']['questions'][ex['qid']]
        texts = ['False', 'True'] if ex['type']=='boolean' else ex['candidate_texts']
        options = '\n'.join(f"{chr(65+i)}: {s}" for i,s in enumerate(texts))
        prompt = f"State:\n{ex['source']['state']}\nQuestion:\n{q['instructions']}\nOptions:\n{options}\nAnswer with only the option letter."
        if ex['type'] == 'boolean' and q.get('criteria'):
            prompt += '\nCriteria:\n' + json.dumps(q['criteria'], ensure_ascii=False, sort_keys=True)
        if use_chat:
            ids = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=True,
                                                add_generation_prompt=True, enable_thinking=False,
                                                return_dict=False)
        else:
            ids = tokenizer.encode(prompt+'\nAnswer:', add_special_tokens=False)
        all_tokens.append(ids)
    records = []
    lm.eval()
    for start in range(0,len(examples),batch):
        group, paths = examples[start:start+batch], all_tokens[start:start+batch]
        lengths = torch.tensor(list(map(len, paths)), device='cuda')
        tokens = torch.full((len(paths), int(lengths.max())), tokenizer.pad_token_id,device='cuda',dtype=torch.long)
        for i, ids in enumerate(paths):
            tokens[i,:len(ids)] = torch.tensor(ids,device='cuda')
        mask = torch.arange(tokens.shape[1],device='cuda')[None,:] < lengths[:,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h = lm.model(input_ids=tokens,attention_mask=mask,use_cache=False).last_hidden_state
            h = h[torch.arange(len(paths),device='cuda'),lengths-1]
            # Only the required output rows, not a full vocabulary projection.
            z = F.linear(h, lm.get_output_embeddings().weight[label_ids]).float()
        records.extend(prediction_record(ex, logits) for ex,logits in zip(group,z))
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))


@torch.no_grad()
def benchmark(model, examples, pad_token):
    model.eval()
    states = {}
    for ex in examples:
        states.setdefault(ex['state_id'], []).append(ex)
    groups = list(states.values())
    results = []
    for n in [1,4,8]:
        batch = sum(groups[:n], [])
        timings = []
        for repeat in range(8):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                model(batch,pad_token)
            torch.cuda.synchronize()
            if repeat >= 2:
                timings.append((time.perf_counter()-start)*1000)
        ms = sorted(timings)[len(timings)//2]
        results.append(dict(states=n,questions=len(batch),candidate_paths=sum(len(ex['leaf_tokens']) for ex in batch),
                            median_ms=ms,questions_per_second=1000*len(batch)/ms,repeats=len(timings)))
    return {'scope':'warm GPU inference including tensor assembly and device copies; excludes tokenization/network',
            'prefix_sharing':False,'autoregressive_decode_steps':0,'results':results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model', default='Qwen/Qwen3-0.6B-Base')
    p.add_argument('--revision', default='main')
    p.add_argument('--objective', choices=['teacher','gold'],default='teacher')
    p.add_argument('--set-head', choices=['none','attention'],default='attention')
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--head-steps',type=int,default=12)
    p.add_argument('--batch-questions',type=int,default=12)
    p.add_argument('--eval-every',type=int,default=24)
    p.add_argument('--max-length',type=int,default=512)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--backbone-lr',type=float,default=2e-5)
    p.add_argument('--head-lr',type=float,default=2e-4)
    p.add_argument('--skip-native-baseline',action='store_true')
    p.add_argument('--disable-native-triton',action='store_true',
                   help='Torch 2.14 process-local ATen fallback for hosts without Python headers')
    args = p.parse_args()
    if args.steps <= 0 or args.head_steps < 0 or args.eval_every <= 0 or args.batch_questions <= 0:
        p.error('steps/eval-every/batch-questions must be positive; head-steps must be nonnegative')
    if args.disable_native_triton:
        from torch._native import triton_utils
        triton_utils.deregister_op_overrides()
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    # FP32 parameter storage gives FP32 Adam states; forward uses BF16 autocast.
    lm = AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,dtype=torch.float32,
                                             attn_implementation='sdpa').cuda()
    lm.config.use_cache = False
    examples, audit = load_examples(args.input, tokenizer, args.max_length)
    dump(out/'target_audit.json',audit)
    bysplit = {split:[e for e in examples if e['split']==split] for split in ['train','dev','calibration','test','ood']}
    if any(not group for group in bysplit.values()):
        raise ValueError('All five splits must be nonempty')
    if args.objective == 'teacher' and not any(e['teacher_probs'] is not None for e in bysplit['dev']):
        raise ValueError('No eligible dev teacher targets')
    evaluation = sum([bysplit[s] for s in ['dev','calibration','test','ood']],[])
    if not args.skip_native_baseline:
        frozen_native_baseline(lm,tokenizer,evaluation,args.batch_questions,out/'native_baseline.jsonl',
                               use_chat=not args.model.endswith('-Base'))
    model = DecisionModel(lm.model,args.set_head).cuda()
    del lm
    train = [e for e in bysplit['train'] if args.objective=='gold' or e['teacher_probs'] is not None]
    if len(train) < args.batch_questions: raise ValueError('Too few valid training questions')
    config = {**vars(args),'resolved_model_revision':getattr(model.backbone.config,'_commit_hash',None),
              'data_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              'deps':{k:importlib.metadata.version(k) for k in ['torch','transformers','safetensors','numpy']},
              'gpu':torch.cuda.get_device_name(0),'train_questions':len(train),'all_train_questions':len(bysplit['train']),
              'parameter_storage':'float32','forward_autocast':'bfloat16','teacher_target':'identity rounded proxy, sum=1 only',
              'parallelism':'flat complete candidate paths in one backbone forward; no prefix sharing',
              'parameter_count':sum(t.numel() for t in model.parameters())}
    dump(out/'config.json',config)
    tokenizer.save_pretrained(out/'tokenizer')
    model.backbone.config.save_pretrained(out/'backbone_config')
    evaluate(model,evaluation,tokenizer.pad_token_id,args.batch_questions,out/'untrained_head.jsonl')
    head = [param for name,param in model.named_parameters() if not name.startswith('backbone.')]
    body = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW([{'params':body,'lr':args.backbone_lr}, {'params':head,'lr':args.head_lr}],
                                  weight_decay=0.01)
    logs=[]; best=float('inf'); best_step=None
    start=time.perf_counter()
    for step in range(args.head_steps+args.steps):
        warm = step < args.head_steps
        for param in body: param.requires_grad_(not warm)
        optimizer.param_groups[1]['lr'] = 1e-3 if warm else args.head_lr
        batch = random.sample(train,args.batch_questions)
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z,_ = model(batch,tokenizer.pad_token_id)
            loss = loss_for(z,batch,args.objective).mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        item={'step':step+1,'phase':'head' if warm else 'full','loss':float(loss.detach()),
              'elapsed_seconds':time.perf_counter()-start}
        if not warm and ((step+1-args.head_steps)%args.eval_every==0 or step+1==args.head_steps+args.steps):
            metrics=evaluate(model,bysplit['dev'],tokenizer.pad_token_id,args.batch_questions)
            item['dev']=metrics
            score=metrics['teacher_ce' if args.objective=='teacher' else 'gold_nll']
            if score is not None and score < best:
                best,best_step=score,step+1
                save_file({k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()},out/'best.safetensors')
        logs.append(item)
        if step%12==0 or 'dev' in item: print(json.dumps(item),flush=True)
    if best_step is None: raise RuntimeError('No checkpoint selected on dev')
    model.load_state_dict(load_file(out/'best.safetensors'))
    final=evaluate(model,evaluation,tokenizer.pad_token_id,args.batch_questions,out/'predictions.jsonl')
    timing=benchmark(model,bysplit['test'],tokenizer.pad_token_id)
    dump(out/'timing.json',timing)
    dump(out/'train_log.json',logs)
    dump(out/'summary.json',{'best_step':best_step,'selected_on':'dev teacher CE' if args.objective=='teacher' else 'dev gold NLL',
                            'training_seconds':time.perf_counter()-start,'final_all_eval':final,
                            'max_gpu_allocated_gb':torch.cuda.max_memory_allocated()/1e9})
    print(json.dumps({'done':str(out),'best_step':best_step,'eval':final}),flush=True)


if __name__ == '__main__':
    main()
