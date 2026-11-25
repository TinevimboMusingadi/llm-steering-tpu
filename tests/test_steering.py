import torch

from batched_steering.generate import Generator, decode
from batched_steering.model import chat, encode
from batched_steering.steer import BatchedSteer, SingleSteer
from batched_steering.sweep import Config, Sweep

PROMPTS = ["Write two sentences about a dog.", "Hi", "Write two sentences about the stars at night, please."]
N_NEW = 10


def _vec(lm, seed, scale=8.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(lm.d_model, generator=g) * scale


def _single_tokens(lm, prompts, setup):
    """Reference path: one setting at a time with SingleSteer, greedy, same generator shapes."""
    ids, mask, pos = encode(lm.tok, [chat(lm.tok, p) for p in prompts], lm.device)
    gen = Generator(lm, len(prompts), ids.shape[1], N_NEW)
    with SingleSteer(lm.layers) as s:
        setup(s)
        return gen(ids, mask, pos, temperature=0)


def test_generate_matches_hf(lm):
    ids, mask, pos = encode(lm.tok, [chat(lm.tok, p) for p in PROMPTS], lm.device)
    mine = Generator(lm, len(PROMPTS), ids.shape[1], N_NEW)(ids, mask, pos, temperature=0)
    ref = lm.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=N_NEW, min_new_tokens=N_NEW,
                            do_sample=False)[:, ids.shape[1]:]
    assert torch.equal(mine, ref)


def test_zero_push_is_identity(lm):
    ids, mask, pos = encode(lm.tok, [chat(lm.tok, p) for p in PROMPTS], lm.device)
    gen = Generator(lm, len(PROMPTS), ids.shape[1], N_NEW)
    base = gen(ids, mask, pos, temperature=0)
    with BatchedSteer(lm.layers, len(PROMPTS), lm.d_model, lm.device) as st:
        st.set_token_mask(mask)
        assert torch.equal(gen(ids, mask, pos, temperature=0), base)


def test_layer_numbering_matches_hidden_states(lm):
    """Layer L push adds exactly vec to the output of layers[L-1] (= hidden_states[L] unsteered) and nothing before it.
    (transformers v5 records hidden_states before user hooks run, so we read block outputs with our own hooks.)"""
    ids, mask, pos = encode(lm.tok, ["hello there"], lm.device)
    v, L = _vec(lm, 1), 2

    def outputs():
        seen = {}
        hs = [lm.layers[i].register_forward_hook(lambda m, inp, o, i=i: seen.__setitem__(i, o.clone()))
              for i in range(lm.n_layers)]
        full = lm.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True).hidden_states
        for h in hs:
            h.remove()
        return seen, full

    o0, full0 = outputs()
    with BatchedSteer(lm.layers, 1, lm.d_model, lm.device) as st:
        lm_mask = torch.zeros(1, lm.n_layers); lm_mask[0, L - 1] = 1
        st.load(vec=v[None], strength=torch.ones(1), layer_mask=lm_mask, prompt_scale=torch.ones(1))
        st.set_token_mask(mask)
        o1, _ = outputs()   # registered after the steer hooks -> sees steered outputs
    real = mask[0].bool()
    assert torch.allclose(o0[L - 1][0, real], full0[L][0, real], atol=1e-5)        # layer L == hidden_states[L]
    assert torch.allclose(o1[L - 2][0, real], o0[L - 2][0, real])
    assert torch.allclose(o1[L - 1][0, real] - o0[L - 1][0, real], v.expand(int(real.sum()), -1), atol=1e-4)
    assert torch.allclose(o1[L - 1][0, ~real], o0[L - 1][0, ~real])                # no push on left padding


def _configs(lm):
    v1, v2, r = _vec(lm, 1), _vec(lm, 2, 4.0), _vec(lm, 3)
    return [
        (Config("none"), lambda s: None),
        (Config("push L2 x1", vec=v1, strength=1.0, layers=(2,)), lambda s: s.add(2, v1)),
        (Config("push L3 x-2", vec=v2, strength=-2.0, layers=(3,)), lambda s: s.add(3, -2.0 * v2)),
        (Config("gen-only L1", vec=v1, strength=1.5, layers=(1,), gen_only=True), lambda s: s.add(1, 1.5 * v1, gen_only=True)),
        (Config("remove 1-4", unit=r, ablate=1.0, layers=(1, 2, 3, 4)), lambda s: s.remove_dir(r, 1.0)),
        (Config("remove 2-3 x0.6", unit=r, ablate=0.6, layers=(2, 3)), lambda s: s.remove_dir(r, 0.6, range(2, 4))),
    ]


def test_batched_equals_one_at_a_time(lm):
    pairs = _configs(lm)
    sw = Sweep(lm, batch_size=8, n_new=N_NEW, temperature=0)   # 6 configs x 3 prompts = 18 rows -> 3 chunks, 6 dummy rows
    try:
        batched = sw.generate([c for c, _ in pairs], PROMPTS)
    finally:
        sw.close()
    changed = 0
    for (cfg, setup), got in zip(pairs, batched):
        ref = decode(lm.tok, _single_tokens(lm, PROMPTS, setup))
        assert got == ref, cfg.name
        changed += got != batched[0]
    assert changed >= 3, "steering should change most outputs, otherwise the test proves nothing"


def test_batch_size_and_padding_do_not_matter(lm):
    pairs = _configs(lm)[:3]
    outs = []
    for B in (4, 16):
        sw = Sweep(lm, batch_size=B, n_new=N_NEW, temperature=0)
        try:
            outs.append(sw.generate([c for c, _ in pairs], PROMPTS))
        finally:
            sw.close()
    assert outs[0] == outs[1]


def test_fact_damage_matches_loop(lm):
    from batched_steering.score import facts, kl
    prompts, answers = facts(lm.tok)
    prompts, answers = prompts[:6], answers[:6]
    pairs = _configs(lm)
    sw = Sweep(lm, batch_size=8, n_new=N_NEW, temperature=0)
    try:
        before = sw.last_logprobs([Config()], prompts)[0]
        acc, kls = sw.fact_damage([c for c, _ in pairs], prompts, answers, before)
    finally:
        sw.close()
    for (cfg, setup), a, k in zip(pairs, acc, kls):
        with SingleSteer(lm.layers) as s:
            setup(s)
            lps = []
            for p in prompts:
                ids = lm.tok(p, return_tensors="pt").input_ids
                lps.append(lm.model(ids).logits[0, -1].float().log_softmax(-1))
            after = torch.stack(lps)
        assert abs(kl(before, after) - float(k)) < 1e-3, cfg.name
        ref_acc = sum(int(lp.argmax() == t) for lp, t in zip(after, answers)) / len(answers)
        assert abs(ref_acc - float(a)) < 1e-6, cfg.name


def test_buffers_are_reused(lm):
    st = BatchedSteer(lm.layers, 4, lm.d_model, lm.device)
    ptrs = {k: v.data_ptr() for k, v in st.cfg.items()}
    st.load(strength=torch.ones(4), vec=torch.randn(4, lm.d_model))
    st.load(strength=torch.zeros(4))
    assert ptrs == {k: v.data_ptr() for k, v in st.cfg.items()}
    st.remove()


def test_two_pushes_and_removal_match_loop(lm):
    """Mixing (E5): a push on all tokens + a decode-only push at other blocks, and push + removal together."""
    from batched_steering.loop import apply
    v1, v2, r = _vec(lm, 1), _vec(lm, 2, 4.0), _vec(lm, 3)
    configs = [
        Config("mix", vec=v1, strength=1.0, layers=(2,), extra=[Config(vec=v2, strength=2.0, layers=(3, 4), gen_only=True)]),
        Config("mix same block", vec=v1, strength=-1.0, layers=(1,), extra=[Config(vec=v2, strength=1.5, layers=(1,))]),
        Config("push + remove", vec=v1, strength=1.0, unit=r, ablate=0.7, layers=(2, 3)),
        Config("removal only + extra push", unit=r, ablate=1.0, layers=(1, 2), extra=[Config(vec=v1, strength=1.0, layers=(2,))]),
    ]
    sw = Sweep(lm, batch_size=8, n_new=N_NEW, temperature=0)
    try:
        batched = sw.generate(configs, PROMPTS)
    finally:
        sw.close()
    for c, got in zip(configs, batched):
        assert got == decode(lm.tok, _single_tokens(lm, PROMPTS, lambda s, c=c: apply(s, c))), c.name
    single = Sweep(lm, batch_size=8, n_new=N_NEW, temperature=0)
    try:
        alone = single.generate([Config("v1 only", vec=v1, strength=1.0, layers=(2,))], PROMPTS)[0]
    finally:
        single.close()
    assert batched[0] != alone, "the extra push should change the text"
