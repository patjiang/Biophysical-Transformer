#!/usr/bin/env python
"""Residue-anchored span masking for stride-1 k-mer tokens.

Single-token masking is near-trivial under this tokenization: token *j*
covers residues ``(j, j+1, j+2)``, so its two neighbours jointly reveal
all three residues.  Span masking hides enough contiguous tokens that
at least one residue has *no* visible token covering it.

Masking *W* residues starting at anchor *a* requires masking tokens
``{a-2, ..., a+W-1}`` (i.e. ``W+2`` tokens).  Edge tokens still leak
their outer residues, which preserves a gradient of difficulty.

``replace_frac`` implements BERT-style corruption: a fraction of selected
spans are rewritten with residue-consistent random residues (labelled with
the truth) instead of ``[MASK]``.  Under stride-1 3-mers the corruption
must be residue-consistent or it's detectable from the token overlap alone.
This needs a ``KmerCodec`` built from ``data.seqs_to_tens``.

Usage::

    from span_masking import span_mask_tokens

    masked_seqs, mlm_labels = span_mask_tokens(
        seq_tensor, PAD_TOKEN_ID, MASK_TOKEN_ID,
        mask_rate=0.15, span_residues=3)
"""

import math

import torch
import torch.nn.functional as F

K = 3   # k-mer width; token j covers residues j, j+1, j+2
AAS = "ACDEFGHIKLMNPQRSTVWY"


class KmerCodec:
    """Invertible map between token ids and ``(r0, r1, r2)`` residue indices.

    Built once from ``data.seqs_to_tens``; raises if the encoding isn't a
    bijection over the 8000 standard 3-mers.
    """

    def __init__(self, ids_for_all_kmers):
        ids = torch.as_tensor(ids_for_all_kmers).reshape(-1).long()
        n = len(AAS) ** K
        if ids.numel() != n:
            raise ValueError(f"KmerCodec needs {n} ids, got {ids.numel()}")
        if not torch.equal(ids.sort().values, torch.arange(n)):
            raise ValueError(
                "data.seqs_to_tens does not map the 8000 standard 3-mers one-to-"
                f"one onto 0..7999 ({ids.unique().numel()} distinct ids, range "
                f"{int(ids.min())}..{int(ids.max())}). Residue-consistent "
                "replacement needs an invertible encoding.")
        res = torch.cartesian_prod(*[torch.arange(len(AAS))] * K)
        self.n = n
        self.table = torch.empty(n, K, dtype=torch.long)
        self.table[ids] = res
        self.enc = torch.empty(n, dtype=torch.long)
        self.enc[(res * torch.tensor([400, 20, 1])).sum(1)] = ids

    @classmethod
    def from_seqs_to_tens(cls, fn):
        """Build from the data module's tokenizer function."""
        kmers = [a + b + c for a in AAS for b in AAS for c in AAS]
        return cls(fn(kmers))

    def decode(self, ids):
        """Token ids -> ``(N, 3)`` residue indices."""
        return self.table[torch.as_tensor(ids).reshape(-1).long().cpu()]

    def encode(self, res):
        """``(N, 3)`` residue indices -> token ids."""
        res = torch.as_tensor(res).reshape(-1, K).long().cpu()
        return self.enc[res[:, 0] * 400 + res[:, 1] * 20 + res[:, 2]]


def _runs(row):
    """Inclusive ``(start, end)`` of each contiguous True run in a 1-D bool tensor."""
    idx = torch.nonzero(row, as_tuple=False).flatten().tolist()
    out, s, prev = [], None, None
    for i in idx:
        if s is None:
            s = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append((s, prev))
            s = prev = i
    if s is not None:
        out.append((s, prev))
    return out


def _corrupt_run(tokens, s, e, codec, generator):
    """Residue-consistent corruption of tokens ``s..e`` (inclusive).

    Residues fully hidden by the run are redrawn at random; residues shared
    with visible tokens are kept so every token in the string stays
    consistent.  Returns the new ids for ``s..e``, or ``None`` when the run
    hides no residue completely.
    """
    n = tokens.numel()
    known = (tokens >= 0) & (tokens < codec.n)
    res = torch.full((n + K - 1,), -1, dtype=torch.long)
    idx = torch.nonzero(known, as_tuple=False).flatten()
    dec = codec.decode(tokens[known])
    for p in range(K):
        res[idx + p] = dec[:, p]
    r = torch.arange(n + K - 1)
    hidden = (torch.clamp(r - (K - 1), min=0) >= s) & (torch.clamp(r, max=n - 1) <= e)
    if not hidden.any():
        return None
    new = torch.randint(0, len(AAS), (int(hidden.sum()),), generator=generator)
    res[hidden] = newdata
    t = torch.arange(s, e + 1)
    tri = torch.stack([res[t], res[t + 1], res[t + 2]], 1)
    if (tri < 0).any():
        return None
    return codec.encode(tri)


def _to_batch(body, pad_id):
    """Normalise input into a padded ``(B, L)`` LongTensor.

    Accepts a ``(B, L)`` tensor, a bare 1-D sequence, or a list/tuple of
    ragged 1-D tensors.  Returns ``(batch, restore)`` where ``restore()``
    maps a ``(B, L)`` result back to the caller's container, trimming each
    row to its true length.
    """
    if torch.is_tensor(body):
        if body.dim() == 1:
            return body.long().unsqueeze(0), lambda t: t.squeeze(0)
        if body.dim() == 2:
            return body.long(), lambda t: t
        raise ValueError(
            f"span_mask_tokens: expected a 1-D or 2-D token tensor, got shape "
            f"{tuple(body.shape)}")

    if isinstance(body, (list, tuple)):
        if len(body) == 0:
            raise ValueError("span_mask_tokens: empty batch")
        if isinstance(body[0], str):
            raise TypeError(
                "span_mask_tokens received k-mer strings "
                f"(e.g. {body[0]!r}), not token ids. Tokenise first:\n"
                "    seq_tensor = seqs_to_tens(seqs).unsqueeze(0)\n"
                "    masked, labels = span_mask_tokens(seq_tensor, ...)")
        rows = [r.long() if torch.is_tensor(r)
                else torch.as_tensor(r, dtype=torch.long) for r in body]
        if any(r.dim() != 1 for r in rows):
            raise ValueError(
                "span_mask_tokens: expected a sequence of 1-D token tensors, "
                f"got dims {[r.dim() for r in rows]}")
        lengths = [int(r.numel()) for r in rows]
        device = rows[0].device
        L = max(lengths)
        batch = torch.full((len(rows), L), pad_id, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            if r.numel():
                batch[i, :r.numel()] = r.to(device)

        def restore(t):
            out = [t[i, :n] for i, n in enumerate(lengths)]
            return out if isinstance(body, list) else tuple(out)

        return batch, restore

    raise TypeError(
        f"span_mask_tokens: unsupported input type {type(body).__name__}; "
        "pass a (B, L) tensor, a 1-D tensor, or a list of 1-D tensors")


def span_mask_tokens(body, pad_id, mask_id, mask_rate=0.15, span_residues=3,
                     generator=None, min_one=True, replace_frac=0.0, codec=None):
    """Mask contiguous token spans so whole residues are hidden.

    Parameters
    ----------
    body : Tensor or list
        K-mer ids with BOS already stripped.  Accepts ``(B, L)``, bare 1-D,
        or a list of ragged 1-D tensors.  Return mirrors the input shape.
    mask_rate : float
        Target fraction of non-PAD tokens to mask.
    span_residues : int
        Consecutive residues each anchor hides (each anchor masks
        ``span_residues + K - 1`` tokens).
    replace_frac : float
        Fraction of selected spans corrupted instead of masked.  Needs
        *codec*.
    codec : KmerCodec or None
        Required when ``replace_frac > 0``.

    Returns
    -------
    masked_body, labels
        Labels are ``-100`` where unmasked; corrupted positions are
        labelled with the true token.
    """
    if span_residues < 1:
        raise ValueError("span_residues must be >= 1")
    if replace_frac > 0 and codec is None:
        raise ValueError("replace_frac > 0 needs a KmerCodec (codec=...)")
    width = span_residues + K - 1

    body, restore = _to_batch(body, pad_id)
    B, L = body.shape
    valid = body != pad_id

    # Invert the overlap so mask_rate means what it says at any span width:
    # p_token = 1 - (1 - p_anchor)^width  =>  p_anchor = 1 - (1 - mask_rate)^(1/width)
    p_anchor = 1.0 - math.pow(max(1e-6, 1.0 - mask_rate), 1.0 / width)

    if generator is not None:
        r = torch.rand(body.shape, generator=generator).to(body.device)
    else:
        r = torch.rand(body.shape, device=body.device)
    anchors = (r < p_anchor) & valid

    # Dilate each anchor into its token window via a max-pool.
    a = anchors.float().unsqueeze(1)
    a = F.pad(a, (span_residues - 1, K - 1))
    sel = F.max_pool1d(a, kernel_size=width, stride=1).squeeze(1) > 0.5
    sel = sel[:, :L] & valid

    if min_one:
        empty = ~sel.any(dim=1)
        if empty.any():
            first = valid.float().argmax(dim=1)
            for b in torch.nonzero(empty, as_tuple=False).flatten().tolist():
                if valid[b].any():
                    j = int(first[b])
                    sel[b, j: min(j + width, L)] = valid[b, j: min(j + width, L)]

    labels = torch.full_like(body, -100)
    labels[sel] = body[sel]
    orig = body
    body = body.clone()
    body[sel] = mask_id
    if replace_frac > 0:
        for b in range(B):
            nz = torch.nonzero(valid[b], as_tuple=False)
            if nz.numel() == 0:
                continue
            n = int(nz.max()) + 1
            row = orig[b, :n].cpu()
            for s, e in _runs(sel[b, :n].cpu()):
                if float(torch.rand((), generator=generator)) >= replace_frac:
                    continue
                new = _corrupt_run(row, s, e, codec, generator)
                if new is not None:
                    body[b, s:e + 1] = new.to(body.device)
    return restore(body), restore(labels)


def leakage_report(body, sel, pad_id):
    """Count masked tokens that still have both neighbours visible (trivially recoverable)."""
    valid = body != pad_id
    left = F.pad(sel[:, :-1], (1, 0), value=False)
    right = F.pad(sel[:, 1:], (0, 1), value=False)
    recoverable = sel & ~left & ~right & valid
    n = int(sel.sum())
    return int(recoverable.sum()), n, (recoverable.sum() / max(n, 1)).item()


if __name__ == "__main__":
    PAD, MASK = 8000, 8001
    torch.manual_seed(0)
    B, L = 64, 400
    body = torch.randint(0, 7999, (B, L))
    body[:, 350:] = PAD

    print(f"{'scheme':<28}{'masked%':>9}{'fully recoverable':>20}")
    print("-" * 57)

    sel = (torch.rand(body.shape) < 0.15) & (body != PAD)
    rec, n, frac = leakage_report(body, sel, PAD)
    print(f"{'scattered p=0.15 (current)':<28}"
          f"{100 * n / int((body != PAD).sum()):>8.1f}%{frac:>19.1%}")

    for w in (1, 2, 3, 5):
        mb, lab = span_mask_tokens(body, PAD, MASK, mask_rate=0.15,
                                   span_residues=w)
        sel = lab != -100
        rec, n, frac = leakage_report(body, sel, PAD)
        print(f"{'span W=' + str(w) + f' ({w + K - 1} tokens)':<28}"
              f"{100 * n / int((body != PAD).sum()):>8.1f}%{frac:>19.1%}")

    print("\n  'fully recoverable' = masked tokens with both neighbours intact.")

    mb, lab = span_mask_tokens(body, PAD, MASK, mask_rate=0.15, span_residues=3)
    assert (mb[lab != -100] == MASK).all(), "masked positions not set to MASK"
    assert (lab[lab != -100] == body[lab != -100]).all(), "labels corrupted"
    assert (mb[body == PAD] == PAD).all(), "PAD was masked"
    assert (lab[body == PAD] == -100).all(), "PAD got a label"
    print("  invariants OK: MASK written, labels match originals, PAD untouched")

    print("\n  input forms")

    ragged = [torch.randint(0, 7999, (n,)) for n in (120, 45, 400, 7, 260)]
    mseqs, mlabels = span_mask_tokens(ragged, PAD, MASK, mask_rate=0.15,
                                      span_residues=1)
    assert isinstance(mseqs, list) and isinstance(mlabels, list)
    assert len(mseqs) == len(ragged)
    for orig, m, lb in zip(ragged, mseqs, mlabels):
        assert m.shape == orig.shape, f"length changed: {orig.shape} -> {m.shape}"
        assert lb.shape == orig.shape
        assert (m[lb == -100] == orig[lb == -100]).all(), "unmasked token altered"
        assert (m[lb != -100] == MASK).all()
        assert (lb[lb != -100] == orig[lb != -100]).all()
        assert (m != PAD).all(), "internal padding leaked into the output"
    print(f"    list of {len(ragged)} ragged tensors  -> list, lengths preserved "
          f"{[int(t.numel()) for t in mseqs]}")

    one = torch.randint(0, 7999, (300,))
    m1, l1 = span_mask_tokens(one, PAD, MASK, span_residues=1)
    assert m1.shape == one.shape and l1.shape == one.shape
    print(f"    bare 1-D tensor {tuple(one.shape)}        -> {tuple(m1.shape)}")

    m2, l2 = span_mask_tokens(body, PAD, MASK, span_residues=1)
    assert m2.shape == body.shape
    print(f"    padded batch {tuple(body.shape)}       -> {tuple(m2.shape)}")

    tup = span_mask_tokens(tuple(ragged), PAD, MASK, span_residues=1)
    assert isinstance(tup[0], tuple)
    print("    tuple in                      -> tuple out")

    for bad, why in ((torch.zeros(2, 3, 4, dtype=torch.long), "3-D tensor"),
                     ("not tokens", "str"), ([], "empty list")):
        try:
            span_mask_tokens(bad, PAD, MASK)
            print(f"    !! {why} did not raise")
        except (ValueError, TypeError) as e:
            print(f"    rejects {why:<22} {type(e).__name__}")
