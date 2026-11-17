#!/usr/bin/env python3
"""How full is the hashed tile-coding table? (load factor and collisions)

Mirrors the C hash in c/src/tr_linear.c exactly (FNV-1a over tile coordinates,
seeded per tiling, then mask or modulo by M) and enumerates every tile that
exists: tilings * (tiles+1)^d of them. Reports, for each feature count M:

  load factor       = tiles that exist / M
  shared fraction   = fraction of tiles whose weight slot is shared with another tile,
                      for legacy plain FNV (tile_dense = 2), the default FNV + fmix32 (tile_dense = 0) and an ideal
                      random hash (1 - exp(-load))
  dense fits        = whether tile_dense=1 is possible (M >= tilings * (tiles+1)^d)

Pair it with configs/tile_collisions.json, which learns with the same M values,
to plot learning performance against load factor: a curve the literature lacks
(docs/research_review.md).

    python3 scripts/tile_collisions.py --dims 2 --tiles 8 --tilings 8 --features 64 128 256 512 1024
"""
import argparse
import itertools
import math

M32 = 0xFFFFFFFF


def fnv_index(t, coords, M, fmix=False):
    h = (2166136261 ^ ((t * 0x9E3779B1) & M32)) & M32
    for c in coords:
        h ^= c
        h = (h * 16777619) & M32
    if fmix:                                    # tile_dense = 0 (default) in C; plain FNV is tile_dense = 2
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & M32
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & M32
        h ^= h >> 16
    return h & (M - 1) if (M & (M - 1)) == 0 else h % M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=2)
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--tilings", type=int, default=8)
    ap.add_argument("--features", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 4096])
    a = ap.parse_args()
    side = a.tiles + 1
    n_tiles = a.tilings * side ** a.dims
    print(f"d={a.dims}, tiles={a.tiles}, tilings={a.tilings}: {n_tiles} tiles exist")
    print(f"{'M':>6}  {'load':>6}  {'legacy FNV':>10}  {'default fmix':>12}  {'random hash':>11}  dense fits")
    for M in a.features:
        shared = []
        for fmix in (False, True):
            counts = {}
            for t in range(a.tilings):
                for coords in itertools.product(range(side), repeat=a.dims):
                    k = fnv_index(t, coords, M, fmix)
                    counts[k] = counts.get(k, 0) + 1
            shared.append(sum(c for c in counts.values() if c > 1) / n_tiles)
        rand = 1 - math.exp(-n_tiles / M)
        print(f"{M:6d}  {n_tiles / M:6.2f}  {shared[0]:10.2%}  {shared[1]:12.2%}  {rand:11.2%}  "
              f"{'yes' if M >= n_tiles else 'no'}")


if __name__ == "__main__":
    main()
