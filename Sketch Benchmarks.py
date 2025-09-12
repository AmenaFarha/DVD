import numpy as np
import time
import hashlib
import math
import pandas as pd

SEED = 123
np.random.seed(SEED)

n = 5000          # number of vectors
d = 256           # dimension
r = 128           # projection dimension for RP
cs_t = 4          # Count-Sketch rows
cs_w = 4096       # Count-Sketch columns per row
cs_topk = 10      # recover top-k coordinates
hll_p = 14        # HLL precision -> m=2^p registers (~16K)
sig_top = 3       # signature uses top-3 abs indices + signs for HLL

def hash_to_int(x: bytes, seed: int = 0) -> int:
    h = hashlib.blake2b(x, digest_size=8, person=str(seed).encode())
    return int.from_bytes(h.digest(), byteorder="big", signed=False)

def sign_hash(key: int, seed: int) -> int:
    return 1 if (hash_to_int(key.to_bytes(8, "big"), seed) & 1) == 0 else -1

def bucket_hash(key: int, seed: int, buckets: int) -> int:
    return hash_to_int(key.to_bytes(8, "big"), seed) % buckets

# Random Projection

def random_projection_matrix(d: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((d, r)) / math.sqrt(r)
    return R

def random_projection(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    return X @ R

def rp_quality_check(X: np.ndarray, Y: np.ndarray, samples: int = 200) -> float:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(X.shape[0], size=min(samples, X.shape[0]), replace=False)
    x_norm2 = np.einsum("ij,ij->i", X[idx], X[idx])
    y_norm2 = np.einsum("ij,ij->i", Y[idx], Y[idx])
    rel_err = np.mean(np.abs(y_norm2 - x_norm2) / (x_norm2 + 1e-12))
    return float(rel_err)

# Count-Sketch

class CountSketch:
    def __init__(self, t, w, seeds_h, seeds_s):
        self.t = t
        self.w = w
        self.seeds_h = seeds_h
        self.seeds_s = seeds_s
        self.table = np.zeros((t, w), dtype=np.float64)

    @classmethod
    def create(cls, t: int, w: int, seed: int):
        rng = np.random.default_rng(seed)
        seeds_h = list(rng.integers(1, 2**31 - 1, size=t))
        seeds_s = list(rng.integers(1, 2**31 - 1, size=t))
        return cls(t, w, seeds_h, seeds_s)

    def update(self, key: int, value: float):
        for i in range(self.t):
            b = bucket_hash(key, self.seeds_h[i], self.w)
            s = sign_hash(key, self.seeds_s[i])
            self.table[i, b] += s * value

    def estimate(self, key: int) -> float:
        ests = []
        for i in range(self.t):
            b = bucket_hash(key, self.seeds_h[i], self.w)
            s = sign_hash(key, self.seeds_s[i])
            ests.append(s * self.table[i, b])
        return float(np.median(ests))

def count_sketch_build(X: np.ndarray, t: int, w: int, seed: int):
    S = X.sum(axis=0)  # coordinate sums
    cs = CountSketch.create(t, w, seed)
    for j in range(S.shape[0]):
        cs.update(j, float(S[j]))
    return cs, S

def count_sketch_recover_topk(cs: CountSketch, k: int, d: int):
    ests = np.array([cs.estimate(j) for j in range(d)])
    idx = np.argpartition(-np.abs(ests), kth=min(k, d - 1))[:k]
    idx = idx[np.argsort(-np.abs(ests[idx]))]
    return [(int(j), float(ests[j])) for j in idx]

def cs_quality_check(cs: CountSketch, true_sums: np.ndarray, k: int):
    ests = np.array([cs.estimate(j) for j in range(true_sums.shape[0])])
    k = min(k, true_sums.shape[0])
    est_top_idx = np.argpartition(-np.abs(ests), kth=k-1)[:k]
    true_top_idx = np.argpartition(-np.abs(true_sums), kth=k-1)[:k]
    overlap = len(set(est_top_idx.tolist()) & set(true_top_idx.tolist())) / float(k)
    denom = np.linalg.norm(true_sums) + 1e-12
    rel_l2 = np.linalg.norm(ests - true_sums) / denom
    return float(overlap), float(rel_l2)

# HyperLogLog

class HyperLogLog:
    def __init__(self, p: int):
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=np.uint8)

    @staticmethod
    def _leading_zeros_64(x: int) -> int:
        if x == 0:
            return 64
        return (x.bit_length() ^ 63)

    def add_hash(self, h: int):
        idx = (h >> (64 - self.p)) & ((1 << self.p) - 1)
        w = (h << self.p) & ((1 << 64) - 1)
        lz = self._leading_zeros_64(w)
        rho = lz + 1
        if rho > self.registers[idx]:
            self.registers[idx] = rho

    def add(self, obj_bytes: bytes, seed: int = 0):
        h = hash_to_int(obj_bytes, seed)
        self.add_hash(h)

    def estimate(self) -> float:
        inv_sum = np.sum(2.0 ** (-self.registers))
        if self.m == 16:
            alpha_m = 0.673
        elif self.m == 32:
            alpha_m = 0.697
        elif self.m == 64:
            alpha_m = 0.709
        else:
            alpha_m = 0.7213 / (1 + 1.079 / self.m)
        raw = alpha_m * (self.m ** 2) / inv_sum
        V = np.count_nonzero(self.registers == 0)
        if raw <= 2.5 * self.m and V > 0:
            return self.m * math.log(self.m / V)
        return float(raw)

def vector_signature(x: np.ndarray, k: int = 3) -> bytes:
    k = min(k, x.shape[0])
    idx = np.argpartition(-np.abs(x), kth=k-1)[:k]
    idx = idx[np.argsort(-np.abs(x[idx]))]
    signs = (x[idx] >= 0).astype(np.int8)
    payload = ",".join(f"{int(i)}:{int(s)}" for i, s in zip(idx, signs))
    return payload.encode()

# Data generation

def make_data(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    heavy_cols = rng.choice(d, size=8, replace=False)
    X[:, heavy_cols] += rng.normal(0, 3.0, size=(n, 8))
    return X

# Benchmark

def benchmark_all():
    X = make_data(n, d, seed=SEED)
    results = []

    # Random Projection
    t0 = time.perf_counter()
    R = random_projection_matrix(d, r, seed=SEED + 1)
    Y = random_projection(X, R)
    rp_time = time.perf_counter() - t0
    rp_err = rp_quality_check(X, Y, samples=200)
    results.append({
        "Method": "RandomProjection",
        "Runtime_sec": rp_time,
        #"Quality": f"mean rel. norm error ≈ {rp_err:.3f}"
    })

    # Count-Sketch
    t0 = time.perf_counter()
    cs, true_sums = count_sketch_build(X, cs_t, cs_w, seed=SEED + 2)
    cs_time = time.perf_counter() - t0
    overlap, rel_l2 = cs_quality_check(cs, true_sums, k=cs_topk)
    results.append({
        "Method": "CountSketch",
        "Runtime_sec": cs_time,
        #"Quality": f"top-{cs_topk} overlap={overlap:.2f}, rel-L2={rel_l2:.3f}"
    })

    # HyperLogLog
    t0 = time.perf_counter()
    hll = HyperLogLog(p=hll_p)
    for i in range(n):
        sig = vector_signature(X[i], k=sig_top)
        hll.add(sig, seed=SEED + 3)
    hll_time = time.perf_counter() - t0
    est = hll.estimate()
    rel_err = abs(est - n) / max(n, 1)
    results.append({
        "Method": "HyperLogLog",
        "Runtime_sec": hll_time,
        #"Quality": f"est={int(est)} (rel. err={rel_err:.3f})"
    })

    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    benchmark_all()
