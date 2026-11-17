# -*- coding: utf-8 -*-
"""[cleanfix B2] t-SNE CSV for the domain-gap figure from the saved clean features (feats_KPSs0.npz, partgap_clean.py).
Features: layer2 (the paper's layer), seed 0; region 'window' = whole-window pooled (pool win), region 'airframe' = mean over the 8 airframe
points (pool body8). Per region separately: per-channel standardisation over sim + real, PCA 50, exact t-SNE (perplexity 30, 1000 iterations,
early exaggeration 12 for 250, PCA init, seed 0; own torch implementation - sklearn is not installed in env city).
Output: CSV with columns domain,region,x,y (domain = sim | real).
usage: python tsne_csv.py <feats.npz> <out.csv> [layer=layer2]"""
import sys, csv
import numpy as np, torch


def tsne(X, perp=30.0, iters=1000, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(X, dtype=torch.float64); n = len(X)
    D = torch.cdist(X, X) ** 2
    P = torch.zeros(n, n, dtype=torch.float64); logU = np.log(perp)
    for i in range(n):          # binary search of beta per point
        lo, hi, beta = 0.0, np.inf, 1.0; d = D[i].clone(); d[i] = np.inf
        for _ in range(60):
            p = torch.exp(-d * beta); s = p.sum()
            H = torch.log(s) + beta * (d[torch.isfinite(d)] * p[torch.isfinite(d)]).sum() / s
            if abs(float(H) - logU) < 1e-5:
                break
            if float(H) > logU:
                lo = beta; beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta; beta = (beta + lo) / 2
        P[i] = p / s
    P = (P + P.T) / (2 * n); P = torch.clamp(P, min=1e-12)
    Xc = X - X.mean(0); U, S, V = torch.pca_lowrank(Xc, q=2, center=False)
    Y = (Xc @ V[:, :2]); Y = Y / Y[:, 0].std() * 1e-4
    Y.requires_grad_(True)
    opt = torch.optim.SGD([Y], lr=max(n / 12.0 / 4, 50.0), momentum=0.5)
    for it in range(iters):
        ex = 12.0 if it < 250 else 1.0
        if it == 250:
            for g in opt.param_groups:
                g['momentum'] = 0.8
        q = 1.0 / (1.0 + torch.cdist(Y, Y) ** 2); q = q - torch.diag(torch.diag(q))
        Q = torch.clamp(q / q.sum(), min=1e-12)
        loss = (ex * P * torch.log(ex * P / Q)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return Y.detach().numpy()


def main():
    fz, out = sys.argv[1], sys.argv[2]; L = sys.argv[3] if len(sys.argv) > 3 else 'layer2'
    Z = np.load(fz); dom = Z['domain']
    rows = []
    for region, pool in (('window', 'win'), ('airframe', 'body8')):
        F = Z['%s__%s' % (L, pool)].astype(np.float64)
        sd = F.std(0); sd = np.maximum(sd, 0.1 * np.median(sd) + 1e-8)
        Fs = (F - F.mean(0)) / sd
        U, S, Vt = np.linalg.svd(Fs - Fs.mean(0), full_matrices=False)
        Y = tsne((Fs - Fs.mean(0)) @ Vt[:min(50, Vt.shape[0])].T)
        rows += [(str(d), region, float(y[0]), float(y[1])) for d, y in zip(dom, Y)]
        print(region, 'done', len(Y), flush=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['domain', 'region', 'x', 'y']); w.writerows(rows)
    print('wrote', out, len(rows), 'rows')


if __name__ == '__main__':
    main()
