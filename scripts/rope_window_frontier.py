import numpy as np
BASE=1e6; R=65536
def freqs(d):
    k=np.arange(d//2); return BASE**(-2.0*k/d)
def usable_window(theta, amp):
    d=np.arange(R+1); s=np.cos(np.outer(d,theta))@amp
    rmax=np.maximum.accumulate(s[::-1])[::-1]
    wins=s[:-1]>rmax[1:]
    W=int(np.argmin(wins)) if not wins.all() else R
    # smallest-separation resolution: normalized per-step gap averaged near a target sep
    def ngap(D): return (s[D]-s[D+1])/s[0]
    return W, s[0], ngap
print("Frontier: lower theta_max widens W but kills near-Delta resolution.")
print("Need: W >= ~1922 (fixture max look-back), AND resolvable per-step gap at the")
print("min observed separation (~60) staying above fp32 floor ~1e-6.\n")
print(f"{'theta_max':>10} {'n_planes':>8} {'W':>7} {'peak':>7} {'ngap@60':>10} {'ngap@500':>10} {'ngap@1900':>11}")
d_head=256
theta_all=freqs(d_head)
for tmax in [0.3,0.1,0.03,0.01,0.003,0.001]:
    keep=(theta_all<tmax)&(theta_all>np.pi/R)
    th=theta_all[keep]; n=len(th)
    if n<2: continue
    hann=0.5-0.5*np.cos(2*np.pi*np.arange(n)/(n-1))+1e-3
    W,peak,ngap=usable_window(th,hann)
    print(f"{tmax:>10.3g} {n:>8} {W:>7} {peak:>7.2f} {ngap(60):>10.2e} {ngap(500):>10.2e} {ngap(1900):>11.2e}")
