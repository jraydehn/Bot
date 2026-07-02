"""
BTC 15m paper trade re-analysis with corrected breakeven WR formula.
"""

import pandas as pd
import numpy as np

# ── 1. Load & filter ──────────────────────────────────────────────────────────
df_raw = pd.read_csv('/Users/justindehn/Documents/ClaudeCode/kalshi_btc/results/paper_trades_btc15m.csv')
df = df_raw[
    (df_raw['asset'] == 'BTC') &
    (df_raw['decision'] == 'trade') &
    (df_raw['resolved_yes'].notna())
].copy()

print(f"Total rows in CSV: {len(df_raw)}")
print(f"Rows after filter (BTC, trade, resolved): {len(df)}")
print(f"Side breakdown:\n{df['side'].value_counts()}")
print(f"resolved_yes counts:\n{df['resolved_yes'].value_counts()}")
print()

# ── 2. Per-trade formulas ─────────────────────────────────────────────────────
df = df.reset_index(drop=True)
pm = df['p_market'].values
side = df['side'].values
df['fee'] = 0.07 * np.minimum(pm, 1 - pm)
df['be_wr'] = np.where(side == 'no', 1 - pm + df['fee'], pm + df['fee'])

# would_win is already in CSV (1=win, 0=loss); use would_pnl directly
df['win'] = df['would_win'].astype(float)

# ── helper ────────────────────────────────────────────────────────────────────
def bucket_stats(mask, label=''):
    if hasattr(mask, 'values'):
        mask = mask.values
    sub = df[mask]
    n = len(sub)
    if n == 0:
        return None
    wr     = sub['win'].mean() * 100
    pnl    = sub['would_pnl'].sum()
    be_wr  = sub['be_wr'].mean() * 100
    vs_be  = wr - be_wr
    return dict(label=label, n=n, WR=wr, PnL=pnl, BE_WR=be_wr, vs_BE=vs_be)

def print_row(r):
    if r is None:
        return
    print(f"  {r['label']:<55}  n={r['n']:>4}  WR={r['WR']:>6.1f}%  "
          f"BE={r['BE_WR']:>6.1f}%  vsB={r['vs_BE']:>+6.1f}pp  PnL=${r['PnL']:>+8.2f}")

def section(title):
    print()
    print('=' * 90)
    print(f"  {title}")
    print('=' * 90)

# ── 3. Overall stats ──────────────────────────────────────────────────────────
section("OVERALL")
r = bucket_stats(np.ones(len(df), dtype=bool), 'All trades')
print_row(r)
print(f"       avg_BE_WR = {df['be_wr'].mean()*100:.2f}%   WR-avg(BE_WR) = {(df['win'].mean()-df['be_wr'].mean())*100:+.2f}pp")

# ── 4. Signal bucket sweep ────────────────────────────────────────────────────

# ── side ──
section("SIDE")
for s in ['yes', 'no']:
    r = bucket_stats(df['side'] == s, f"side={s}")
    print_row(r)

# ── markov_regime_1h ──
section("MARKOV_REGIME_1H")
for reg in ['Bull', 'Bear', 'Sideways']:
    r = bucket_stats(df['markov_regime_1h'] == reg, f"markov_regime_1h={reg}")
    print_row(r)
# by side
for reg in ['Bull', 'Bear', 'Sideways']:
    for s in ['yes', 'no']:
        r = bucket_stats((df['markov_regime_1h'] == reg) & (df['side'] == s), f"  markov_1h={reg} side={s}")
        print_row(r)

# ── markov_regime_15m ──
section("MARKOV_REGIME_15M")
for reg in ['Bull', 'Bear', 'Sideways']:
    r = bucket_stats(df['markov_regime_15m'] == reg, f"markov_regime_15m={reg}")
    print_row(r)
for reg in ['Bull', 'Bear', 'Sideways']:
    for s in ['yes', 'no']:
        r = bucket_stats((df['markov_regime_15m'] == reg) & (df['side'] == s), f"  markov_15m={reg} side={s}")
        print_row(r)

# ── ema_bias ──
section("EMA_BIAS")
for v in sorted(df['ema_bias'].dropna().unique()):
    r = bucket_stats(df['ema_bias'] == v, f"ema_bias={v}")
    print_row(r)
for v in sorted(df['ema_bias'].dropna().unique()):
    for s in ['yes', 'no']:
        r = bucket_stats((df['ema_bias'] == v) & (df['side'] == s), f"  ema_bias={v} side={s}")
        print_row(r)

# ── dir_15m ──
section("DIR_15M")
for v in [1, -1]:
    r = bucket_stats(df['dir_15m'] == v, f"dir_15m={v}")
    print_row(r)
for v in [1, -1]:
    for s in ['yes', 'no']:
        r = bucket_stats((df['dir_15m'] == v) & (df['side'] == s), f"  dir_15m={v} side={s}")
        print_row(r)

# ── body_15m bands ──
section("BODY_15M")
bands_body = [
    ('<0.30',  df['body_15m'] < 0.30),
    ('0.30-0.60', (df['body_15m'] >= 0.30) & (df['body_15m'] < 0.60)),
    ('>=0.60', df['body_15m'] >= 0.60),
]
for label, mask in bands_body:
    r = bucket_stats(mask, f"body_15m {label}")
    print_row(r)
for label, mask in bands_body:
    for s in ['yes', 'no']:
        r = bucket_stats(mask & (df['side'] == s), f"  body_15m {label} side={s}")
        print_row(r)

# ── stoch_k_15m bands ──
section("STOCH_K_15M")
bands_stk15 = [
    ('<20',   df['stoch_k_15m'] < 20),
    ('20-40', (df['stoch_k_15m'] >= 20) & (df['stoch_k_15m'] < 40)),
    ('40-60', (df['stoch_k_15m'] >= 40) & (df['stoch_k_15m'] < 60)),
    ('60-80', (df['stoch_k_15m'] >= 60) & (df['stoch_k_15m'] < 80)),
    ('>=80',  df['stoch_k_15m'] >= 80),
]
for label, mask in bands_stk15:
    r = bucket_stats(mask, f"stoch_k_15m {label}")
    print_row(r)
for label, mask in bands_stk15:
    for s in ['yes', 'no']:
        r = bucket_stats(mask & (df['side'] == s), f"  stoch_k_15m {label} side={s}")
        print_row(r)

# ── stoch_k_1h bands ──
section("STOCH_K_1H")
bands_stk1h = [
    ('<30',   df['stoch_k_1h'] < 30),
    ('30-50', (df['stoch_k_1h'] >= 30) & (df['stoch_k_1h'] < 50)),
    ('50-70', (df['stoch_k_1h'] >= 50) & (df['stoch_k_1h'] < 70)),
    ('>=70',  df['stoch_k_1h'] >= 70),
]
for label, mask in bands_stk1h:
    r = bucket_stats(mask, f"stoch_k_1h {label}")
    print_row(r)
for label, mask in bands_stk1h:
    for s in ['yes', 'no']:
        r = bucket_stats(mask & (df['side'] == s), f"  stoch_k_1h {label} side={s}")
        print_row(r)

# ── p_market bands ──
section("P_MARKET")
bands_pm = [
    ('<0.40',    df['p_market'] < 0.40),
    ('0.40-0.50', (df['p_market'] >= 0.40) & (df['p_market'] < 0.50)),
    ('0.50-0.60', (df['p_market'] >= 0.50) & (df['p_market'] < 0.60)),
    ('0.60-0.70', (df['p_market'] >= 0.60) & (df['p_market'] < 0.70)),
    ('>=0.70',   df['p_market'] >= 0.70),
]
for label, mask in bands_pm:
    r = bucket_stats(mask, f"p_market {label}")
    print_row(r)
for label, mask in bands_pm:
    for s in ['yes', 'no']:
        r = bucket_stats(mask & (df['side'] == s), f"  p_market {label} side={s}")
        print_row(r)

# ── bp_4h bands ──
section("BP_4H (non-null only)")
df_bp = df[df['bp_4h'].notna()].copy().reset_index(drop=True)
print(f"  Rows with bp_4h: {len(df_bp)}")
bands_bp4h = [
    ('<0.35',    (df_bp['bp_4h'] < 0.35).values),
    ('0.35-0.65', ((df_bp['bp_4h'] >= 0.35) & (df_bp['bp_4h'] < 0.65)).values),
    ('>=0.65',   (df_bp['bp_4h'] >= 0.65).values),
]
for label, mask in bands_bp4h:
    sub = df_bp[mask]
    n = len(sub)
    if n == 0:
        continue
    wr   = sub['win'].mean() * 100
    pnl  = sub['would_pnl'].sum()
    be   = sub['be_wr'].mean() * 100
    vs_b = wr - be
    print(f"  {'bp_4h '+label:<55}  n={n:>4}  WR={wr:>6.1f}%  BE={be:>6.1f}%  vsB={vs_b:>+6.1f}pp  PnL=${pnl:>+8.2f}")
for label, mask in bands_bp4h:
    for s in ['yes', 'no']:
        sub = df_bp[mask & (df_bp['side'] == s).values]
        n = len(sub)
        if n < 5:
            continue
        wr   = sub['win'].mean() * 100
        pnl  = sub['would_pnl'].sum()
        be   = sub['be_wr'].mean() * 100
        vs_b = wr - be
        print(f"    {'bp_4h '+label+' side='+s:<53}  n={n:>4}  WR={wr:>6.1f}%  BE={be:>6.1f}%  vsB={vs_b:>+6.1f}pp  PnL=${pnl:>+8.2f}")

# ── Cross-tabs ────────────────────────────────────────────────────────────────
section("CROSS-TAB: markov_regime_1h × ema_bias")
for reg in ['Bull', 'Bear', 'Sideways']:
    for eb in sorted(df['ema_bias'].dropna().unique()):
        r = bucket_stats((df['markov_regime_1h'] == reg) & (df['ema_bias'] == eb),
                         f"1h={reg} ema={eb}")
        print_row(r)

section("CROSS-TAB: markov_regime_15m × ema_bias")
for reg in ['Bull', 'Bear', 'Sideways']:
    for eb in sorted(df['ema_bias'].dropna().unique()):
        r = bucket_stats((df['markov_regime_15m'] == reg) & (df['ema_bias'] == eb),
                         f"15m={reg} ema={eb}")
        print_row(r)

section("CROSS-TAB: dir_15m × ema_bias")
for d in [1, -1]:
    for eb in sorted(df['ema_bias'].dropna().unique()):
        r = bucket_stats((df['dir_15m'] == d) & (df['ema_bias'] == eb),
                         f"dir={d} ema={eb}")
        print_row(r)

# ── 5. Worst buckets (n≥12, most negative PnL) → rescue sub-slices ───────────
section("WORST BUCKETS — PnL < 0, n ≥ 12  (rescue sub-slice search)")

# Build all primary buckets
all_buckets = []

# side
for s in ['yes', 'no']:
    m = df['side'] == s
    r = bucket_stats(m, f"side={s}")
    if r: all_buckets.append((m, r))

# markov 1h
for reg in ['Bull', 'Bear', 'Sideways']:
    for s in ['yes', 'no']:
        m = (df['markov_regime_1h'] == reg) & (df['side'] == s)
        r = bucket_stats(m, f"markov_1h={reg} side={s}")
        if r: all_buckets.append((m, r))

# markov 15m
for reg in ['Bull', 'Bear', 'Sideways']:
    for s in ['yes', 'no']:
        m = (df['markov_regime_15m'] == reg) & (df['side'] == s)
        r = bucket_stats(m, f"markov_15m={reg} side={s}")
        if r: all_buckets.append((m, r))

# ema_bias
for v in sorted(df['ema_bias'].dropna().unique()):
    for s in ['yes', 'no']:
        m = (df['ema_bias'] == v) & (df['side'] == s)
        r = bucket_stats(m, f"ema_bias={v} side={s}")
        if r: all_buckets.append((m, r))

# stoch_k_15m
for label, mask in bands_stk15:
    for s in ['yes', 'no']:
        m = mask & (df['side'] == s)
        r = bucket_stats(m, f"stoch15m {label} side={s}")
        if r: all_buckets.append((m, r))

# stoch_k_1h
for label, mask in bands_stk1h:
    for s in ['yes', 'no']:
        m = mask & (df['side'] == s)
        r = bucket_stats(m, f"stoch1h {label} side={s}")
        if r: all_buckets.append((m, r))

# p_market
for label, mask in bands_pm:
    for s in ['yes', 'no']:
        m = mask & (df['side'] == s)
        r = bucket_stats(m, f"pm {label} side={s}")
        if r: all_buckets.append((m, r))

# dir_15m
for v in [1, -1]:
    for s in ['yes', 'no']:
        m = (df['dir_15m'] == v) & (df['side'] == s)
        r = bucket_stats(m, f"dir_15m={v} side={s}")
        if r: all_buckets.append((m, r))

# body_15m
for label, mask in bands_body:
    for s in ['yes', 'no']:
        m = mask & (df['side'] == s)
        r = bucket_stats(m, f"body15m {label} side={s}")
        if r: all_buckets.append((m, r))

# Filter to worst (PnL<0, n≥12) sorted by PnL ascending
worst = [(m, r) for m, r in all_buckets if r['n'] >= 12 and r['PnL'] < 0]
worst.sort(key=lambda x: x[1]['PnL'])

# Rescue signals to try
rescue_signals = [
    ('markov_1h=Bull',      df['markov_regime_1h'] == 'Bull'),
    ('markov_1h=Bear',      df['markov_regime_1h'] == 'Bear'),
    ('markov_1h=Sideways',  df['markov_regime_1h'] == 'Sideways'),
    ('markov_15m=Bull',     df['markov_regime_15m'] == 'Bull'),
    ('markov_15m=Bear',     df['markov_regime_15m'] == 'Bear'),
    ('markov_15m=Sideways', df['markov_regime_15m'] == 'Sideways'),
    ('ema_bias=-1',         df['ema_bias'] == -1),
    ('ema_bias=0',          df['ema_bias'] == 0),
    ('ema_bias=1',          df['ema_bias'] == 1),
    ('dir_15m=1',           df['dir_15m'] == 1),
    ('dir_15m=-1',          df['dir_15m'] == -1),
    ('stoch15m<20',         df['stoch_k_15m'] < 20),
    ('stoch15m 20-40',      (df['stoch_k_15m'] >= 20) & (df['stoch_k_15m'] < 40)),
    ('stoch15m 40-60',      (df['stoch_k_15m'] >= 40) & (df['stoch_k_15m'] < 60)),
    ('stoch15m 60-80',      (df['stoch_k_15m'] >= 60) & (df['stoch_k_15m'] < 80)),
    ('stoch15m>=80',        df['stoch_k_15m'] >= 80),
    ('stoch1h<30',          df['stoch_k_1h'] < 30),
    ('stoch1h 30-50',       (df['stoch_k_1h'] >= 30) & (df['stoch_k_1h'] < 50)),
    ('stoch1h 50-70',       (df['stoch_k_1h'] >= 50) & (df['stoch_k_1h'] < 70)),
    ('stoch1h>=70',         df['stoch_k_1h'] >= 70),
    ('body15m<0.30',        df['body_15m'] < 0.30),
    ('body15m>=0.60',       df['body_15m'] >= 0.60),
    ('pm<0.40',             df['p_market'] < 0.40),
    ('pm 0.40-0.50',        (df['p_market'] >= 0.40) & (df['p_market'] < 0.50)),
    ('pm 0.50-0.60',        (df['p_market'] >= 0.50) & (df['p_market'] < 0.60)),
    ('pm>=0.60',            df['p_market'] >= 0.60),
]

for bk_mask, bk_r in worst[:8]:
    print()
    print(f"  >>> BUCKET: {bk_r['label']}  "
          f"n={bk_r['n']}  WR={bk_r['WR']:.1f}%  BE={bk_r['BE_WR']:.1f}%  "
          f"vsB={bk_r['vs_BE']:+.1f}pp  PnL=${bk_r['PnL']:+.2f}")

    # Find any sub-slice that is profitable (PnL>0, WR>BE_WR, n≥6)
    rescues = []
    for sig_label, sig_mask in rescue_signals:
        sub = df[bk_mask & sig_mask]
        if len(sub) < 5:
            continue
        wr  = sub['win'].mean() * 100
        pnl = sub['would_pnl'].sum()
        be  = sub['be_wr'].mean() * 100
        vs_b = wr - be
        if pnl > 0 and vs_b > 0:
            rescues.append((sig_label, len(sub), wr, be, vs_b, pnl))

    blocks = []
    for sig_label, sig_mask in rescue_signals:
        sub = df[bk_mask & ~sig_mask]
        if len(sub) < 5:
            continue
        wr  = sub['win'].mean() * 100
        pnl = sub['would_pnl'].sum()
        be  = sub['be_wr'].mean() * 100
        vs_b = wr - be
        if pnl > 0 and vs_b > 0:
            blocks.append((sig_label, len(sub), wr, be, vs_b, pnl))

    if rescues:
        print(f"     RESCUE (keep when signal=True):")
        for sig, n, wr, be, vs_b, pnl in sorted(rescues, key=lambda x: -x[5])[:5]:
            print(f"       keep if {sig:<30}  n={n:>3}  WR={wr:.1f}%  BE={be:.1f}%  vsB={vs_b:+.1f}pp  PnL=${pnl:+.2f}")
    if blocks:
        print(f"     BLOCK (improve by removing signal=True, keeping remainder):")
        for sig, n, wr, be, vs_b, pnl in sorted(blocks, key=lambda x: -x[5])[:5]:
            print(f"       block when {sig:<28}  rem_n={n:>3}  WR={wr:.1f}%  BE={be:.1f}%  vsB={vs_b:+.1f}pp  PnL=${pnl:+.2f}")
    if not rescues and not blocks:
        print(f"     [no profitable rescue/block slice found with n≥5]")

# ── 6. Gate candidates table ──────────────────────────────────────────────────
section("GATE CANDIDATE SUMMARY  (n≥12, PnL<0, WR well below BE)")
print(f"  {'Condition':<55}  {'n':>4}  {'WR':>7}  {'BE_WR':>7}  {'vs_BE':>7}  {'PnL':>10}  Rescue")
print(f"  {'-'*55}  {'----':>4}  {'------':>7}  {'------':>7}  {'------':>7}  {'--------':>10}  ------")

worst_by_pnl = sorted(worst, key=lambda x: x[1]['PnL'])
for bk_mask, bk_r in worst_by_pnl[:15]:
    rescue_found = 'check above'
    print(f"  {bk_r['label']:<55}  {bk_r['n']:>4}  {bk_r['WR']:>6.1f}%  "
          f"{bk_r['BE_WR']:>6.1f}%  {bk_r['vs_BE']:>+6.1f}pp  ${bk_r['PnL']:>+8.2f}  {rescue_found}")

print()
print("Analysis complete.")
