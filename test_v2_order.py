"""test_v2_order.py — ONE 1-contract live NO order to verify the V2 migration.
Picks the cheapest-NO OTM BTC contract, places 1 NO contract, dumps the RAW V2
response (to confirm field shape), then checks positions to confirm a NO position
appeared (direction correctness). Bounded cost: < ~$1."""
import json, sys
import live_trading as lt
from live_signal import load_auth, fetch_contracts_for_nearest_expiry, fetch_recent_1m_candles

auth = load_auth()
assert auth is not None, "no auth"
bal0 = lt.get_balance(auth)
print(f"balance before: ${bal0}")

# spot from last 1m close
c = fetch_recent_1m_candles(lookback_bars=5, asset="BTC")
spot = float(c["close"].iloc[-1])
print(f"BTC spot ≈ {spot:.2f}")

cons = fetch_contracts_for_nearest_expiry(auth, spot=spot, asset="BTC")
if not cons:
    print("no liquid OTM contracts right now — try again shortly"); sys.exit(1)
# cheapest NO = highest yes_bid (NO cost = 1 - yes_bid)
cons.sort(key=lambda x: x["bid"], reverse=True)
t = cons[0]
yes_price = max(1, min(99, int(t["bid"] * 100)))      # floor(bid) → sell YES into bid → fills as NO buy
no_cost = (100 - yes_price) / 100.0
print(f"\nTEST contract: {t['ticker']}  strike={t['floor_strike']}  yes_bid={t['bid']:.2f} yes_ask={t['ask']:.2f}")
print(f"placing: NO x1 @ yes_price={yes_price}¢  → NO cost ≈ ${no_cost:.2f}")

res = lt.place_order(auth, ticker=t["ticker"], side="no", count=1, yes_price=yes_price)
print("\n=== RAW result (normalized envelope) ===")
print(json.dumps(res, indent=2, default=str))
print("\n=== RAW V2 response (raw key) ===")
print(json.dumps(res.get("raw", {}), indent=2, default=str))

print("\n=== open positions (confirm a NO/short-YES position appeared) ===")
pos = lt.get_open_positions(auth)
for p in pos:
    if p.get("ticker") == t["ticker"] or t["ticker"] in str(p):
        print("  MATCH:", json.dumps(p, default=str))
print(f"  (total open positions: {len(pos)})")
print(f"\nbalance after: ${lt.get_balance(auth)}  (was ${bal0})")
