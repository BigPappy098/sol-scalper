#!/usr/bin/env python3
"""Diagnose Hyperliquid API connectivity and account state.

Run this inside the RunPod container:
    python3 /root/sol-scalper/scripts/diagnose_hl.py
"""

import json
import os
import sys
import requests

import eth_account

def main():
    print("=" * 60)
    print("  Hyperliquid API Diagnostic")
    print("=" * 60)

    # 1. Check env vars
    mode = os.getenv("TRADING_MODE", "paper")
    pk = os.getenv("HL_PRIVATE_KEY", "")
    configured_addr = os.getenv("HL_WALLET_ADDRESS", "")

    print(f"\n[1] Environment")
    print(f"  TRADING_MODE     = {mode}")
    print(f"  HL_PRIVATE_KEY   = {'SET (' + pk[:6] + '...' + pk[-4:] + ')' if pk else 'NOT SET'}")
    print(f"  HL_WALLET_ADDRESS= {configured_addr or 'NOT SET'}")

    if not pk:
        print("\n  ERROR: HL_PRIVATE_KEY is not set. Cannot proceed.")
        sys.exit(1)

    # 2. Derive address from private key
    try:
        wallet = eth_account.Account.from_key(pk)
        derived_addr = wallet.address
        print(f"\n[2] Wallet")
        print(f"  Derived (API wallet) = {derived_addr}")
        if configured_addr and configured_addr.lower() != derived_addr.lower():
            print(f"  API wallet setup detected:")
            print(f"    API wallet (signs):    {derived_addr}")
            print(f"    Main wallet (funds):   {configured_addr}")
            print(f"  Bot will query main wallet for balance, sign with API wallet key.")
            # Use main wallet address for all balance queries below
            derived_addr = configured_addr
        elif configured_addr:
            print(f"  OK: configured address matches derived address")
    except Exception as e:
        print(f"\n  ERROR deriving wallet from private key: {e}")
        sys.exit(1)

    # 3. Test API connectivity
    if mode == "paper":
        base_url = "https://api.hyperliquid-testnet.xyz"
    else:
        base_url = "https://api.hyperliquid.xyz"

    print(f"\n[3] API Connectivity")
    print(f"  Base URL: {base_url}")

    # 3a. Test basic connectivity with meta endpoint
    try:
        resp = requests.post(f"{base_url}/info", json={"type": "meta"}, timeout=10)
        meta = resp.json()
        universe = meta.get("universe", [])
        sol_found = any(a["name"] == "SOL" for a in universe)
        print(f"  Meta endpoint: OK ({len(universe)} assets, SOL={'FOUND' if sol_found else 'NOT FOUND'})")
    except Exception as e:
        print(f"  Meta endpoint: FAILED ({e})")
        sys.exit(1)

    # 3b. Test user_state with derived address
    print(f"\n[4] Account State (derived address: {derived_addr})")
    try:
        resp = requests.post(
            f"{base_url}/info",
            json={"type": "clearinghouseState", "user": derived_addr},
            timeout=10,
        )
        state = resp.json()
        print(f"  Raw response keys: {list(state.keys())}")

        # Check margin summaries
        cross = state.get("crossMarginSummary", {})
        margin = state.get("marginSummary", {})

        print(f"\n  crossMarginSummary:")
        if cross:
            for k, v in cross.items():
                print(f"    {k}: {v}")
        else:
            print(f"    (empty/missing)")

        print(f"\n  marginSummary:")
        if margin:
            for k, v in margin.items():
                print(f"    {k}: {v}")
        else:
            print(f"    (empty/missing)")

        # Show account value from both
        equity_cross = float(cross.get("accountValue", 0)) if cross else 0
        equity_margin = float(margin.get("accountValue", 0)) if margin else 0
        print(f"\n  Equity (crossMarginSummary.accountValue): ${equity_cross:.2f}")
        print(f"  Equity (marginSummary.accountValue):       ${equity_margin:.2f}")

        # Check asset positions
        positions = state.get("assetPositions", [])
        print(f"\n  Open positions: {len(positions)}")
        for pos in positions:
            p = pos.get("position", pos)
            print(f"    {p.get('coin', '?')}: size={p.get('szi', 0)} entry={p.get('entryPx', 0)}")

        # Check withdrawable
        withdrawable = state.get("withdrawable", "?")
        print(f"\n  Withdrawable: {withdrawable}")

    except Exception as e:
        print(f"  user_state FAILED: {e}")

    # 3c. Check SPOT balance (separate from perps clearinghouse)
    print(f"\n[5] Spot Balance (derived address: {derived_addr})")
    try:
        resp = requests.post(
            f"{base_url}/info",
            json={"type": "spotClearinghouseState", "user": derived_addr},
            timeout=10,
        )
        spot = resp.json()
        balances = spot.get("balances", [])
        if balances:
            for bal in balances:
                coin = bal.get("coin", "?")
                total = bal.get("total", "0")
                print(f"    {coin}: {total}")
            # Check if USDC is in spot but perps is empty
            usdc_bal = next((b for b in balances if b.get("coin") == "USDC"), None)
            if usdc_bal and float(usdc_bal.get("total", 0)) > 0 and equity_cross == 0:
                print(f"\n  *** YOUR USDC IS IN SPOT, NOT PERPS! ***")
                print(f"  *** Transfer from Spot → Perps in the Hyperliquid UI to enable trading. ***")
        else:
            print(f"    (no spot balances)")
    except Exception as e:
        print(f"  Spot check FAILED: {e}")

    # 3e. Also try with configured address if different
    if configured_addr and configured_addr.lower() != derived_addr.lower():
        print(f"\n[6] Account State (configured address: {configured_addr})")
        try:
            resp = requests.post(
                f"{base_url}/info",
                json={"type": "clearinghouseState", "user": configured_addr},
                timeout=10,
            )
            state = resp.json()
            cross = state.get("crossMarginSummary", {})
            margin = state.get("marginSummary", {})
            equity_cross = float(cross.get("accountValue", 0)) if cross else 0
            equity_margin = float(margin.get("accountValue", 0)) if margin else 0
            print(f"  Equity (crossMarginSummary): ${equity_cross:.2f}")
            print(f"  Equity (marginSummary):      ${equity_margin:.2f}")
            if equity_cross > 0 or equity_margin > 0:
                print(f"\n  *** FOUND FUNDS on the configured address but NOT on the derived address! ***")
                print(f"  *** Your HL_PRIVATE_KEY does not correspond to HL_WALLET_ADDRESS. ***")
                print(f"  *** Use the private key for the wallet at {configured_addr} ***")
        except Exception as e:
            print(f"  FAILED: {e}")

    # 3f. Also check the SDK's user_state method directly
    print(f"\n[7] SDK user_state check")
    try:
        from hyperliquid.info import Info
        info = Info(base_url, skip_ws=True)
        sdk_state = info.user_state(derived_addr)
        print(f"  SDK response type: {type(sdk_state).__name__}")
        if sdk_state is None:
            print(f"  SDK returned None!")
        else:
            print(f"  SDK response keys: {list(sdk_state.keys()) if isinstance(sdk_state, dict) else 'not a dict'}")
            if isinstance(sdk_state, dict):
                cross = sdk_state.get("crossMarginSummary", {})
                margin = sdk_state.get("marginSummary", {})
                print(f"  SDK crossMarginSummary.accountValue: {cross.get('accountValue', 'MISSING')}")
                print(f"  SDK marginSummary.accountValue: {margin.get('accountValue', 'MISSING')}")
    except Exception as e:
        print(f"  SDK check FAILED: {e}")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
