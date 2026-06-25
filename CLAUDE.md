
# Claude Code Guidelines: Solana Meme Bot

## Operational & Security Rules
- NEVER hardcode private keys, seed phrases, or RPC secret URLs.
- Use environment variables (`os.environ` / `python-dotenv`) for all sensitive credentials.
- Ensure all key files (e.g., `*keypair.json`, `wallet.json`, `.env`) are strictly added to `.gitignore`.
- Before executing any live network commands, confirm whether the target is `devnet` or `mainnet-beta`.
- All strategy modules default to dry-run; live execution requires an explicit flag.

## Technical Stack & Frameworks
- **Language**: Python 3.10+ (asynchronous)
- **Solana Libraries**: `solana` (solana-py) and `solders` (instructions, transactions, types).
- **APIs**: Jupiter V6 HTTP API (via `httpx` or `aiohttp`); Raydium / Pump.fun integrations as needed.

## Code Style & Architecture
- **Async First**: Use `async/await` for all RPC requests and API fetching to avoid blocking the trading loop.
- **Type Hinting**: Use explicit type hints (`from typing import Optional, Dict`, etc.).
- **Transaction Safety**: Implement robust retry logic with exponential backoff for transaction broadcasting.
- **Slippage & Priority Fees**: Calculate dynamic priority fees; enforce strict max-slippage protection.
- **Error Handling**: Catch `SolanaException` and `RpcException` explicitly. Log clear decoded error messages, not raw hex codes.

## Trading Strategy Principles
- Each strategy is a separate module under `strategies/`.
- **Profit-taking** (fires each tier once): at >=2x sell 50% of original, at >=5x sell 25% of original, at >=10x sell 15% of original, retain a 10% moonbag.
- **Hard exits** (sell entire remaining position, moonbag included) trigger on: rolling 15m volume dropping more than 70% from peak (i.e. current volume below 30% of peak), liquidity below 70% of entry liquidity, or a single sell exceeding 5% of pool liquidity.
- **Priority rule**: hard-exit checks run BEFORE profit-taking in the main loop. If a hard exit fires, exit fully and skip profit logic.

## MEV / Sandwich Protection
- Enforce strict slippage caps on every swap (default 1%); reject fills exceeding the cap.
- Use dynamic priority fees; consider Jito bundle submission for entries on contested launches.
- Prefer smaller position sizes on low-liquidity pools to reduce attack surface.

## Test & Build Commands
- Run bot simulation: `python main.py --dry-run`
- Run test suite: `pytest`
- Format / lint check: `black .` or `flake8`
## 4. WORKFLOW & INITIAL TASK
We will build Railgun Algo using a strict step-by-step workflow. 

Before writing any code or initializing files, your first task is to:
1. Acknowledge these execution-level guardrails and confirm you understand the multi-layered scope of Railgun Algo.
2. Recommend the optimal tech stack for an execution bot (e.g., Node.js/TypeScript for orchestration, or Rust if required for critical parsing speeds) and the best RPC connection methods (such as Helius or QuickNode).
3. Provide a high-level file tree architecture map showing exactly how the data stream feeds into the parser, triggers the router, passes through the risk controller, and reaches the signer.

Do not generate code blocks or transaction handlers yet. Present the structural blueprint first and ask for my approval to begin.
