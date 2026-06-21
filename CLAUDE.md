  # Claude Code Guidelines: Solana Meme Bot

## Operational & Security Rules
- NEVER hardcode private keys, seed phrases, or RPC secret URLs. 
- Use environment variables (`process.env` or `dotenv`) for all sensitive credentials.
- Ensure all file paths containing keys (e.g., `*.json`, `.env`) are strictly added to `.gitignore`.
- Before executing any live network commands, confirm whether the target is `devnet` or `mainnet-beta`.

## Technical Stack & Frameworks
- **Language**: TypeScript / Node.js
- **Solana Libraries**: `@solana/web3.js` (v1.x legacy or v2.x web3, stick to existing package signatures) and `@solana/spl-token`.
- **DEX Protocols**: Raydium AMM (V4/Clmm), Jupiter AG API (V6) for swaps and routing, or Pump.fun bonding curves.

## Code Style & Architecture
- **Transaction Safety**: Always implement robust retry logic with exponential backoff for transaction broadcasting.
- **Slippage & Priority Fees**: Explicitly calculate dynamic Compute Budget priority fees (`ComputeBudgetProgram.setComputeUnitPrice`) to avoid stuck transactions. Handle max slippage protection safely.
- **Error Handling**: Use aggressive `try/catch` blocks around RPC network calls. Log clear decoded error messages, not just raw anchor/program hex codes.

## Test & Build Commands
- Build project: `npm run build` or `tsc`
- Run bot in dry-run mode: `npm run dev:dry`
- Execute test suite: `npm test`
"Implement a clean TypeScript module that interacts with the Jupiter V6 API to swap SOL for a specified mint address. Ensure the module calculates dynamic priority fees using the latest blockhash, sets a 1% slippage guardrail, and exports a reusable function. Follow the safety rules in CLAUDE.md."
"Create a simulation wrapper for our buy/sell execution logic. It should read live token pools and prices via RPC, but instead of signing and broadcasting the actual transaction, it should log a detailed 'Dry Run Success' breakdown showing expected tokens bought, fees paid, and simulated slippage. Do not expose any private keys."
"Write an asynchronous listener script using a WebSocket connection (onLogsoronProgramAccountChange) to detect when a new token market is created on Raydium. Parse the transaction log to extract the new token mint address and pass it to our log framework."
# 1. Ensure your environment files are hidden from Claude's accidental file reads
echo ".env" >> .gitignore
echo "*keypair.json" >> .gitignore

# 2. Launch Claude with your prompt directly
claude "Review my current package.json and outline the architecture for our Solana trading bot"
"Create a simulation wrapper for our buy/sell execution logic in Python. The script should use AsyncClient from the solana.rpc.async_apipackage to fetch live pool data and prices. Instead of signing and broadcasting an actual transaction usingKeypair, it must intercept the logic before the send step. Log a structured, detailed 'Dry Run Success' dictionary or printout showing: expected tokens bought, exact lamports/fees paid, and simulated slippage. Follow the security and environment guardrails in CLAUDE.md."
## Technical Stack & Frameworks
- **Language**: Python 3.10+ (Asynchronous)
- **Solana Libraries**: `solana` (solana-py) and `solders` (for instructions, transactions, and types).
- **APIs**: Jupiter V6 HTTP API (via `httpx` or `aiohttp`) or Raydium Python integrations.

## Code Style & Architecture
- **Async First**: Always use `async/await` syntax for RPC requests and API fetching to prevent blocking the trading loop.
- **Type Hinting**: Use explicit Python type hints (`from typing import Optional, Dict`, etc.).
- **Error Handling**: Catch `SolanaException` and `RpcException` explicitly.

## Test & Build Commands
- Run bot simulation: `python main.py --dry-run`
- Run test suite: `pytest`
- Format check: `black .` or `flake8`
