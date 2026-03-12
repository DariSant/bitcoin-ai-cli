# Jules AI Agent Instructions: Bitcoin AI CLI Tool

## 1. Project Overview
* Goal: Build a free, command-line tool that combines live Bitcoin data, technical analysis, and AI to generate trading commentary.
* Language: Python.
* Key Integrations: Cryptocurrency exchange APIs (e.g., Binance, Kraken, or CoinGecko) and an AI/LLM API (e.g., OpenAI, Google Gemini).

## 2. User Profile & Communication Rules
* The human user has very little coding experience. 
* Never assume prior knowledge of Python environments, package managers, or advanced syntax.
* Use direct, clear language with literal instructions. Avoid overly complex metaphors.
* When providing terminal commands to run, install, or test code, provide the exact, copy-pasteable command.
* Always explain *why* a specific library or approach is being chosen before implementing it.

## 3. Coding Standards for this Project
* Prioritize readability and simplicity over clever, highly optimized, or complex code.
* Include extensive inline comments in the Python files explaining what each block of code does in plain English.
* Use clear, descriptive variable names (e.g., `bitcoin_current_price` instead of `btc_px`).
* Implement robust but simple error handling so the CLI tool prints a user-friendly message if an API fails, rather than a massive wall of red error text.

## 4. Autonomy & Workflow
* Minor Changes: You have autonomy to fix syntax errors, resolve simple bugs, and write localized functions independently.
* Major Changes: You MUST pause and ask for the user's approval before making major structural changes.
* Approval Triggers: Ask for permission before installing new external libraries, changing the folder structure, or completely rewriting a core feature.
* Always present a clear "Plan" of action for new features before writing the code.

## 5. Tooling

* Always use uv add <library> for installations.

## 6. Architecture

* Keep everything in app.py for now to minimize complexity.