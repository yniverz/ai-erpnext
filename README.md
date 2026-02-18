# AI ERPNext Assistant

A conversational AI interface to ERPNext financials. Describe what you bought, sold, or need to manage and the AI handles the ERPNext data entry for you.

![Chat UI](https://img.shields.io/badge/UI-Flask%20%2B%20Jinja-blue)

## Features

- **Natural language** — tell the AI what you did ("I bought office supplies for €120 from Amazon") and it creates the right documents
- **Clarification flow** — the AI asks follow-up questions when details are missing
- **Confirmation before writes** — nothing is created/updated/deleted without your explicit OK
- **All financial doctypes** — Sales Invoice, Purchase Invoice, Journal Entry, Payment Entry, Expense Claim, and more
- **Read & query** — ask about balances, list invoices, look up accounts
- **Multi-provider AI** — OpenAI, Anthropic (Claude), or local Ollama
- **Dark / light mode** chat UI

## Quick Start

### 1. Clone & install

```bash
cd ai-erpnext
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# edit .env with your ERPNext URL and AI provider keys
```

No ERPNext API keys needed — users log in with their ERPNext username/password directly in the browser.

### 3. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) — you'll see a login page. Sign in with your ERPNext credentials.

## Configuration

| Variable | Description |
|---|---|
| `ERPNEXT_URL` | Your ERPNext instance URL (e.g. `https://erp.example.com`) |
| `AI_PROVIDER` | `openai`, `anthropic`, or `ollama` |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) |
| `OPENAI_MODEL` | Model name (default: `gpt-4o`) |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using Claude) |
| `ANTHROPIC_MODEL` | Model name (default: `claude-sonnet-4-20250514`) |
| `OLLAMA_URL` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Ollama model name (default: `llama3`) |
| `FLASK_SECRET_KEY` | Random secret for Flask sessions |
| `FLASK_PORT` | Port to run on (default: `5000`) |

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Browser UI  │────▶│  Flask App   │────▶│  AI Agent    │
│  (Jinja/JS)  │◀────│  (app.py)    │◀────│  (agent.py)  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    │                         │
                              ┌─────▼─────┐           ┌──────▼──────┐
                              │ AI Provider│           │ ERPNext API │
                              │ (OpenAI /  │           │  Client     │
                              │  Claude /  │           │             │
                              │  Ollama)   │           └─────────────┘
                              └────────────┘
```

The AI model receives tool definitions for all ERPNext operations. When the user sends a message, the model decides which tools to call, the agent executes them against the ERPNext API, feeds results back to the model, and the loop repeats until the model produces a final text answer.

## Example Conversations

> **You:** I bought a new keyboard for €45 from Amazon yesterday  
> **AI:** I'll create a Purchase Invoice for that. A few questions:
> - Which company should this be under?
> - Which expense account? (e.g. "Office Equipment", "Office Supplies")
> - Do you have Amazon set up as a supplier already?

> **You:** Show me all unpaid sales invoices  
> **AI:** *lists invoices with amounts and due dates in a table*

> **You:** Create a journal entry to record a bank fee of €15  
> **AI:** Here's what I'll create: *(shows preview)* — shall I go ahead?

## Project Structure

```
ai-erpnext/
├── app.py              # Flask application & routes
├── agent.py            # AI agent with tool definitions & execution
├── ai_providers.py     # OpenAI / Anthropic / Ollama abstraction
├── erpnext_client.py   # ERPNext REST API client
├── config.py           # Configuration from .env
├── requirements.txt
├── .env.example
└── templates/
    └── index.html      # Chat UI
```
