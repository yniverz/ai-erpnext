"""
Agent — ties together the AI provider and the ERPNext tools.
Maintains per-session conversation history and handles the tool-call loop.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import date
from typing import Any

from ai_providers import get_provider, AIProvider
from erpnext_client import ERPNextClient

logger = logging.getLogger("ai-erpnext")

# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert ERPNext financial assistant. The current date is {today}.

Your job is to help the user manage their ERPNext financials through natural conversation.
You can create, read, update, and delete documents such as Sales Invoices, Purchase Invoices,
Journal Entries, Payment Entries, Expense Claims, and more.

**Workflow:**
1. The user describes something in natural language (e.g. "I bought office supplies for €50 from Amazon").
2. You determine what ERPNext action(s) are needed.
3. If any information is missing (supplier, account, cost center, date, etc.), ask the user.
4. Before creating or modifying any document, show the user a summary and ask for confirmation.
5. After confirmation, execute the action(s) via the available tools and report the result.

**Rules:**
- Always confirm before creating, updating, or deleting documents.
- When listing or querying, go ahead without confirmation.
- Use proper ERPNext doctype names and field names.
- If the user asks something outside of ERPNext financials, politely redirect.
- Format monetary values with proper currency symbols.
- When you don't know a value (like an account name), use the search/list tools to find it first.
- When creating documents, use the exact account/customer/supplier/item names from the context below.
- Present information in clean, readable markdown.

---

## Current ERPNext Context

Below is a live snapshot of the user's ERPNext instance. Use this to answer questions,
pick correct account names, auto-fill known values, and avoid unnecessary clarification
questions when the answer is already here.

{erpnext_context}
"""


def _format_context(ctx: dict) -> str:
    """Turn the context dict from ERPNext into readable text for the system prompt."""
    import json

    sections = []

    if ctx.get("companies"):
        lines = []
        for c in ctx["companies"]:
            name = c.get("name", c.get("company_name", "?"))
            currency = c.get("default_currency", "")
            country = c.get("country", "")
            lines.append(f"- **{name}** (currency: {currency}, country: {country})")
        sections.append("### Companies\n" + "\n".join(lines))

    if ctx.get("fiscal_years"):
        lines = []
        for fy in ctx["fiscal_years"]:
            lines.append(f"- {fy.get('name', '?')}: {fy.get('year_start_date', '?')} → {fy.get('year_end_date', '?')}")
        sections.append("### Fiscal Years\n" + "\n".join(lines))

    if ctx.get("accounts"):
        # Group accounts by root_type for readability
        by_root: dict[str, list] = {}
        for a in ctx["accounts"]:
            rt = a.get("root_type", "Other")
            by_root.setdefault(rt, []).append(a)
        lines = []
        for rt in ["Asset", "Liability", "Equity", "Income", "Expense", "Other"]:
            accts = by_root.get(rt, [])
            if not accts:
                continue
            lines.append(f"\n**{rt}:**")
            for a in accts:
                atype = f" ({a['account_type']})" if a.get("account_type") else ""
                lines.append(f"- {a.get('name', '?')}{atype}")
        sections.append("### Chart of Accounts (leaf accounts)" + "\n".join(lines))

    if ctx.get("cost_centers"):
        lines = [f"- {c.get('name', '?')}" for c in ctx["cost_centers"] if not c.get("is_group")]
        if lines:
            sections.append("### Cost Centers\n" + "\n".join(lines))

    if ctx.get("modes_of_payment"):
        lines = [f"- {m.get('name', '?')} (type: {m.get('type', '?')})" for m in ctx["modes_of_payment"]]
        sections.append("### Modes of Payment\n" + "\n".join(lines))

    if ctx.get("customers"):
        lines = [f"- {c.get('name', '?')} — {c.get('customer_name', '')}" for c in ctx["customers"]]
        sections.append("### Customers\n" + "\n".join(lines))

    if ctx.get("suppliers"):
        lines = [f"- {s.get('name', '?')} — {s.get('supplier_name', '')}" for s in ctx["suppliers"]]
        sections.append("### Suppliers\n" + "\n".join(lines))

    if ctx.get("items"):
        lines = []
        for it in ctx["items"]:
            rate = it.get("standard_rate", "")
            rate_str = f" @ {rate}" if rate else ""
            lines.append(f"- {it.get('name', '?')} — {it.get('item_name', '')}{rate_str}")
        sections.append("### Items\n" + "\n".join(lines))

    for label, key in [
        ("Recent Sales Invoices", "recent_sales_invoices"),
        ("Recent Purchase Invoices", "recent_purchase_invoices"),
        ("Recent Journal Entries", "recent_journal_entries"),
    ]:
        docs = ctx.get(key, [])
        if docs:
            lines = []
            for d in docs:
                parts = [d.get("name", "?")]
                for f in ["customer", "supplier", "title"]:
                    if d.get(f):
                        parts.append(d[f])
                for f in ["grand_total", "total_debit"]:
                    if d.get(f) is not None:
                        parts.append(str(d[f]))
                if d.get("status"):
                    parts.append(f"[{d['status']}]")
                if d.get("posting_date"):
                    parts.append(str(d["posting_date"]))
                lines.append("- " + " | ".join(parts))
            sections.append(f"### {label}\n" + "\n".join(lines))

    if not sections:
        return "(Could not fetch ERPNext context — the AI will query as needed.)"

    return "\n\n".join(sections)

# ── Tool definitions ─────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_documents",
        "description": "List documents of a given ERPNext doctype (e.g. 'Sales Invoice', 'Purchase Invoice', 'Journal Entry', 'Payment Entry', 'Expense Claim', 'Account', 'Item', 'Customer', 'Supplier'). Returns a list of matching documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "doctype": {
                    "type": "string",
                    "description": "ERPNext doctype name, e.g. 'Sales Invoice'",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to return. Default: ['name']",
                },
                "filters": {
                    "type": "object",
                    "description": "Filters as {field: value} or {field: ['operator', value]}",
                },
                "order_by": {
                    "type": "string",
                    "description": "Order by clause, e.g. 'creation desc'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results (default 20)",
                },
            },
            "required": ["doctype"],
        },
    },
    {
        "name": "get_document",
        "description": "Get the full details of a single ERPNext document by its doctype and name.",
        "parameters": {
            "type": "object",
            "properties": {
                "doctype": {"type": "string", "description": "ERPNext doctype name"},
                "name": {"type": "string", "description": "Document name/ID"},
            },
            "required": ["doctype", "name"],
        },
    },
    {
        "name": "create_document",
        "description": "Create a new ERPNext document. Provide the doctype and the document data as a JSON object. Use proper ERPNext field names. For child tables (like items), use the appropriate child table field name.",
        "parameters": {
            "type": "object",
            "properties": {
                "doctype": {"type": "string", "description": "ERPNext doctype name"},
                "data": {
                    "type": "object",
                    "description": "Document data with field names as keys",
                },
            },
            "required": ["doctype", "data"],
        },
    },
    {
        "name": "update_document",
        "description": "Update an existing ERPNext document.",
        "parameters": {
            "type": "object",
            "properties": {
                "doctype": {"type": "string", "description": "ERPNext doctype name"},
                "name": {"type": "string", "description": "Document name/ID"},
                "data": {
                    "type": "object",
                    "description": "Fields to update",
                },
            },
            "required": ["doctype", "name", "data"],
        },
    },
    {
        "name": "delete_document",
        "description": "Delete an ERPNext document.",
        "parameters": {
            "type": "object",
            "properties": {
                "doctype": {"type": "string", "description": "ERPNext doctype name"},
                "name": {"type": "string", "description": "Document name/ID"},
            },
            "required": ["doctype", "name"],
        },
    },
    {
        "name": "submit_document",
        "description": "Submit a draft ERPNext document (sets docstatus=1). Only for submittable doctypes like invoices, journal entries, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "doctype": {"type": "string", "description": "ERPNext doctype name"},
                "name": {"type": "string", "description": "Document name/ID"},
            },
            "required": ["doctype", "name"],
        },
    },
    {
        "name": "cancel_document",
        "description": "Cancel a submitted ERPNext document (sets docstatus=2).",
        "parameters": {
            "type": "object",
            "properties": {
                "doctype": {"type": "string", "description": "ERPNext doctype name"},
                "name": {"type": "string", "description": "Document name/ID"},
            },
            "required": ["doctype", "name"],
        },
    },
    {
        "name": "search_link",
        "description": "Search for a document by name (autocomplete-style). Useful to find accounts, items, customers, suppliers, etc. by partial name.",
        "parameters": {
            "type": "object",
            "properties": {
                "doctype": {"type": "string", "description": "Doctype to search in"},
                "query": {"type": "string", "description": "Search text"},
            },
            "required": ["doctype", "query"],
        },
    },
    {
        "name": "get_accounts",
        "description": "Get the Chart of Accounts. Optionally filter by company and/or root_type (Asset, Liability, Equity, Income, Expense).",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name (optional)"},
                "root_type": {
                    "type": "string",
                    "description": "Root type filter: Asset, Liability, Equity, Income, or Expense",
                    "enum": ["Asset", "Liability", "Equity", "Income", "Expense"],
                },
            },
        },
    },
    {
        "name": "get_companies",
        "description": "Get a list of all companies in ERPNext.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_customers",
        "description": "Get a list of customers.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results (default 50)"},
            },
        },
    },
    {
        "name": "get_suppliers",
        "description": "Get a list of suppliers.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results (default 50)"},
            },
        },
    },
    {
        "name": "get_items",
        "description": "Get a list of items.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results (default 50)"},
            },
        },
    },
    {
        "name": "call_method",
        "description": "Call any whitelisted ERPNext/Frappe server method. Use this for reports, special actions, or any API endpoint not covered by other tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Dotted method path, e.g. 'erpnext.accounts.utils.get_balance_on'",
                },
                "args": {
                    "type": "object",
                    "description": "Keyword arguments for the method",
                },
            },
            "required": ["method"],
        },
    },
]


# ── Tool executor ────────────────────────────────────────────────────


class Agent:
    def __init__(self):
        self.provider: AIProvider = get_provider()
        # session_id → message list
        self.sessions: dict[str, list[dict]] = {}
        # session_id → cached context string
        self._context_cache: dict[str, str] = {}

    def _get_erpnext_context(self, session_id: str, erp_client: ERPNextClient) -> str:
        """Fetch and cache the ERPNext context snapshot per session."""
        if session_id not in self._context_cache:
            logger.info("\033[35m[CONTEXT]\033[0m Fetching ERPNext context for session %s...", session_id[:8])
            try:
                ctx = erp_client.fetch_context()
                self._context_cache[session_id] = _format_context(ctx)
                logger.info("\033[35m[CONTEXT]\033[0m Loaded (%d chars)", len(self._context_cache[session_id]))
            except Exception as e:
                self._context_cache[session_id] = f"(Failed to fetch ERPNext context: {e})"
                logger.warning("\033[31m[CONTEXT]\033[0m Failed: %s", e)
        return self._context_cache[session_id]

    def refresh_context(self, session_id: str) -> None:
        """Force a re-fetch of ERPNext context for a session."""
        self._context_cache.pop(session_id, None)

    def _build_system_prompt(self, session_id: str, erp_client: ERPNextClient) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            today=date.today().isoformat(),
            erpnext_context=self._get_erpnext_context(session_id, erp_client),
        )

    def _ensure_session(self, session_id: str, erp_client: ERPNextClient) -> list[dict]:
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {"role": "system", "content": self._build_system_prompt(session_id, erp_client)}
            ]
        return self.sessions[session_id]

    def execute_tool(self, name: str, args: dict, erp_client: ERPNextClient) -> Any:
        """Route a tool call to the right ERPNext client method."""
        try:
            if name == "list_documents":
                return erp_client.get_list(
                    doctype=args["doctype"],
                    fields=args.get("fields"),
                    filters=args.get("filters"),
                    order_by=args.get("order_by"),
                    limit_page_length=args.get("limit", 20),
                )
            elif name == "get_document":
                return erp_client.get_doc(args["doctype"], args["name"])
            elif name == "create_document":
                return erp_client.create_doc(args["doctype"], args["data"])
            elif name == "update_document":
                return erp_client.update_doc(args["doctype"], args["name"], args["data"])
            elif name == "delete_document":
                return erp_client.delete_doc(args["doctype"], args["name"])
            elif name == "submit_document":
                return erp_client.submit_doc(args["doctype"], args["name"])
            elif name == "cancel_document":
                return erp_client.cancel_doc(args["doctype"], args["name"])
            elif name == "search_link":
                return erp_client.search_link(args["doctype"], args["query"])
            elif name == "get_accounts":
                return erp_client.get_accounts(
                    company=args.get("company"),
                    root_type=args.get("root_type"),
                )
            elif name == "get_companies":
                return erp_client.get_companies()
            elif name == "get_customers":
                return erp_client.get_customers(limit=args.get("limit", 50))
            elif name == "get_suppliers":
                return erp_client.get_suppliers(limit=args.get("limit", 50))
            elif name == "get_items":
                return erp_client.get_items(limit=args.get("limit", 50))
            elif name == "call_method":
                return erp_client.call_method(args["method"], **(args.get("args") or {}))
            else:
                return {"success": False, "error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def chat(self, session_id: str, user_message: str, erp_client: ERPNextClient) -> str:
        """Process a user message and return the assistant's response."""
        logger.info("\033[34m[CHAT]\033[0m User message: %s", user_message[:120])
        messages = self._ensure_session(session_id, erp_client)
        messages.append({"role": "user", "content": user_message})

        # Bind the erp_client into the tool executor
        def tool_executor(name: str, args: dict) -> Any:
            return self.execute_tool(name, args, erp_client)

        response = self.provider.chat(
            messages=messages,
            tools=TOOLS,
            tool_executor=tool_executor,
        )

        messages.append({"role": "assistant", "content": response})
        return response

    def get_history(self, session_id: str) -> list[dict]:
        """Return conversation history (excluding system prompt)."""
        msgs = self.sessions.get(session_id, [])
        return [m for m in msgs if m["role"] != "system"]

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self._context_cache.pop(session_id, None)
