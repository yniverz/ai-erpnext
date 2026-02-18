"""
ERPNext API Client — wraps the Frappe/ERPNext REST API.
Supports all financial doctypes: Sales Invoice, Purchase Invoice,
Journal Entry, Payment Entry, Expense Claim, and generic CRUD.

Authentication is cookie-based: the user logs in with their ERPNext
credentials and gets a requests.Session with the session cookie.
"""

import json
import requests
from typing import Any
from config import Config


class ERPNextClient:
    """Per-user ERPNext API client authenticated via cookie session."""

    def __init__(self, session: requests.Session | None = None):
        self.base_url = Config.ERPNEXT_URL
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    @classmethod
    def login(cls, username: str, password: str) -> "ERPNextClient":
        """
        Authenticate against ERPNext using username/password.
        Returns a new ERPNextClient with the authenticated session.
        Raises ValueError if login fails.
        """
        sess = requests.Session()
        resp = sess.post(
            f"{Config.ERPNEXT_URL}/api/method/login",
            json={"usr": username, "pwd": password},
        )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("message", "Login failed")
            except Exception:
                detail = resp.text or "Login failed"
            raise ValueError(detail)

        return cls(session=sess)

    def get_logged_in_user(self) -> str:
        """Return the email/username of the currently logged-in user."""
        resp = self.session.get(
            f"{self.base_url}/api/method/frappe.auth.get_logged_user"
        )
        if resp.status_code == 200:
            return resp.json().get("message", "")
        return ""

    # ── generic helpers ──────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/{path}"

    def _handle(self, resp: requests.Response) -> dict:
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            return {"success": False, "status": resp.status_code, "error": detail}
        try:
            return {"success": True, "data": resp.json().get("data", resp.json())}
        except Exception:
            return {"success": True, "data": resp.text}

    # ── CRUD ─────────────────────────────────────────────────────────

    def get_list(
        self,
        doctype: str,
        fields: list[str] | None = None,
        filters: dict | list | None = None,
        order_by: str | None = None,
        limit_page_length: int = 20,
        limit_start: int = 0,
    ) -> dict:
        """List documents of a given doctype."""
        params: dict[str, Any] = {
            "limit_page_length": limit_page_length,
            "limit_start": limit_start,
        }
        if fields:
            params["fields"] = json.dumps(fields)
        if filters:
            params["filters"] = json.dumps(filters)
        if order_by:
            params["order_by"] = order_by
        return self._handle(
            self.session.get(self._url(f"resource/{doctype}"), params=params)
        )

    def get_doc(self, doctype: str, name: str) -> dict:
        """Get a single document."""
        return self._handle(
            self.session.get(self._url(f"resource/{doctype}/{name}"))
        )

    def create_doc(self, doctype: str, data: dict) -> dict:
        """Create a new document."""
        return self._handle(
            self.session.post(self._url(f"resource/{doctype}"), json={"data": data})
        )

    def update_doc(self, doctype: str, name: str, data: dict) -> dict:
        """Update an existing document."""
        return self._handle(
            self.session.put(
                self._url(f"resource/{doctype}/{name}"), json={"data": data}
            )
        )

    def delete_doc(self, doctype: str, name: str) -> dict:
        """Delete a document."""
        return self._handle(
            self.session.delete(self._url(f"resource/{doctype}/{name}"))
        )

    def submit_doc(self, doctype: str, name: str) -> dict:
        """Submit a document (change docstatus to 1)."""
        return self.update_doc(doctype, name, {"docstatus": 1})

    def cancel_doc(self, doctype: str, name: str) -> dict:
        """Cancel a submitted document (change docstatus to 2)."""
        return self.update_doc(doctype, name, {"docstatus": 2})

    # ── Method calls ─────────────────────────────────────────────────

    def call_method(self, method: str, **kwargs) -> dict:
        """Call a whitelisted server method."""
        return self._handle(
            self.session.post(self._url(f"method/{method}"), json=kwargs)
        )

    # ── Convenience: financial queries ───────────────────────────────

    def get_balance_sheet(self, fiscal_year: str | None = None, company: str | None = None) -> dict:
        params = {}
        if fiscal_year:
            params["fiscal_year"] = fiscal_year
        if company:
            params["company"] = company
        return self.call_method(
            "erpnext.accounts.report.balance_sheet.balance_sheet.execute",
            **params,
        )

    def get_profit_and_loss(self, fiscal_year: str | None = None, company: str | None = None) -> dict:
        params = {}
        if fiscal_year:
            params["fiscal_year"] = fiscal_year
        if company:
            params["company"] = company
        return self.call_method(
            "erpnext.accounts.report.profit_and_loss_statement.profit_and_loss_statement.execute",
            **params,
        )

    def get_general_ledger(
        self,
        account: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        company: str | None = None,
        limit: int = 50,
    ) -> dict:
        params: dict[str, Any] = {"limit_page_length": limit}
        if account:
            params["account"] = account
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        if company:
            params["company"] = company
        return self.call_method(
            "erpnext.accounts.report.general_ledger.general_ledger.execute",
            **params,
        )

    def get_accounts(self, company: str | None = None, root_type: str | None = None) -> dict:
        """List Chart of Accounts."""
        filters: dict[str, Any] = {}
        if company:
            filters["company"] = company
        if root_type:
            filters["root_type"] = root_type
        return self.get_list(
            "Account",
            fields=["name", "account_name", "root_type", "account_type", "parent_account", "is_group"],
            filters=filters,
            limit_page_length=200,
        )

    def get_cost_centers(self, company: str | None = None) -> dict:
        filters: dict[str, Any] = {}
        if company:
            filters["company"] = company
        return self.get_list(
            "Cost Center",
            fields=["name", "cost_center_name", "parent_cost_center", "is_group"],
            filters=filters,
            limit_page_length=100,
        )

    def get_companies(self) -> dict:
        return self.get_list(
            "Company",
            fields=["name", "company_name", "default_currency", "country"],
            limit_page_length=100,
        )

    def get_suppliers(self, limit: int = 50) -> dict:
        return self.get_list(
            "Supplier",
            fields=["name", "supplier_name", "supplier_group", "country"],
            limit_page_length=limit,
        )

    def get_customers(self, limit: int = 50) -> dict:
        return self.get_list(
            "Customer",
            fields=["name", "customer_name", "customer_group", "territory"],
            limit_page_length=limit,
        )

    def get_items(self, limit: int = 50) -> dict:
        return self.get_list(
            "Item",
            fields=["name", "item_name", "item_group", "stock_uom", "standard_rate"],
            limit_page_length=limit,
        )

    def search_link(self, doctype: str, txt: str) -> dict:
        """Use ERPNext link search (auto-complete style)."""
        return self._handle(
            self.session.get(
                self._url("method/frappe.client.get_list"),
                params={
                    "doctype": doctype,
                    "filters": f'{{"name": ["like", "%{txt}%"]}}',
                    "fields": '["name"]',
                    "limit_page_length": 10,
                },
            )
        )

    # ── Context snapshot ─────────────────────────────────────────────

    def fetch_context(self) -> dict:
        """
        Fetch a snapshot of the current ERPNext setup so the AI has
        immediate context: companies, accounts, fiscal years,
        recent documents, customers, suppliers, items, cost centers.
        Returns a dict with each section; failed fetches are noted
        but don't block the rest.
        """
        ctx: dict[str, Any] = {}

        # Companies
        try:
            result = self.get_companies()
            ctx["companies"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["companies"] = []

        # Fiscal Years
        try:
            result = self.get_list(
                "Fiscal Year",
                fields=["name", "year_start_date", "year_end_date"],
                order_by="year_start_date desc",
                limit_page_length=5,
            )
            ctx["fiscal_years"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["fiscal_years"] = []

        # Chart of Accounts (leaf accounts only, up to 200)
        try:
            result = self.get_list(
                "Account",
                fields=["name", "account_name", "root_type", "account_type", "parent_account", "is_group"],
                filters={"is_group": 0},
                limit_page_length=200,
            )
            ctx["accounts"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["accounts"] = []

        # Cost Centers
        try:
            result = self.get_cost_centers()
            ctx["cost_centers"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["cost_centers"] = []

        # Customers (top 30)
        try:
            result = self.get_customers(limit=30)
            ctx["customers"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["customers"] = []

        # Suppliers (top 30)
        try:
            result = self.get_suppliers(limit=30)
            ctx["suppliers"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["suppliers"] = []

        # Items (top 30)
        try:
            result = self.get_items(limit=30)
            ctx["items"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["items"] = []

        # Recent Sales Invoices (last 10)
        try:
            result = self.get_list(
                "Sales Invoice",
                fields=["name", "customer", "grand_total", "status", "posting_date"],
                order_by="posting_date desc",
                limit_page_length=10,
            )
            ctx["recent_sales_invoices"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["recent_sales_invoices"] = []

        # Recent Purchase Invoices (last 10)
        try:
            result = self.get_list(
                "Purchase Invoice",
                fields=["name", "supplier", "grand_total", "status", "posting_date"],
                order_by="posting_date desc",
                limit_page_length=10,
            )
            ctx["recent_purchase_invoices"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["recent_purchase_invoices"] = []

        # Recent Journal Entries (last 10)
        try:
            result = self.get_list(
                "Journal Entry",
                fields=["name", "title", "total_debit", "posting_date", "voucher_type"],
                order_by="posting_date desc",
                limit_page_length=10,
            )
            ctx["recent_journal_entries"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["recent_journal_entries"] = []

        # Payment methods / Mode of Payment
        try:
            result = self.get_list(
                "Mode of Payment",
                fields=["name", "type"],
                limit_page_length=20,
            )
            ctx["modes_of_payment"] = result.get("data", []) if result.get("success") else []
        except Exception:
            ctx["modes_of_payment"] = []

        return ctx
