"""
tools.py – Tool definitions for the AutoStream AI Agent.

Contains the mock_lead_capture tool that fires when a user is qualified.
In production, this would call a CRM API (HubSpot, Salesforce, etc.)
"""

import json
from datetime import datetime


# Mock Lead Capture Tool

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a lead in a CRM system.

    Args:
        name     : Full name of the lead
        email    : Email address of the lead
        platform : Creator platform (YouTube, Instagram, etc.)

    Returns:
        dict with lead details and timestamp
    """
    lead = {
        "name": name,
        "email": email,
        "platform": platform,
        "product_interest": "AutoStream Pro Plan",
        "source": "AI Agent – Social Chat",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "new_lead",
    }

    # ── Console output (simulates CRM write) ──
    print("\n" + "=" * 50)
    print("  🎯 LEAD CAPTURED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Name     : {lead['name']}")
    print(f"  Email    : {lead['email']}")
    print(f"  Platform : {lead['platform']}")
    print(f"  Interest : {lead['product_interest']}")
    print(f"  Source   : {lead['source']}")
    print(f"  Time     : {lead['timestamp']}")
    print("=" * 50 + "\n")

    # ── Optionally persist to a local JSON log ──
    _save_lead(lead)

    return lead


def _save_lead(lead: dict) -> None:
    """Append the lead to a local leads.json file (simulates DB write)."""
    import os

    leads_file = os.path.join(os.path.dirname(__file__), "leads.json")

    # Load existing leads
    if os.path.exists(leads_file):
        with open(leads_file, "r") as f:
            try:
                leads = json.load(f)
            except json.JSONDecodeError:
                leads = []
    else:
        leads = []

    leads.append(lead)

    with open(leads_file, "w") as f:
        json.dump(leads, f, indent=2)


# Tool registry (for LangChain tool calling)
TOOLS = {
    "mock_lead_capture": mock_lead_capture,
}


# Quick test

if __name__ == "__main__":
    result = mock_lead_capture(
        name="kaiser",
        email="kaiser@gmail.com",
        platform="YouTube"
    )
    print("Tool returned:", result)