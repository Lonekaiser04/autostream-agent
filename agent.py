"""
AutoStream Conversational AI Agent
Social-to-Lead Agentic Workflow using LangGraph + groq
"""

import json
import os
import re
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from tools import mock_lead_capture
from rag import retrieve_knowledge
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: Optional[str]                      # casual | inquiry | high_intent
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool                      # flag: are we in lead collection mode?


# ─────────────────────────────────────────────
# 2. LLM SETUP
# ─────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=1024,
)


# ─────────────────────────────────────────────
# 3. INTENT DETECTION NODE
# ─────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a SaaS video editing platform.

Classify the latest user message into EXACTLY ONE of these intents:
- casual        → Greetings, small talk, goodbye
- inquiry       → Questions about product, pricing, features, policies
- high_intent   → User is ready to sign up, buy, try, or start a plan

Respond with ONLY the intent label (lowercase, no punctuation).
Examples:
  "Hi there!" → casual
  "What does the Pro plan include?" → inquiry
  "I want to try the Pro plan for my YouTube channel" → high_intent
  "Sign me up!" → high_intent
"""

def detect_intent(state: AgentState) -> AgentState:
    """Classify user intent from the latest message."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    response = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_human),
    ])

    raw = response.content.strip().lower()
    if "high" in raw:
        intent = "high_intent"
    elif "inquiry" in raw or "product" in raw or "pric" in raw:
        intent = "inquiry"
    else:
        intent = "casual"

    return {**state, "intent": intent}


# ─────────────────────────────────────────────
# 4. RAG RETRIEVAL + RESPONSE NODE
# ─────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are AutoStream's friendly AI sales assistant.
AutoStream is a SaaS platform offering automated video editing for content creators.

Use the following knowledge base context to answer the user's question accurately.
If information is not in the context, say you'll connect them with the team.
Be concise, helpful, and end with a soft call-to-action when appropriate.

Context:
{context}
"""

def rag_response(state: AgentState) -> AgentState:
    """Retrieve relevant knowledge and generate a grounded response."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    context = retrieve_knowledge(last_human)

    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))]
    messages += state["messages"]

    response = llm.invoke(messages)
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {**state, "messages": new_messages}


# ─────────────────────────────────────────────
# 5. LEAD COLLECTION NODE
# ─────────────────────────────────────────────

LEAD_SYSTEM_PROMPT = """You are AutoStream's AI assistant helping qualify a potential customer.

The user has shown interest in signing up. Your job:
1. Collect their Name, Email, and Creator Platform (YouTube, Instagram, TikTok, etc.)
2. Ask for ONE missing piece of information at a time — don't ask for everything at once.
3. Be warm and conversational, not robotic.
4. If you already have some info (shown below), move on to what's missing.

Already collected:
- Name: {name}
- Email: {email}
- Platform: {platform}

If ALL THREE are collected, say exactly: "LEAD_COMPLETE" on a line by itself, then a warm closing message.
"""

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PLATFORM_KEYWORDS = ["youtube", "instagram", "tiktok", "twitter", "facebook",
                     "linkedin", "snapchat", "twitch", "x ", " x,"]

def _extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None

def _extract_platform(text: str) -> Optional[str]:
    lower = text.lower()
    for kw in PLATFORM_KEYWORDS:
        if kw in lower:
            return kw.strip().capitalize()
    return None

def _extract_name(text: str, existing_name: Optional[str]) -> Optional[str]:
    """Heuristic: if short phrase with no @ and no platform keyword, treat as name."""
    if existing_name:
        return existing_name
    text = text.strip()
    if len(text.split()) <= 4 and "@" not in text:
        lower = text.lower()
        if not any(kw in lower for kw in PLATFORM_KEYWORDS):
            return text.title()
    return None


def collect_lead(state: AgentState) -> AgentState:
    """Progressively collect lead info and call mock_lead_capture when complete."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    # Try to extract missing fields from the latest message
    name = state.get("lead_name") or _extract_name(last_human, state.get("lead_name"))
    email = state.get("lead_email") or _extract_email(last_human)
    platform = state.get("lead_platform") or _extract_platform(last_human)

    # Build LLM prompt
    messages = [
        SystemMessage(content=LEAD_SYSTEM_PROMPT.format(
            name=name or "Not yet provided",
            email=email or "Not yet provided",
            platform=platform or "Not yet provided",
        ))
    ] + state["messages"]

    response = llm.invoke(messages)
    reply = response.content

    # Check if lead is complete
    lead_captured = state.get("lead_captured", False)
    if "LEAD_COMPLETE" in reply and not lead_captured:
        mock_lead_capture(
            name=name or "Unknown",
            email=email or "Unknown",
            platform=platform or "Unknown",
        )
        lead_captured = True
        reply = reply.replace("LEAD_COMPLETE", "").strip()

    new_messages = state["messages"] + [AIMessage(content=reply)]
    return {
        **state,
        "messages": new_messages,
        "lead_name": name,
        "lead_email": email,
        "lead_platform": platform,
        "lead_captured": lead_captured,
        "collecting_lead": not lead_captured,
    }


# ─────────────────────────────────────────────
# 6. GREETING NODE
# ─────────────────────────────────────────────

GREETING_SYSTEM_PROMPT = """You are AutoStream's cheerful AI assistant.
AutoStream provides automated AI-powered video editing for content creators.
Respond warmly to greetings and let the user know you can help with:
- Pricing & plan details
- Product features
- Getting started

Keep it short and friendly (2–3 sentences max)."""

def greet(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=GREETING_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {**state, "messages": new_messages}


# ─────────────────────────────────────────────
# 7. ROUTING LOGIC
# ─────────────────────────────────────────────

def route(state: AgentState) -> str:
    """Decide next node after intent detection."""
    # If already collecting lead, stay in that flow
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return "collect_lead"
    intent = state.get("intent", "casual")
    if intent == "high_intent":
        return "collect_lead"
    elif intent == "inquiry":
        return "rag_response"
    else:
        return "greet"


# ─────────────────────────────────────────────
# 8. GRAPH CONSTRUCTION
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("greet", greet)
    graph.add_node("rag_response", rag_response)
    graph.add_node("collect_lead", collect_lead)

    graph.set_entry_point("detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        route,
        {
            "greet": "greet",
            "rag_response": "rag_response",
            "collect_lead": "collect_lead",
        }
    )

    graph.add_edge("greet", END)
    graph.add_edge("rag_response", END)
    graph.add_edge("collect_lead", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 9. MAIN CHAT LOOP
# ─────────────────────────────────────────────

def chat():
    print("=" * 60)
    print("  AutoStream AI Agent  |  Type 'quit' to exit")
    print("=" * 60)

    app = build_graph()

    state: AgentState = {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nAgent: Thanks for chatting! Have a great day 🎬")
            break

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        result = app.invoke(state)
        state = result

        # Print the last AI message
        last_ai = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            ""
        )
        print(f"\nAgent: {last_ai}")

        if state.get("lead_captured"):
            print("\n[System] ✅ Lead successfully captured! Ending session.")
            break


if __name__ == "__main__":
    chat()