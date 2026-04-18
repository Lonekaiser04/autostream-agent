# AutoStream AI Agent 🎬
### Social-to-Lead Agentic Workflow 

---

## Overview

An AI-powered conversational agent for **AutoStream** (a fictional SaaS video editing platform) that:
- Understands user intent (greeting / inquiry / high-intent)
- Answers product questions using RAG (local knowledge base)
- Detects high-intent users and progressively collects lead info
- Captures leads via a mock CRM tool

**Stack:** Python 3.9+ · LangGraph · LLaMA 3.1 via Groq · Local JSON RAG
---

## Project Structure

```
autostream_agent/
├── agent.py            # Main LangGraph agent & chat loop
├── rag.py              # RAG pipeline (knowledge retrieval)
├── tools.py            # mock_lead_capture tool + lead logger
├── knowledge.json # AutoStream pricing, plans, policies, FAQs
├── leads.json          # Auto-generated: persisted captured leads
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/lonekaiser04/autostream-agent.git
cd autostream-agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Or create a `.env` file:
```
GROQ_API_KEY=gsk_...
```

### 5. Run the agent
```bash
python agent.py
```


### 6. Example conversation
```
You: Hi, tell me about your pricing.
Agent: Hey there! AutoStream has two plans...

You: I want to try the Pro plan for my YouTube channel.
Agent: That's exciting! Could I get your name to get you started?

You: Alex Johnson
Agent: Thanks Alex! What's your email address?

You: alex@gmail.com
Agent: Perfect! And which platform — you mentioned YouTube, right?

You: Yes, YouTube
Agent: Awesome! You're all set...
[System] ✅ Lead successfully captured!
```

---

## WhatsApp Deployment (Webhook Architecture)

To deploy this agent on WhatsApp:

1. **Webhook Setup**: Configure a public endpoint (e.g., using ngrok or cloud function)
2. **Verify Token**: Implement Meta's webhook verification
3. **Message Flow**: 
   - Receive incoming message → Pass to agent → Return response
4. **State Persistence**: Store conversation state per phone number in Redis
5. **Template Messages**: Use WhatsApp templates for lead capture confirmation



## Architecture Explanation

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, inspectable state machines** — ideal for a multi-step lead qualification flow where the agent must know *exactly* what information has been collected at each turn. AutoGen's multi-agent conversation model is powerful but adds unnecessary complexity for a single-agent, stateful collection task.

### How State is Managed

The agent uses LangGraph's `StateGraph` with a typed `AgentState` dictionary that persists across all conversation turns. This state carries:

- **`messages`** — Full conversation history (via `add_messages` reducer, so it safely appends)
- **`intent`** — Detected intent of the latest user message
- **`lead_name / lead_email / lead_platform`** — Progressively filled lead fields
- **`collecting_lead`** — Flag that short-circuits intent detection once the lead flow starts
- **`lead_captured`** — Prevents double-firing the tool

Each graph node reads the current state and returns a *partial update*. LangGraph merges these updates, ensuring no data loss across turns. The graph entry point is always `detect_intent`, which then routes to `greet`, `rag_response`, or `collect_lead` based on intent — unless the `collecting_lead` flag is active, in which case the router skips intent detection and goes directly to `collect_lead`.

### RAG Pipeline

The RAG module loads `knowledge.json` at startup, flattens it into tagged text chunks, and performs keyword-scored retrieval at query time — no vector DB or embeddings required. This keeps the setup dependency-free while still grounding responses accurately.

---

## Evaluation Checklist

| Criterion | Implementation |
|---|---|
| Intent detection | 3-class LLM classifier in `detect_intent` node |
| RAG knowledge retrieval | Keyword-scored chunked retrieval in `rag.py` |
| State management | LangGraph `AgentState` TypedDict across all turns |
| Tool calling | `mock_lead_capture` fires only after all 3 fields collected |
| Code clarity | Modular: agent / rag / tools / knowledge_base separated |
| Real-world deployability | WhatsApp webhook architecture documented above |

---

## License

MIT — Built for the ServiceHive × Inflx ML Intern Assignment.