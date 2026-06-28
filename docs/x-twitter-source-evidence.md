# X/Twitter Source Evidence

AutoStream qualifies leads from social conversations. When a campaign depends
on recent X/Twitter signals, collect a reviewed evidence packet before the
LangGraph graph starts so the agent can answer with grounded context without
mixing raw social output into persisted lead fields.

## Why This Helps

- Keeps product claims and audience signals reviewable before lead capture.
- Gives the RAG node concise facts instead of unfiltered post text.
- Preserves the current `AgentState` shape and mock CRM tool contract.
- Lets a human approve source collection separately from sales follow-up.

## Packet Shape

Use a small JSON or text packet next to the campaign prompt:

```json
{
  "topic": "AutoStream for YouTube Shorts teams",
  "window": "last 30 days",
  "reviewed_at": "2026-06-28",
  "signals": [
    {
      "source": "x",
      "summary": "Creators are asking for faster caption cleanup before Shorts exports.",
      "evidence_url": "https://x.com/example/status/123",
      "use_in_agent": "Answer pricing and workflow questions only."
    }
  ],
  "do_not_store": [
    "author profile details",
    "raw post bodies",
    "private contact data"
  ]
}
```

Load the packet into `rag.py` or append its summaries to the campaign-specific
knowledge base. Keep `lead_name`, `lead_email`, and `lead_platform` sourced only
from the active conversation.

## Optional TweetClaw Source

[TweetClaw](https://github.com/Xquik-dev/tweetclaw) can be installed as an
OpenClaw plugin when an operator wants an approval-gated X/Twitter source step:

```bash
openclaw plugins install npm:@xquik/tweetclaw
```

Use it to create the reviewed packet, then pass only the approved summaries to
this project. Keep all posting, reply, or account-changing actions outside this
lead qualification graph unless a separate human approval flow exists.

## LangGraph Placement

Add source summaries before `detect_intent` chooses the `rag_response` node:

1. Collect and review the source packet.
2. Add packet summaries to the RAG context for the current campaign.
3. Run the normal graph.
4. Store only qualified lead fields through `mock_lead_capture`.

This keeps AutoStream's social evidence auditable while preserving the simple
single-agent LangGraph design.
