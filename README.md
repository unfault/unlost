<h2 align="center">
  <br>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unfault/unlost/main/public/logo.png">
    <img alt="Unfault" src="https://raw.githubusercontent.com/unfault/unlost/main/public/logo.png" >
  </picture>
</h2>

<h4 align="center">Unlost - Agent orientation that prevents the babysitting tax</h4>

---

## Mission

Keep your agents oriented.

You know the drill:

- You ask an agent to rename a function. It renames it, then renames it again. Then again. You finally intervene: "stop renaming, just add a wrapper."
- An agent spent forty minutes "implementing" a feature. Turns out it only modified the README.
- You come back after the weekend. The agent made decisions you don't understand. Nobody remembers why.
- An agent says it finished. It didn't. It created a file that was never committed.

That's the babysitting tax. It adds up fast.

Unlost intercepts before you pay. It detects these failure modes and guides agents back on track:

| Failure Mode | What It Looks Like | What Unlost Does |
|--------------|-------------------|------------------|
| **Drift** | Agent thinks the system works one way. The code says otherwise. | Surfaces the contradiction before the agent compounds the error. |
| **Rediscovery** | You explain the same thing you explained last week. | Reminds the agent of what was already decided. |
| **Decision Conflict** | Agent starts implementing something that contradicts a project decision. | Flags the conflict and reminds the agent of the constraint. |
| **Retry Spiral** | Agent tries the same failed approach. Again. And again. | Catches the loop before another hour burns. |
| **False Progress** | Agent claims done. Verification would fail. | Detects the claim and flags it for review. |
| **Unbounded Horizon** | Agent wanders into unrelated side-quests. | Nudges back toward the original goal. |

## Use unlost with your favourite agent

### Claude Code

All your Claude Code projects, forever, with zero per-repo config:

```bash
unlost config agent claudecode --global
```

That's it. Unlost hooks into every Claude Code session, checks for friction before each prompt, injects guidance when something feels off, and quietly records what actually happened (intent, decision, rationale, next steps) into local capsules you can query anytime.

### OpenCode

All your OpenCode projects (global config):

```bash
unlost config agent opencode --global
```

Or one project at a time:

```bash
unlost config agent opencode --path .
```

Same deal. Unlost spots drift, catches false progress, and builds a local trail of decisions — without storing full transcripts.

## How it works

```
1. Your agent is about to send a prompt → unlost checks for friction
2. If something feels off → injects guidance before the agent goes off-track
3. After each exchange → extract a capsule (what we tried, why, next steps)
4. Capsules stay local → query anytime: "why did we do X?"
```

No transcripts. No external storage. Just small, queryable capsules that remember what your agents decided and why.

## Configure your LLM (optional, for better capsules)

Unlost uses a small LLM to extract structured capsules from agent exchanges. By default, it uses whatever LLM your agent is already configured with. You can override this for better results:

```bash
# Use Claude for extraction
unlost config llm anthropic --model claude-sonnet-4-5-20250929

# Or OpenAI
unlost config llm openai --model gpt-4o-mini
```

**What the LLM is fed:**
- The raw user → assistant exchange (just the text, not tool outputs)
- Recent capsule history from the workspace (to detect contradictions)

**What it produces:**
- A structured capsule with: category, intent, decision, rationale, next_steps, symbols, failure_mode

**Where it runs:**
- The LLM call goes through your configured provider (Anthropic, OpenAI, etc.)
- Capsule storage is entirely local — embeddings, the capsules themselves, and query history never leave your machine

## Recall & Query

Your agents built a memory trail. Here's how to query it:

```bash
# What did we decide here?
unlost recall src/http_proxy.rs

# Why did we rename the capsules table?
unlost query "why did we rename the capsules table?"

# Find everything about the proxy routing
unlost query --symbol proxy_request
```

## Install

Download the binary from [releases](https://github.com/unfault/unlost/releases).

## Dev

```bash
cargo test
cargo build
```

## Privacy First

Everything unlost stores stays on your machine:

- **Capsules** — Stored locally in `~/.local/share/unlost/workspaces/`
- **Embeddings** — Generated locally with fastembed
- **Query history** — Never leaves your disk

The only network call unlost makes is to the LLM provider you configure for extraction. That LLM sees only the exchange text (no tool outputs), and it produces a capsule that never goes back upstream.

## License

MIT. See [LICENSE](LICENSE) for details.

## Docs

- `agents/README.md` - Agent integrations
