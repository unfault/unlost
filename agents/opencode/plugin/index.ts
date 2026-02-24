// OpenCode plugin: unlost/unloop
//
// Records conversations and detects friction loops.
// Spawns `unlost companion` as a child process for all logic.

import { spawn, type ChildProcess, execFileSync } from "node:child_process"
import { createInterface } from "node:readline"
import type { Plugin } from "@opencode-ai/plugin"

function findUnlostBinary(): string {
  return process.env.UNLOST_BIN || "unlost"
}

interface CheckResponse {
  note: string | null
  error?: string
}

interface RecordResponse {
  ok: boolean
  error?: string
}

type CompanionResponse = CheckResponse | RecordResponse | { ready: true }

export const UnlostPlugin: Plugin = async ({ client, directory }) => {
  let companion: ChildProcess | null = null
  const pendingRequests = new Map<number, (resp: CompanionResponse) => void>()
  let requestId = 0
  let ready = false
  const workingDirectory = directory || process.cwd()

  type Usage = {
    provider_id?: string
    model_id?: string
    cost?: number
    tokens?: {
      input?: number
      output?: number
      reasoning?: number
      cache?: {
        read?: number
        write?: number
      }
    }
  }

  // Track messages by ID -> { role, sessionId, text, usage? }
  const messageData = new Map<
    string,
    { role: string; sessionId: string; text: string; usage?: Usage }
  >()
  
  // Track session's last user/assistant for recording
  const sessionExchanges = new Map<string, { userMessageId: string | null; assistantMessageId: string | null }>()

  // Avoid recording the same assistant message multiple times.
  const recordedAssistantMessageIds = new Set<string>()

  // Safety net: record partial exchanges if we never get session.idle/step-finish.
  const partialFlushTimers = new Map<string, NodeJS.Timeout>()
  const PARTIAL_FLUSH_MS = 60_000

  function log(level: "debug" | "info" | "warn" | "error", message: string) {
    client.app.log({ body: { service: "unlost", level, message } })
  }

  function safeJson(v: unknown, max = 20_000) {
    let s: string
    try {
      s = JSON.stringify(v)
    } catch (e) {
      s = `<<unstringifiable: ${String(e)}>>`
    }
    return s.length > max
      ? s.slice(0, max) + `...<<truncated ${s.length - max} chars>>`
      : s
  }

  const dumpEvents = process.env.UNLOST_DEBUG_DUMP_EVENTS === "1"
  const recordTouchedPaths = process.env.UNLOST_RECORD_TOUCHED_PATHS !== "0"
  const recordGitPaths = process.env.UNLOST_RECORD_GIT_PATHS === "1"

  // Best-effort: track last session we saw activity for.
  // Some events (eg `file.edited`) may not carry session IDs.
  let lastSeenSessionId: string | null = null

  // Track touched paths per session. This is the preferred signal source for
  // file association in capsules. It comes from OpenCode events like `file.edited`
  // and tool execution hooks.
  const touchedPathsBySession = new Map<string, Set<string>>()

  // Track normalized tool outcomes per session.
  // Only lifecycle tools are captured (build, test, publish, git commit/push, etc.).
  // Each entry is a short fact string: "succeeded" or "failed: <snippet>".
  type ToolOutcome = { name: string; output: string }
  const toolOutcomesBySession = new Map<string, ToolOutcome[]>()

  // Allow-list: only capture bash tool calls whose command matches one of these patterns.
  // We match against the command content, not the tool name, since everything runs via bash.
  const TOOL_OUTCOME_ALLOW_RE = /\b(cargo\s+(build|test|check|clippy|publish|run)|go\s+(build|test|run|vet|install)|npm\s+(build|test|run|publish|install)|pnpm\s+(build|test|run|publish|install)|yarn\s+(build|test|run|publish)|bun\s+(build|test|run|publish|install)|pytest|python\s+-m\s+pytest|mvn\s+(package|install|test|deploy)|gradle\s+(build|test|publish)|make(\s+\w+)?|git\s+(commit|push|merge|rebase|tag))/

  // Patterns that indicate success in tool output.
  const TOOL_SUCCESS_RE = /\b(Finished|finished|PASSED|passed|succeeded|success|ok\b|OK\b|Done\b|done\b|published|pushed|merged|committed)/
  // Patterns that indicate failure in tool output.
  const TOOL_FAILURE_RE = /\b(error|Error|ERROR|FAILED|failed|FAIL\b|panic|panicked|fatal|Fatal)/

  function normalizeTooOutcome(toolName: string, output: string): ToolOutcome | null {
    const out = (output || "").trim()
    if (!out) return null

    if (TOOL_SUCCESS_RE.test(out)) {
      return { name: toolName, output: "succeeded" }
    }
    if (TOOL_FAILURE_RE.test(out)) {
      // Capture first 200 chars of output as the error snippet
      const snippet = out.slice(0, 200).replace(/\n+/g, " ").trim()
      return { name: toolName, output: `failed: ${snippet}` }
    }
    // Ambiguous — omit rather than guess
    return null
  }

  function addToolOutcome(sessionId: string | null, toolName: string, command: string, output: string) {
    if (!TOOL_OUTCOME_ALLOW_RE.test(command)) return
    const outcome = normalizeTooOutcome(toolName, output)
    if (!outcome) return

    const sid = resolveSessionId(sessionId)
    const list = toolOutcomesBySession.get(sid) || []
    list.push(outcome)
    toolOutcomesBySession.set(sid, list)
  }

  function drainToolOutcomes(sessionId: string): ToolOutcome[] {
    const sid = sessionId || UNKNOWN_SESSION_KEY
    const out = toolOutcomesBySession.get(sid) || []
    toolOutcomesBySession.delete(sid)
    // Also drain unknown-session outcomes
    if (sid !== UNKNOWN_SESSION_KEY) {
      const unknown = toolOutcomesBySession.get(UNKNOWN_SESSION_KEY) || []
      toolOutcomesBySession.delete(UNKNOWN_SESSION_KEY)
      return [...out, ...unknown]
    }
    return out
  }

  const UNKNOWN_SESSION_KEY = "*"

  function normalizeTouchedPath(p: string): string | null {
    let s = (p || "").trim()
    if (!s) return null

    // Normalize slashes.
    s = s.replace(/\\/g, "/")

    // Strip leading ./
    if (s.startsWith("./")) s = s.slice(2)

    // If absolute under workingDirectory, strip prefix.
    const wd = workingDirectory.replace(/\\/g, "/").replace(/\/$/, "")
    if (s.startsWith(wd + "/")) {
      s = s.slice(wd.length + 1)
    }

    // Drop obvious non-path junk.
    if (!s) return null
    if (s === "/") return null
    if (s.startsWith("http://") || s.startsWith("https://")) return null

    // Keep it workspace-relative.
    if (s.startsWith("/")) s = s.slice(1)
    if (!s) return null

    // Size guard.
    if (s.length > 260) return null

    return s
  }

  function resolveSessionId(sessionId: string | null): string {
    return sessionId || lastSeenSessionId || UNKNOWN_SESSION_KEY
  }

  function addTouchedPath(sessionId: string | null, p: string) {
    if (!recordTouchedPaths) return
    const norm = normalizeTouchedPath(p)
    if (!norm) return

    const sid = resolveSessionId(sessionId)
    const set = touchedPathsBySession.get(sid) || new Set<string>()
    set.add(norm)
    touchedPathsBySession.set(sid, set)
  }

  function drainTouchedPaths(sessionId: string): string[] {
    const sid = sessionId || UNKNOWN_SESSION_KEY

    const out: string[] = []
    const seen = new Set<string>()

    const takeFrom = (key: string) => {
      const set = touchedPathsBySession.get(key)
      if (!set) return
      touchedPathsBySession.delete(key)
      for (const p of set) {
        if (out.length >= 64) break
        if (!seen.has(p)) {
          seen.add(p)
          out.push(p)
        }
      }
    }

    takeFrom(sid)
    if (sid !== UNKNOWN_SESSION_KEY) {
      takeFrom(UNKNOWN_SESSION_KEY)
    }
    return out
  }

  function getGitTouchedPaths(): string[] {
    if (!recordGitPaths) return []

    try {
      const out = execFileSync(
        "git",
        ["-C", workingDirectory, "status", "--porcelain=v1"],
        { encoding: "utf8" }
      )
      const paths: string[] = []
      const seen = new Set<string>()

      for (const rawLine of out.split(/\r?\n/)) {
        const line = rawLine.trimEnd()
        if (!line) continue

        // porcelain v1: XY <path> OR ?? <path>
        // Rename: XY <from> -> <to>
        const rest = line.length >= 3 ? line.slice(3) : ""
        if (!rest) continue

        let p = rest
        const arrow = rest.lastIndexOf(" -> ")
        if (arrow !== -1) {
          p = rest.slice(arrow + 4)
        }
        p = p.trim()
        if (!p) continue

        // Best-effort normalization
        p = p.replace(/\\/g, "/")

        if (!seen.has(p)) {
          seen.add(p)
          paths.push(p)
        }

        if (paths.length >= 64) break
      }

      return paths
    } catch (e) {
      log("debug", `git touched_paths failed: ${String(e)}`)
      return []
    }
  }

  function schedulePartialFlush(sessionId: string) {
    const existing = partialFlushTimers.get(sessionId)
    if (existing) {
      clearTimeout(existing)
    }

    partialFlushTimers.set(
      sessionId,
      setTimeout(() => {
        partialFlushTimers.delete(sessionId)
        // Best-effort: record whatever we have so far.
        recordExchange(sessionId)
      }, PARTIAL_FLUSH_MS)
    )
  }

  function spawnCompanion() {
    const bin = findUnlostBinary()
    try {
      companion = spawn(bin, ["shim", "opencode"], {
        stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env },
      })

      companion.on("error", (err) => {
        log("error", `companion spawn error: ${err.message}`)
        companion = null
        ready = false
      })

      companion.on("exit", (code) => {
        if (code !== 0 && code !== null) {
          log("warn", `companion exited with code: ${code}`)
        }
        companion = null
        ready = false
      })

      const rl = createInterface({ input: companion.stdout! })
      rl.on("line", (line) => {
        try {
          const resp = JSON.parse(line) as CompanionResponse
          if ("ready" in resp && resp.ready === true) {
            ready = true
            log("info", "companion ready")
            return
          }
          // Resolve oldest pending request
          for (const [id, resolve] of pendingRequests) {
            pendingRequests.delete(id)
            resolve(resp)
            break
          }
        } catch {
          // Ignore malformed JSON
        }
      })

      companion.stderr?.on("data", (data: Buffer) => {
        log("debug", data.toString())
      })
    } catch (e) {
      const err = e as Error
      log("error", `failed to spawn companion: ${err.message}`)
      companion = null
    }
  }

  async function sendRequest<T extends CompanionResponse>(
    method: string,
    params: Record<string, unknown>,
    timeoutMs = 10000
  ): Promise<T> {
    if (!companion || !ready) {
      return { note: null, ok: false } as unknown as T
    }

    const id = ++requestId
    const req = JSON.stringify({ method, params }) + "\n"

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        pendingRequests.delete(id)
        resolve({ note: null, ok: false, error: "timeout" } as unknown as T)
      }, timeoutMs)

      pendingRequests.set(id, (resp) => {
        clearTimeout(timeout)
        resolve(resp as T)
      })

      try {
        companion!.stdin!.write(req)
      } catch (e) {
        const err = e as Error
        clearTimeout(timeout)
        pendingRequests.delete(id)
        resolve({ note: null, ok: false, error: err.message } as unknown as T)
      }
    })
  }

  function recordExchange(sessionId: string) {
    const exchange = sessionExchanges.get(sessionId)
    if (!exchange) return

    if (exchange.assistantMessageId && recordedAssistantMessageIds.has(exchange.assistantMessageId)) {
      // Already recorded this assistant message; clear and move on.
      sessionExchanges.set(sessionId, { userMessageId: null, assistantMessageId: null })
      return
    }

    const userMsg = exchange.userMessageId ? messageData.get(exchange.userMessageId) : null
    const assistantMsg = exchange.assistantMessageId ? messageData.get(exchange.assistantMessageId) : null

    const userText = userMsg?.text || ""
    const assistantText = assistantMsg?.text || ""
    const touchedPaths = drainTouchedPaths(sessionId)
    const gitTouchedPaths = getGitTouchedPaths()
    const touchedPathsToSend = [...touchedPaths, ...gitTouchedPaths].slice(0, 64)
    const toolCallsToSend = drainToolOutcomes(sessionId)

    if (!userText && !assistantText) return

    const usageToSend = assistantMsg?.usage || null
    // turn_key is a stable, content-free identity for this exchange.
    // The shim uses it to deduplicate via replayed.txt so that a plugin
    // restart never writes the same capsule twice to capsules.jsonl.
    const turnKey =
      exchange.userMessageId && exchange.assistantMessageId
        ? `${exchange.userMessageId}:${exchange.assistantMessageId}`
        : undefined

    log("info", `recording exchange: session=${sessionId} turn_key=${turnKey} user=${userText.slice(0, 50)}... assistant=${assistantText.slice(0, 50)}... tool_outcomes=${toolCallsToSend.length} usage=${safeJson(usageToSend)}`)

    sendRequest<RecordResponse>("record", {
      user_text: userText,
      assistant_text: assistantText,
      directory: workingDirectory,
      touched_paths: touchedPathsToSend,
      tool_calls: toolCallsToSend,
      agent_session_id: sessionId,
      usage: usageToSend,
      turn_key: turnKey,
    }).catch(() => {})

    if (exchange.assistantMessageId) {
      recordedAssistantMessageIds.add(exchange.assistantMessageId)
    }

    // Clear exchange tracking for next round
    sessionExchanges.set(sessionId, { userMessageId: null, assistantMessageId: null })
  }

  // Spawn companion on plugin init
  spawnCompanion()
  await new Promise((resolve) => setTimeout(resolve, 500))

  log("info", `plugin initialized, directory=${workingDirectory}`)
  if (dumpEvents) {
    log("info", "USAGE_DUMP_V1 enabled (UNLOST_DEBUG_DUMP_EVENTS=1)")
  }

  return {
    // Transform messages before sending to LLM - check for friction and inject warning
    "experimental.chat.messages.transform": async (_input, output) => {
      log("debug", "chat.messages.transform hook fired")
      
      const messages = output?.messages
      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        log("debug", "no messages to transform")
        return
      }

      // Find last user message
      let userText = ""
      let userMsgIndex = -1
      let userPartIndex = -1

      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]
        const role = msg.info?.role
        if (role !== "user") continue

        const parts = msg.parts
        if (!parts || !Array.isArray(parts)) continue

        for (let j = parts.length - 1; j >= 0; j--) {
          const part = parts[j]
          if (part.type === "text" && typeof part.text === "string" && part.text.trim()) {
            userText = part.text
            userMsgIndex = i
            userPartIndex = j
            break
          }
        }
        if (userText) break
      }

      if (!userText) {
        log("debug", "no user text found")
        return
      }

      log("info", `checking friction for: ${userText.slice(0, 100)}...`)

      // Check for friction
      const resp = await sendRequest<CheckResponse>("check", {
        text: userText,
        directory: workingDirectory,
      })

      log("debug", `friction check result: note=${resp.note ? "yes" : "no"}`)

      // Inject warning if friction detected
      if (resp.note && typeof resp.note === "string" && resp.note.trim()) {
        const part = messages[userMsgIndex].parts[userPartIndex]
        if (part.type === "text" && typeof part.text === "string" && !part.text.startsWith("[SYSTEM NOTE:")) {
          part.text = resp.note.trimEnd() + "\n\n" + part.text
          log("info", "injected friction warning")
        }
      }
    },

    // Capture message metadata and content via events
    event: async ({ event }) => {
      const e = event as any
      const etype = (e?.type || "") as string

      if (dumpEvents) {
        log("info", `RAW_EVENT ${etype} ${safeJson(e)}`)
      }

      // file.edited - track touched paths for better capsule symbols
      if (etype === "file.edited") {
        try {
          const props = e.properties as { file?: string; path?: string; sessionID?: string; sessionId?: string }
          const sid = (props.sessionID || props.sessionId || null) as string | null
          const p = props.file || props.path || ""
          if (p) addTouchedPath(sid, p)
        } catch {
          // ignore
        }
      }

      // file.watcher.updated - can include a batch of updated paths
      if (etype === "file.watcher.updated") {
        try {
          const props = e.properties as {
            file?: string
            files?: string[]
            paths?: string[]
            sessionID?: string
            sessionId?: string
          }
          const sid = (props.sessionID || props.sessionId || null) as string | null

          if (typeof props.file === "string" && props.file) {
            addTouchedPath(sid, props.file)
          }
          const list = (props.paths || props.files || []) as string[]
          for (const p of list) addTouchedPath(sid, p)
        } catch {
          // ignore
        }
      }

      // session.diff - includes file diffs (good touched-path signal)
      if (etype === "session.diff") {
        try {
          const props = e.properties as { sessionID?: string; sessionId?: string; diff?: Array<{ file?: string }> }
          const sid = (props.sessionID || props.sessionId || null) as string | null
          for (const d of props.diff || []) {
            if (d?.file) addTouchedPath(sid, d.file)
          }
        } catch {
          // ignore
        }
      }

      // tool.execute.before/after may include file paths in args/result
      if (etype === "tool.execute.before" || etype === "tool.execute.after") {
        try {
          const props = e.properties as {
            sessionID?: string
            sessionId?: string
            tool?: { name?: string; input?: unknown; args?: unknown }
            input?: unknown
            args?: unknown
            result?: unknown
          }
          const sid = (props.sessionID || props.sessionId || null) as string | null
          const candidates: unknown[] = []
          if (props.tool) {
            candidates.push(props.tool.input, props.tool.args)
          }
          candidates.push(props.input, props.args, props.result)

          const visit = (v: unknown) => {
            if (!v) return
            if (typeof v === "string") {
              // Avoid scanning arbitrary strings; accept only path-like patterns.
              if (v.includes("/") || v.includes("\\") || v.includes(".")) {
                addTouchedPath(sid, v)
              }
              return
            }
            if (Array.isArray(v)) {
              for (const x of v) visit(x)
              return
            }
            if (typeof v === "object") {
              const o = v as Record<string, unknown>
              for (const k of [
                "path",
                "file",
                "file_path",
                "filePath",
                "filepath",
                "filename",
                "target",
              ]) {
                if (o[k]) visit(o[k])
              }
              return
            }
          }

          for (const c of candidates) visit(c)

          // Capture normalized tool outcomes for lifecycle commands (build/test/publish/git).
          // Only on after-events where we have the result.
          if (etype === "tool.execute.after") {
            try {
              const toolName = (props.tool?.name || (props as Record<string, unknown>).tool as string || "bash") as string
              // Extract command from args — bash tool puts command in args.command or args itself
              let command = ""
              const args = props.args || props.tool?.args || props.input || props.tool?.input
              if (typeof args === "string") {
                command = args
              } else if (args && typeof args === "object") {
                const a = args as Record<string, unknown>
                command = (typeof a.command === "string" ? a.command : typeof a.cmd === "string" ? a.cmd : "") as string
              }
              // Extract output string
              let output = ""
              if (typeof props.result === "string") {
                output = props.result
              } else if (props.result && typeof props.result === "object") {
                const r = props.result as Record<string, unknown>
                output = typeof r.output === "string" ? r.output : typeof r.stdout === "string" ? r.stdout : ""
              }
              addToolOutcome(sid, toolName, command, output)
            } catch {
              // ignore
            }
          }
        } catch {
          // ignore
        }
      }

      // message.updated - get role and session mapping
      if (etype === "message.updated") {
        const props = event.properties as {
          info?: {
            id?: string
            sessionID?: string
            role?: string
            providerID?: string
            modelID?: string
            cost?: number
            tokens?: {
              input?: number
              output?: number
              reasoning?: number
              cache?: { read?: number; write?: number }
            }
          }
        }
        const info = props?.info
        if (info?.id && info?.sessionID && info?.role) {
          lastSeenSessionId = info.sessionID

          const existing = messageData.get(info.id)
          const usage: Usage | undefined =
            info.role === "assistant"
              ? {
                  provider_id: info.providerID,
                  model_id: info.modelID,
                  cost: typeof info.cost === "number" ? info.cost : undefined,
                  tokens: info.tokens,
                }
              : undefined

          // Debug: log usage data when available
          if (info.role === "assistant" && (info.cost !== undefined || info.tokens !== undefined)) {
            log("info", `message.updated usage: cost=${info.cost} tokens=${safeJson(info.tokens)}`)
          }

          messageData.set(info.id, {
            role: info.role,
            sessionId: info.sessionID,
            text: existing?.text || "",
            usage: usage ?? existing?.usage,
          })

          // Track in session exchanges
          if (!sessionExchanges.has(info.sessionID)) {
            sessionExchanges.set(info.sessionID, { userMessageId: null, assistantMessageId: null })
          }
          const exchange = sessionExchanges.get(info.sessionID)!
          if (info.role === "user") {
            exchange.userMessageId = info.id
          } else if (info.role === "assistant") {
            exchange.assistantMessageId = info.id
          }
          
          log("debug", `message.updated: id=${info.id} role=${info.role} session=${info.sessionID}`)
        }
      }

      // message.part.updated - get actual text content
      if (etype === "message.part.updated") {
        const props = event.properties as { part?: unknown }
        const part = props?.part as any

        if (typeof part?.sessionID === "string" && part.sessionID) {
          lastSeenSessionId = part.sessionID
        }

        // Rich signals for file association:
        // - patch parts list files
        // - tool parts may have file attachments with `source.path`
        // - file parts have `source.path`
        if (part?.type === "patch" && Array.isArray(part.files) && part.sessionID) {
          for (const f of part.files) {
            if (typeof f === "string") addTouchedPath(part.sessionID, f)
          }
        }

        if (part?.type === "file" && part.sessionID) {
          const p = part?.source?.path
          if (typeof p === "string") addTouchedPath(part.sessionID, p)
          const fn = part?.filename
          if (typeof fn === "string") addTouchedPath(part.sessionID, fn)
        }

        if (part?.type === "tool" && part.sessionID) {
          const atts = part?.state?.attachments
          if (Array.isArray(atts)) {
            for (const a of atts) {
              const p = a?.source?.path
              if (typeof p === "string") addTouchedPath(part.sessionID, p)
              const fn = a?.filename
              if (typeof fn === "string") addTouchedPath(part.sessionID, fn)
            }
          }
        }

        if (part?.type === "text" && part?.messageID && part?.sessionID && typeof part?.text === "string") {
          const existing = messageData.get(part.messageID)
          if (existing) {
            existing.text = part.text
          } else {
            messageData.set(part.messageID, {
              role: "unknown",
              sessionId: part.sessionID,
              text: part.text,
            })
          }
          log("debug", `message.part.updated: msgId=${part.messageID} text=${part.text.slice(0, 50)}...`)

          schedulePartialFlush(part.sessionID)
        }

        // step-finish may carry reliable final usage info.
        if (part?.type === "step-finish" && part?.messageID && part?.sessionID) {
          // Debug: log step-finish usage data
          log("info", `step-finish usage: cost=${part.cost} tokens=${safeJson(part.tokens)}`)

          const existing = messageData.get(part.messageID)
          
          // Merge step-finish usage with existing usage (preserve provider_id/model_id from message.updated)
          const usage: Usage = {
            // Keep provider/model from existing usage if available (from message.updated)
            provider_id: existing?.usage?.provider_id,
            model_id: existing?.usage?.model_id,
            // Use cost and tokens from step-finish (more reliable final values)
            cost: typeof part.cost === "number" ? part.cost : existing?.usage?.cost,
            tokens: part.tokens ?? existing?.usage?.tokens,
          }

          if (existing) {
            existing.role = existing.role === "unknown" ? "assistant" : existing.role
            existing.sessionId = existing.sessionId || part.sessionID
            existing.usage = usage
          } else {
            messageData.set(part.messageID, {
              role: "assistant",
              sessionId: part.sessionID,
              text: "",
              usage,
            })
          }

          // Ensure session exchange knows the assistant message id, even if message.updated races.
          if (!sessionExchanges.has(part.sessionID)) {
            sessionExchanges.set(part.sessionID, { userMessageId: null, assistantMessageId: null })
          }
          const exchange = sessionExchanges.get(part.sessionID)!
          exchange.assistantMessageId = part.messageID

          schedulePartialFlush(part.sessionID)
          // Give final text parts a moment to land, then record.
          setTimeout(() => {
            recordExchange(part.sessionID!)
          }, 150)
        }
      }

      // session.idle - record the exchange
      if (etype === "session.idle") {
        const props = event.properties as { sessionID?: string }
        const sessionId = props?.sessionID
        if (sessionId) {
          lastSeenSessionId = sessionId
          log("debug", `session.idle: ${sessionId}`)
          // Small delay to ensure all message.part.updated events are processed
          setTimeout(() => {
            recordExchange(sessionId)
          }, 500)
        }
      }
    },
  }
}
