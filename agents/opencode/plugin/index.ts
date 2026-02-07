// OpenCode plugin: unlost/unloop
//
// Records conversations and detects friction loops.
// Spawns `unlost companion` as a child process for all logic.

import { spawn, type ChildProcess } from "node:child_process"
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

  // Track messages by ID -> { role, sessionId, text }
  const messageData = new Map<string, { role: string; sessionId: string; text: string }>()
  
  // Track session's last user/assistant for recording
  const sessionExchanges = new Map<string, { userMessageId: string | null; assistantMessageId: string | null }>()

  function log(level: "debug" | "info" | "warn" | "error", message: string) {
    client.app.log({ body: { service: "unlost", level, message } })
  }

  function spawnCompanion() {
    const bin = findUnlostBinary()
    try {
      companion = spawn(bin, ["companion"], {
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
      return { note: null, ok: false } as T
    }

    const id = ++requestId
    const req = JSON.stringify({ method, params }) + "\n"

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        pendingRequests.delete(id)
        resolve({ note: null, ok: false, error: "timeout" } as T)
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
        resolve({ note: null, ok: false, error: err.message } as T)
      }
    })
  }

  function recordExchange(sessionId: string) {
    const exchange = sessionExchanges.get(sessionId)
    if (!exchange) return

    const userMsg = exchange.userMessageId ? messageData.get(exchange.userMessageId) : null
    const assistantMsg = exchange.assistantMessageId ? messageData.get(exchange.assistantMessageId) : null

    const userText = userMsg?.text || ""
    const assistantText = assistantMsg?.text || ""

    if (!userText && !assistantText) return

    log("info", `recording exchange: session=${sessionId} user=${userText.slice(0, 50)}... assistant=${assistantText.slice(0, 50)}...`)

    sendRequest<RecordResponse>("record", {
      user_text: userText,
      assistant_text: assistantText,
      directory: workingDirectory,
      agent_session_id: sessionId,
    }).catch(() => {})

    // Clear exchange tracking for next round
    sessionExchanges.set(sessionId, { userMessageId: null, assistantMessageId: null })
  }

  // Spawn companion on plugin init
  spawnCompanion()
  await new Promise((resolve) => setTimeout(resolve, 500))

  log("info", `plugin initialized, directory=${workingDirectory}`)

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
      // message.updated - get role and session mapping
      if (event.type === "message.updated") {
        const props = event.properties as { info?: { id?: string; sessionID?: string; role?: string } }
        const info = props?.info
        if (info?.id && info?.sessionID && info?.role) {
          const existing = messageData.get(info.id)
          messageData.set(info.id, {
            role: info.role,
            sessionId: info.sessionID,
            text: existing?.text || "",
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
      if (event.type === "message.part.updated") {
        const props = event.properties as { part?: { type?: string; messageID?: string; sessionID?: string; text?: string } }
        const part = props?.part
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
        }
      }

      // session.idle - record the exchange
      if (event.type === "session.idle") {
        const props = event.properties as { sessionID?: string }
        const sessionId = props?.sessionID
        if (sessionId) {
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
