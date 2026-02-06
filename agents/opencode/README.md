# OpenCode Integration

Goal: integrate unlost with OpenCode without requiring users to:
- start `unlost serve` manually
- change provider `baseURL` / endpoint

Strategy: an OpenCode plugin that:
- listens to session/message events to capture ground truth
- uses `experimental.chat.messages.transform` to inject friction warnings pre-flight

Plugin package: `@unfault/opencode-unlost` in `agents/opencode/plugin/`.

Install into a repo:

```bash
unlost config agent opencode-plugin --path .
```
