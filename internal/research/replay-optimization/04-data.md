# Dataset Characterization

## Source Data

The dataset consists of real conversation histories from **OpenCode**, stored in the local storage directory. These sessions cover software engineering tasks across two projects: `unlost` and `unfault`.

### Summary of Available Sessions

| Session ID | Turns | Project | Title / Goal |
|------------|-------|---------|--------------|
| `ses_47f5be...` | 456 | `unfault` | Implementing dev-native landing concept |
| `ses_48670d...` | 261 | `unfault` | Analyzing function impact with CodeLens |
| `ses_4867cc...` | 212 | `unfault` | Listing features supported by this section |
| `ses_3a2ae1...` | 38 | `unlost` | Reviewing workspace design and functionality |
| `ses_486850...` | 16 | `unfault` | Analyzing runtime context in editor |
| `ses_489bc2...` | 14 | `unfault` | Updating landing page for Unfault direction |
| `ses_3a2adf...` | 7 | `unlost` | Explore unlost codebase |

**Total turns available**: ~1,004

---

## Benchmarking Subsets

We will use two subsets for our experiments:

### 1. The "Sprint" Set (Small, Quality-Focused)
- **Size**: 50 turns
- **Source**: Selected from `ses_3a2ae1...` and `ses_4867cc...`
- **Use**: Ground truth creation, manual quality scoring, pivotal turn validation.
- **Coverage**: Includes clear decision points, refactors, and questions.

### 2. The "Marathon" Set (Large, Performance-Focused)
- **Size**: 456 turns
- **Source**: Full `ses_47f5be...` session.
- **Use**: Latency benchmarking, cost projection, batching efficiency tests.
- **Coverage**: Long implementation flow with many incremental turns.

---

## Statistical Profile (Target for Sprint Set)

We aim to select turns that represent the diversity of agent interactions:

| Turn Type | Target % | Characteristics |
|-----------|----------|-----------------|
| **Incremental** | 60% | Small code edits, status checks, minor clarifications. |
| **Pivotal** | 20% | Major architectural decisions, goal changes, corrections. |
| **Friction** | 10% | Loops, retries, "you already tried that" moments. |
| **Informational** | 10% | Questions about the codebase, "how does X work?". |

---

## Identified "Friction Moments" (Preliminary)

We have identified several candidate moments for the Friction Utility benchmark:
1. **The Loop**: Agent tries to fix a CSS alignment issue 4 times using the same strategy.
2. **The Drift**: Agent starts refactoring a file that was explicitly excluded from the task.
3. **The False Progress**: Agent claims "landing page is done" but the mobile view is completely broken.

---

## Data Format (OpenCode)

Each turn consists of a pair of messages (User/Assistant).
- **User Message**: Contains the prompt + `summary.title` (often the intent).
- **Assistant Message**: Contains the response + `summary.diffs` (touched paths).
- **Metadata**: Provider, Model, Tokens, Cost, Timestamp.

**Note**: The presence of `summary.title` in OpenCode's own data provides a unique opportunity to compare our extracted "Intent" against OpenCode's internal summary.
