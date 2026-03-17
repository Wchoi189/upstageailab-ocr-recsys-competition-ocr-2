# Order of Operations — Resume Protocol (Rigid)

This protocol is the only supported way to resume work.

## 0) Resolve “latest session”
The latest session is the lexicographically greatest filename in:
`specs/global-agentqms/session_handovers/` matching `SESSION_HANDOVER_YYYY-MM-DD_HHMM.md`.

## 1) Review previous work
- Open the latest session handover file.
- Read:
  - `session_summary.progress`
  - `backlog_state`
  - `next_session_entry_point` (the authoritative “what to do next”)

## 2) Review specs & backlog
- Open the active run folder referenced by the handover:
  - `specs/global-agentqms/runs/YYYY-MM-DD_HHMM/`
- Read:
  - `spec_index.md`
  - `backlog.md`
- Confirm which spec is active and which backlog items are `in_progress` / `next_up`.

## 3) Execute current session tasks
- Work only the `next_up` items.
- For each completed item, record evidence (command + result).

## 4) Write the next handover (end of session)
- Create a new file:
  - `specs/global-agentqms/session_handovers/SESSION_HANDOVER_YYYY-MM-DD_HHMM.md`
- Include:
  - What changed (files, behavior)
  - Evidence (tests/commands)
  - Updated backlog state
  - A single unambiguous next command under `next_session_entry_point`

