---
title: "Antigravity Jules Plugin: Asynchronous Cloud Engineering Delegation from the IDE"
date: "2026-10-14"
excerpt: "How I built an open-source Antigravity IDE plugin that delegates multi-file coding sessions, bug fixes, and test suites to Google Jules cloud agents while keeping local git working trees completely clean."
category: "AI & Developer Tools"
tags: ["Antigravity", "Google Jules", "AI Agents", "Developer Tools", "CLI", "Automation"]
coverImage: "/images/covers/antigravity-jules-plugin.webp"
---

Local coding agents inside the IDE are extraordinary for interactive architecture and rapid iteration, but they share a structural bottleneck: **they occupy your local working tree and your active attention**. When you need to refactor an authentication module, write an exhaustive unit test suite, or triage a backlog of three independent bug reports simultaneously, running everything sequentially in your local checkout forces constant branch switching and context thrashing.

To solve this, I created **[antigravity-jules-plugin](https://github.com/daniazar/antigravity-jules-plugin)**—an open-source plugin for **Google Antigravity** that bridges local agentic pair-programming with **Google Jules** (`@google/jules`), Google's asynchronous cloud coding agent running on isolated remote virtual machines.

---

## Architectural Philosophy: Local Orchestration + Remote Cloud VMs

Instead of treating local IDE agents and cloud agents as competing workflows, the plugin establishes a two-tier engineering pipeline:

1. **Local Antigravity Agent (Foreground Architect)**: Collaborates with you in real time, inspects your repository metadata (`git remote get-url origin`), verifies your Google authentication (`jules version`), and scopes self-contained engineering tasks with explicit acceptance criteria.
2. **Remote Jules Cloud Sessions (Background Workers)**: Each delegated task spins up a fresh cloud VM cloned from your GitHub repository, installs dependencies, implements the fix or feature, executes test suites, and produces an inspectable diff—or publishes a GitHub Pull Request directly.

Because you can pass `--parallel <N>`, a single prompt in Antigravity can fan out multiple cloud sessions simultaneously without touching a single file in your local working directory.

---

## Core Lifecycle Workflows

The plugin packages a comprehensive `SKILL.md` instruction specification (`skills/jules/SKILL.md`), plugin manifest (`.antigravity-plugin/plugin.json`), and strict operational guardrails (`rules/AGENTS.md`) covering the entire delegation lifecycle:

### 1. Pre-Flight & Repository Resolution
Before dispatching any cloud job, the skill automatically verifies that the `@google/jules` CLI is installed and authenticated, resolves the canonical `owner/repo` slug from your git remotes (handling both SSH `git@github.com:owner/repo.git` and HTTPS URLs), and confirms the repository is connected to your Jules workspace via `jules remote list --repo`.

### 2. Scoped Cloud Dispatch (`jules new`)
Vague prompts produce vague pull requests. The plugin enforces a structured task-scoping protocol that transforms high-level user requests into actionable engineering specifications including target modules, expected behavior, edge cases, and verification commands:

```bash
# Dispatch a cloud coding task to the current repository
jules remote new --repo daniazar/CineZero \
  --session "Add unit tests for the 8-second Veo clip ceiling calculator in src/server/video and verify edge cases for non-multiple shot durations"
```

### 3. Non-Destructive Monitoring (`jules remote list --session`)
Every cloud session returns a unique session ID (`<SESSION_ID>`). Rather than polling in a blocking loop, the local agent checks session status on demand (`Completed`, `Running`, `Awaiting Plan Approval`, or `Failed`).

### 4. Safe Patch Inspection & Teleportation (`jules teleport`)
The most important safety guardrail in `antigravity-jules-plugin` governs how completed cloud work lands back on your machine:

* **Diff-First Inspection**: Before modifying any local file, the agent previews the remote patch cleanly via `jules remote pull --session <SESSION_ID>` (without `--apply`).
* **Dirty Working Tree Protection**: Before running `jules teleport <SESSION_ID>` (which clones/checks out the session branch and applies the patch) or `jules remote pull --session <SESSION_ID> --apply`, the plugin checks `git status --porcelain` to guarantee uncommitted local work is never overwritten.

---

## Source Code & Installation

The plugin is open-source and available on GitHub at **[daniazar/antigravity-jules-plugin](https://github.com/daniazar/antigravity-jules-plugin)**. You can install it directly into your global Antigravity configuration (`~/.gemini/config/plugins/jules`) or workspace `.antigravity/plugins/` directory.
