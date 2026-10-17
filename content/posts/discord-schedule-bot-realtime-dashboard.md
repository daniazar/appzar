---
title: "Discord Schedule Bot: Coordinating Global UTC Community Events with Slash Commands & Supabase Realtime"
date: "2025-07-05"
excerpt: "Building a dual-interface event scheduling system combining Discord.js v14 slash commands (/add) with a live Next.js + Supabase Realtime web dashboard for global gaming and engineering communities."
category: "Full-Stack & Bots"
tags: ["Discord.js", "Next.js", "Supabase", "TypeScript", "Realtime", "WebSockets"]
coverImage: "/images/covers/discord-schedule-bot-realtime-dashboard.webp"
---

Coordinating multiplayer matches or engineering syncs across four continents in Discord is notoriously chaotic: someone posts a time in EST, someone else replies in CET, messages scroll off-screen, and nobody knows which hour actually has a full lobby.

I built **[discord-schedule-bot](https://github.com/daniazar/discord-schedule-bot)** to give distributed Discord communities a deterministic, UTC-normalized scheduling engine that works simultaneously inside Discord via **Slash Commands** and on the web via a **live Supabase Realtime Dashboard**.

![Discord Schedule Bot Live Web Dashboard](/assets/discord-schedule-bot/dashboard.webp)

---

## 1. Dual-Interface Architecture: Discord Slash Commands + Web Matrix

The system shares a single PostgreSQL database (`titles` and `signups` tables in **Supabase**) across two synchronized interfaces:

1. **In-Channel Discord Bot (`bot.ts`)**:
   * Built with **`discord.js` v14**, registering guild/global slash commands via `scripts/register-commands.ts`.
   * Running `/add <hour>` (0–23 UTC) or `/title <event_name>` inside any Discord channel records the user's availability for that specific channel's event and posts an updated embed table grouped by UTC hour and day of the week.
2. **Interactive Next.js Web Dashboard (`pages/dashboard.tsx`)**:
   * Allows community members to view all active channel lobbies, filter by channel or participant, add or remove time slots directly from the browser, and export the full schedule to CSV.

---

## 2. Sub-Second Synchronization with Supabase Realtime

What makes the dashboard feel alive is its **Supabase Realtime (`postgres_changes`)** subscription pipeline:

```typescript
const signupsSubscription = supabase
  .channel('signups_changes')
  .on('postgres_changes', { event: '*', schema: 'public', table: 'signups' }, () => {
    loadData(false); // Refresh lobby matrix without UI flicker
  })
  .subscribe();
```

When a player types `/add 20` inside a Discord text channel, the Next.js web dashboard updates the `20:00 UTC` participant badges in under 200ms over WebSockets—and when an organizer edits a slot on the web dashboard, the bot's state stays 100% consistent.
