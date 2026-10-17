---
title: "ContractApp: AI-Assisted Legal Contract Wizard with Composable Clause Libraries and E-Signatures"
date: "2025-09-12"
excerpt: "Designing a modular contract lifecycle application in Next.js, TypeScript, Material UI, and Genkit AI that combines guided legal questionnaires, drag-and-drop clause assembly, and digital signature stamping."
category: "Full-Stack & AI"
tags: ["Next.js", "TypeScript", "LegalTech", "Genkit", "Material UI", "Playwright"]
coverImage: "/images/covers/contractapp-ai-legal-contract-wizard.webp"
---

Drafting standard commercial agreements—whether residential leases, mutual NDAs, consulting agreements, or employment contracts—usually swings between two extremes: copying brittle Word templates full of leftover clauses, or paying for heavyweight enterprise CLM software.

I built **[ContractApp](https://github.com/daniazar/contracts-app)** to streamline contract creation into a guided, four-step engineering workflow: **AI Questionnaire → Modular Clause Assembly → Live Document Preview → Digital Signature Stamping**.

![ContractApp Guided Creation Wizard & Live Document Preview](/assets/contracts-app/wizard.webp)

---

## 1. Guided Wizard & Composable Clause Architecture

Instead of treating a contract as a monolithic blob of text, ContractApp (`src/app/contracts/create/page.tsx`) models every agreement as an ordered composition of typed **Clause Modules** (`src/app/api/clauses/route.ts`):

* **Context Questionnaire**: Captures the contract type (*Lease Agreement*, *NDA*, *Consulting Agreement*, *Employment Contract*), counterparty identities, governing jurisdiction, and effective dates.
* **Interactive Clause Library**: Users toggle optional legal protections (e.g., *Confidentiality*, *Intellectual Property Assignment*, *Limitation of Liability*, *Early Termination*, *Arbitration*), reorder clauses via drag-and-drop, and customize clause wording with AI assistance powered by **Google Genkit** (`@genkit-ai/googleai`).
* **Real-Time Document Preview**: Every parameter or clause toggle updates the formatted legal document preview alongside the form.

---

## 2. Contract Pipeline Dashboard & Signature Tracking

Once generated, agreements flow into the **Contract Dashboard** (`src/app/dashboard/page.tsx`), built with **Material UI (`@mui/x-data-grid`)**:

![ContractApp Dashboard tracking Active and Awaiting-Signature agreements](/assets/contracts-app/dashboard.webp)

* **Lifecycle Metrics**: Tracks active agreements versus contracts awaiting counterparty signature.
* **Cryptographic & Visual Signature Stamping (`/api/signatures`)**: Supports interactive signature capture and timestamped execution records, verified with Jest unit tests and Playwright end-to-end test suites.
