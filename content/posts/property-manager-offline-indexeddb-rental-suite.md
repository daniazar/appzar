---
title: "Property Manager: Engineering an Offline-First Rental Portfolio & Lease Suite with IndexedDB"
date: "2025-06-06"
excerpt: "How I built a local-first real estate portfolio manager in Next.js, TypeScript, Recharts, and IndexedDB to track rental properties, multi-step lease escalations, tenant payments, property taxes, and encrypted Google Drive backups."
category: "PropTech & Web Engineering"
tags: ["PropTech", "Real Estate", "IndexedDB", "Next.js", "TypeScript", "Offline-First", "Recharts"]
coverImage: "/images/covers/property-manager-offline-indexeddb-rental-suite.webp"
---

Managing a portfolio of rental properties involves surprisingly intricate financial state: leases have multi-tier step-up rent escalations (`rentSteps`), property taxes compound annually (`taxIncrementPercent`), some expenses are paid by the owner while others pass through to the tenant (`tenantPaysTaxes`), and payments settle across multiple bank accounts and currencies.

Most landlords end up maintaining fragile Excel spreadsheets because cloud SaaS property management platforms charge per-unit monthly fees and lock sensitive financial records inside proprietary databases.

To solve this, I built **[Property Manager](https://github.com/daniazar/property-manager)**—an **offline-first Single-Page Application** built with Next.js, TypeScript, Tailwind CSS, Radix UI, and **IndexedDB** (`propManagerDB`) that runs entirely in the browser with instant local persistence, full undo/redo state history, JSON snapshot portability, and optional Google Drive backup sync.

![Property Manager Financial Dashboard — Real-time portfolio KPIs, cashflow projections, and upcoming rent collections](/assets/property-manager/dashboard.webp)

---

## 1. Why IndexedDB (`propManagerDB` v4) Instead of a Cloud Database?

Financial and lease records are deeply personal. By designing a custom typed IndexedDB storage engine (`lib/storage-service.ts`) across ten relational object stores (`properties`, `accounts`, `transactions`, `payments`, `contracts`, `tenants`, `contacts`, `expenses`, `files`, and `history`), Property Manager achieves:

* **Zero Latency & 100% Offline Reliability**: Filtering transactions, projecting 12-month cashflows, and switching views execute in sub-millisecond memory/IndexedDB transactions.
* **Complete Data Ownership (`/import-export`)**: The entire relational database can be exported as a single portable JSON snapshot or backed up to the user's personal Google Drive (`/api/google-drive`).

---

## 2. Portfolio Inventory & Automated Tax Projections

The **Properties** module (`app/properties/page.tsx`) tracks residential and commercial units alongside their purchase valuation, current market value, linked operating bank account, and recurring municipal taxes:

![Property Manager — Rental Property Portfolio View](/assets/property-manager/properties.webp)

Rather than requiring manual entry for recurring property tax bills, `generateTaxExpensesForProperty()` automatically synthesizes scheduled tax expense records and applies annual percentage escalations (`taxIncrementPercent`).

---

## 3. Multi-Step Lease Contracts (`rentSteps`)

In high-inflation or multi-year commercial/residential markets, rent is rarely flat for 36 months. The **Contracts** engine (`app/contracts/page.tsx`) models leases with an arbitrary array of time-bounded **`rentSteps`**:

![Property Manager — Active Lease Contracts with Multi-Step Rent Schedules](/assets/property-manager/contracts.webp)

Each contract links a property, one or more tenants (`tenantIds`), a realtor commission structure, security deposits, payment due days, and pass-through expense rules (`tenantPaysTaxes` / `tenantPaysOtherExpenses`).

---

## 4. Rent Collection Ledger & Cashflow Analytics

From active contracts, the **Payments** engine (`app/payments/page.tsx`) tracks monthly rent collection across `paid`, `pending`, and `overdue` states, automatically reconciling settled payments into the linked bank account's transaction ledger:

![Property Manager — Rent Payments & Collection Status](/assets/property-manager/payments.webp)

On the main dashboard (`app/page.tsx`), four interactive **Recharts** visualizations (`FutureCashflowChart`, `MonthlyIncomeExpenseBarChart`, `IncomeExpensesPieChart`, and `ExpenseBreakdownPieChart`) combine historical ledger transactions with future contract `rentSteps` to give an accurate 12-month forward cashflow projection.
