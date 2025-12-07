---
id: 0003
title: Generate Tasks for AI Robotics Textbook Platform
stage: tasks
date: 2025-12-05
surface: agent
model: claude-haiku-4-5-20251001
feature: ai-robotics-platform
branch: 001-ai-robotics-platform
user: user
command: /sp.tasks
labels: ["tasks", "plan"]
links:
  spec: specs/001-ai-robotics-platform/spec.md
  plan: specs/001-ai-robotics-platform/plan.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-ai-robotics-platform/tasks.md
tests:
---

## Prompt

```text
/sp.tasks

Using the full breakdown created by /sp.plan, generate a complete implementation task list for the entire platform described in /sp.constitution and /sp.specify.

Your goal is to transform each atomic task from /sp.plan into:
- actionable implementation steps,
- detailed coding instructions,
- file creation commands,
- folder paths,
- component specifications,
- endpoint definitions,
- database schema tasks,
- RAG indexing instructions,
- Docusaurus writing tasks,
- and UI integration tasks.

Each task MUST be detailed enough for /sp.implement to produce code and content without ambiguity.

────────────────────────────────────────
TASK GENERATION RULES
────────────────────────────────────────

### 1. FOLLOW STRUCTURE FROM /sp.plan
For every module/submodule/task in /sp.plan:

- Expand into actionable implementation tasks.
- Each task must specify:
  • What file to create or edit
  • Expected content or structure
  • Code-level instructions
  • Integration points
  • Dependencies
  • External configuration needed

### 2. FOLLOW /sp.constitution RULES
All tasks must observe:

- Selected-text answering rule
- AI-native documentation design
- Personalization logic
- Urdu toggle mechanics
- Auth fields specifications
- Qdrant + Neon usage
- Claude subagent structure
- Docusaurus formatting & chunking standards

### 3. CHAPTER CREATION TASKS (19 CHAPTERS)
For each textbook chapter in **Physical AI & Humanoid Robotics**, generate tasks for:

- Create `docs/<chapter-id>-<chapter-name>.md`
- Write learning objectives
- Write theory sections
- Insert example code (ROS2, Isaac Sim, Python/C++)
- Write ASCII diagrams
- Add exercises
- Add RAG summary block
- Add personalized content blocks:
  • Beginner
  • Intermediate
  • Advanced
- Add Urdu translation placeholders per section
- Mark chunk boundaries for RAG indexer

### 4. DOCUSAURUS SETUP TASKS
Create tasks for:

- Installing Docusaurus
- Running initial scaffold
- Setting up `docusaurus.config.js`
- Creating `sidebars.js` structure
- Adding theme customizations
- Adding global CSS
- Adding ChatWidget to layout
- Linking official docs for help:
  https://docusaurus.io/docs

### 5. FRONTEND IMPLEMENTATION TASKS
Break into tasks for:

- `src/components/ChatWidget.jsx`
- Selected text capture logic
- API integration (`/query`, `/translate`, `/personalize`)
- Auth pages (`LoginPage.jsx`, `SignupPage.jsx`, `ProfilePage.jsx`)
- Personalization & Urdu toggles
- State management (localStorage or Neon-backed profile)
- Chapter templates with dynamic rendering

### 6. BACKEND IMPLEMENTATION TASKS
Break into tasks for:

FASTAPI STRUCTURE
- Create `backend/main.py`
- Setup routers: `/query`, `/embed`, `/translate`, `/personalize`, `/auth`, `/agents`
- Setup dependencies & middleware
- Env variable loading

CHATBOT ENDPOINT
- Selected-text-only rule implementation
- RAG fallback implementation
- Context builder
- Safety prompts
- Streaming responses

RAG PIPELINE
- Chunking
- Embeddings generation
- Upload to Qdrant
- Retrieval logic

URDU TRANSLATION
- Translation endpoint
- Neon caching tasks
- Markdown injection tasks

PERSONALIZATION
- API rules
- Neon persistence
- Mode-switch logic

BETTER-AUTH
- Integration instructions
- Signup field persistence
- Session middleware

### 7. DATABASE (NEON + QDRANT) TASKS
Generate tasks to:

- Create Neon tables with schemas
- Create migrations
- Setup Qdrant collection
- Add indexing scripts
- Write utility functions for CRUD operations

### 8. CLAUDE SUBAGENT TASKS
Produce tasks for:

- Summarizer agent
- Quiz generator agent
- Hardware advisor agent
- Curriculum planner agent
- Troubleshooter agent

Each includes:
- API endpoint
- JSON schema
- Prompt template file
- Frontend action button task (optional)

### 9. DEPLOYMENT TASKS
Break into:

FRONTEND
- GitHub Pages or Vercel deployment tasks
- Build commands
- CI tasks

BACKEND
- Render / Railway / Fly.io deployment
- Env variable setup
- Health checks

DATABASE
- Qdrant cluster setup
- Neon project setup

### 10. DOCUMENTATION TASKS
Include tasks to generate:

- `README.md` for the entire repo
- Developer setup guide
- Deployment guide
- API docs
- Troubleshooting guide
- Demo preparation guide (90-second video)

────────────────────────────────────────
OUTPUT FORMAT REQUIREMENTS
────────────────────────────────────────

Output must follow:

- Module
  - Submodule
    - Atomic Implementation Task
    - Atomic Implementation Task
    - …

Every implementation task **must be explicit**:
- File path
- Code stub or behavior description
- Required inputs
- Expected outputs
- Integration notes

Ensure the full list is exhaustive and ready for /sp.implement.
```

## Response snapshot

```text
The `tasks.md` file, detailing the implementation plan for the "Physical AI & Humanoid Robotics Textbook Platform," has been generated at `specs/001-ai-robotics-platform/tasks.md`.

**Summary of Task Generation:**
*   **Total Tasks**: 82 atomic implementation tasks have been defined.
*   **Tasks per User Story**:
    *   User Story 1 (Read Personalized Textbook - P1): 10 tasks (T029-T038)
    *   User Story 2 (Interact with RAG Chatbot - P1): 6 tasks (T039-T044)
    *   User Story 3 (Manage User Account - P2): 11 tasks (T018-T028)
    *   User Story 4 (Utilize Claude Subagents - P3): 5 tasks (T045-T049)
    *   Setup, Foundational, Textbook Content Creation, and Polish phases account for the remaining tasks.
*   **Parallel Opportunities**: Numerous tasks are marked with `[P]`, indicating they can be executed in parallel to optimize development time.
*   **Suggested MVP Scope**: The initial MVP focuses on **User Story 3 (Manage User Account)** for core authentication and profile management, followed by **User Story 1 (Read Personalized Textbook)** for personalized content delivery.

The project is now ready for implementation using the generated task list.

Suggested commit message: `feat: Generate full implementation task list for AI Robotics Textbook Platform`
```

## Outcome

- ✅ Impact: Comprehensive implementation task list generated, detailing all steps required to build the platform.
- 🧪 Tests: N/A
- 📁 Files: Generated `specs/001-ai-robotics-platform/tasks.md`.
- 🔁 Next prompts: Proceed with implementation using `/sp.implement`.
- 🧠 Reflection: Successfully parsed complex plan and spec documents to generate a detailed, structured, and actionable task list.

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment: N/A
