---
id: 0002
title: Generate Plan for AI Robotics Textbook Platform
stage: plan
date: 2025-12-05
surface: agent
model: claude-haiku-4-5-20251001
feature: ai-robotics-platform
branch: 001-ai-robotics-platform
user: user
command: /sp.plan
labels: ["plan", "design"]
links:
  spec: specs/001-ai-robotics-platform/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-ai-robotics-platform/plan.md
  - specs/001-ai-robotics-platform/research.md
  - specs/001-ai-robotics-platform/data-model.md
  - specs/001-ai-robotics-platform/contracts/openapi.yaml
  - specs/001-ai-robotics-platform/quickstart.md
tests:
---

## Prompt

```text
/sp.plan

Generate the full atomic task breakdown for the entire platform described in the /sp.constitution and /sp.specify specifications.

Your output MUST:
1. Break down the entire system into modules.
2. Break each module into atomic, implementable tasks.
3. Ensure every task is actionable, small, unambiguous, and directly tied to a deliverable.
4. Include tasks for frontend, backend, RAG system, Docusaurus book creation, chapter generation, authentication, personalization, Urdu translation, Claude subagents, indexing, devops, deployment, and documentation.
5. Follow the global rules of the /sp.constitution exactly.
6. Cover every feature and chapter listed in /sp.specify and /sp.constitution.
7. Produce tasks in a structured, hierarchical, dependency-respecting format.
8. Ensure tasks are comprehensive enough for /sp.tasks and /sp.implement to generate full code and content.

────────────────────────────────────────
MODULES TO GENERATE TASKS FOR
────────────────────────────────────────

Create tasks for all modules:

### 1. Environment Setup & Docusaurus Installation
- Node.js setup
- Docusaurus scaffolding
- Project folder structure creation
- Connecting documentation to GitHub repo
- Docusaurus config, sidebars, themes
- Setup instructions referencing https://docusaurus.io/docs

### 2. Textbook Content Creation (19 Chapters)
Generate tasks for ALL chapters of:
**Physical AI & Humanoid Robotics**

Chapters include:
1. Introduction
2. Linux + ROS2
3. Gazebo
4. Isaac Sim
5. Robot Control Architecture
6. Sensor Fusion & Localization
7. Kinematics & Dynamics
8. Control Systems
9. Robot Perception
10. VLAs
11. RL for Robotics
12. Imitation Learning
13. Building a Humanoid
14. Humanoid Navigation
15. Safety
16. Capstone Guide
17. Embedded Systems
18. RAG Summary Appendix
19. Urdu Translation Appendix

For *each chapter*, include:
- Learning objectives creation
- Theory writing
- Practical examples
- ROS2/Isaac code blocks
- Diagrams (ASCII descriptions)
- Exercises
- RAG summary section
- Chunking for embeddings
- Personalization variants
- Urdu translation structure

### 3. Frontend (Docusaurus + React)
- Global UI layout
- ChatWidget component
- Selected-text capture logic
- Personalization toggle UI
- Urdu toggle UI
- Auth pages (login, signup, settings)
- Profile page (pull from Neon)
- Embedding chapter metadata
- Calling backend endpoints

### 4. Backend (FastAPI)
- App setup
- /query endpoint (chatbot)
- Selected-text mode logic
- RAG fallback logic
- /embed endpoint
- /translate endpoint
- /personalize endpoint
- /auth/* routes integration
- /agents/* subagent endpoints
- Middleware, logging, streaming

### 5. RAG Pipeline
- Chunk generation
- Embeddings creation using OpenAI
- Qdrant schema creation
- Upload embeddings to Qdrant
- Retrieval strategies
- Context builder
- Safety prompts
- Selected-text override logic

### 6. Authentication (Better-Auth)
- Signup fields:
  • hardware experience
  • software experience
  • robotics background
  • preferred language
  • learning goals
- Login integration
- Session handling
- Protecting routes
- Syncing profile to Neon

### 7. Personalization System
- Difficulty levels (Beginner/Intermediate/Advanced)
- Per-chapter blocks
- Personalization API
- Persistence in Neon
- UI toggle tasks
- Conditional rendering tasks

### 8. Urdu Translation System
- Translation endpoint
- Cache table in Neon
- Toggle UI tasks
- Translation injection into Markdown
- On-demand translation workflow
- Storing translated content

### 9. Claude Subagents (Optional Bonus)
Tasks for:
- Summarizer Agent
- Quiz Generator Agent
- Hardware Advisor Agent
- Curriculum Planner Agent
- Troubleshooter Agent

Include:
- Prompt schemas
- API endpoints
- JSON spec
- Integration with UI

### 10. Database (Neon)
- Create all required tables
- Migrations
- CRUD handlers
- Indexing
- Querying and joining
- User preferences persistence

### 11. Deployment & CI/CD
- GitHub Pages/Vercel deployment tasks
- Backend deployment on Render/Railway/Fly.io
- Environment variables
- Qdrant/Neon connection configuration
- Build pipelines
- Testing and verification
- Production checklist

### 12. Documentation & README
- Root README
- API documentation
- Developer onboarding
- Instructions for running locally
- Instructions for deploying
- Demo instructions (90-second demo outline)

────────────────────────────────────────
OUTPUT FORMAT REQUIREMENTS
────────────────────────────────────────

Your output MUST follow this format:

- High-level module → submodule → atomic tasks
- Use numbering like:
  1.0 Module
     1.1 Submodule
        1.1.1 Task
        1.1.2 Task

Tasks MUST be:
- Precise
- Actionable
- Dependency-aware
- Implementation-ready

All tasks MUST directly follow the /sp.constitution and /sp.specify rules.
```

## Response snapshot

```text
The plan.md and associated design artifacts (research.md, data-model.md, contracts/openapi.yaml, quickstart.md) for the 001-ai-robotics-platform feature have been successfully generated.

The agent context update script failed due to the absence of pwsh. This may require manual updates or a different approach for agent context management.

The planning phase is complete. The project is ready for task breakdown.

Next step: Generate tasks using /sp.tasks.

---
📋 Architectural decision detected: Overall system architecture for the Physical AI & Humanoid Robotics Textbook Platform. Document reasoning and tradeoffs? Run `/sp.adr Overall System Architecture`
```

## Outcome

- ✅ Impact: Comprehensive implementation plan and design artifacts generated for the platform.
- 🧪 Tests: N/A
- 📁 Files: Generated plan.md, research.md, data-model.md, openapi.yaml, quickstart.md.
- 🔁 Next prompts: Generate tasks using `/sp.tasks`, consider creating an ADR.
- 🧠 Reflection: PHR creation script and agent context update script failed, fell back to agent-native tool usage.

## Evaluation notes (flywheel)

- Failure modes observed: PHR creation script not found; `pwsh` not found for agent context update.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment: Investigate platform's script execution capabilities and preferred shell, or rely on agent-native file operations.
