---
id: 0001
title: Amend Constitution to v1.0.0
stage: constitution
date: 2025-12-05
surface: agent
model: claude-haiku-4-5-20251001
feature: none
branch: main
user: user
command: /sp.constitution
labels: ["docs", "constitution"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
---

## Prompt

```text
/sp.constitution

PURPOSE OF THIS CONSTITUTION:
Define the global rules, standards, constraints, quality controls, and semantic structures that every module, chapter, component, backend service, frontend UI, and agent workflow must follow when generating the Physical AI & Humanoid Robotics Textbook Platform. This constitution governs ALL /sp.specify, /sp.plan, /sp.tasks, and /sp.implement outputs.

────────────────────────────────────────
SECTION 1 — HIGH-LEVEL SYSTEM MISSION
────────────────────────────────────────
The project MUST produce a complete AI-Native Documentation System that includes:

1. A Docusaurus digital textbook.
2. A fully functional RAG Chatbot embedded inside the textbook.
3. A backend architecture using:
   - FastAPI
   - Qdrant vector database
   - OpenAI embeddings + ChatCompletion/Agents
   - Neon Postgres database
4. AI-native features:
   - Selected-text answering mode
   - Cross-chapter semantic grounding
   - Personalized content delivery
   - Urdu/English translation toggle
5. User experience features:
   - Better-Auth login/signup
   - Custom user profile fields on signup
   - Per-chapter personalization button
   - Per-chapter Urdu translation button
6. Optional bonus integrations:
   - Claude Code subagents
   - Quiz generator
   - Chapter summarizer
   - Curriculum generator
   - Hardware advisor

All system outputs MUST be optimized for:
- clarity
- modularity
- readability
- reusability
- AI compatibility
- hackathon scoring rubric

────────────────────────────────────────
SECTION 2 — GLOBAL STYLE & QUALITY RULES
────────────────────────────────────────

2.1 Documentation Style
- Use clean, numbered headers.
- Avoid long paragraphs; break into blocks.
- Keep language simple but technical.
- Each chapter must include:
  1. Learning objectives
  2. Core theory
  3. Practical examples
  4. Code blocks
  5. Diagrams (ASCII descriptions)
  6. Exercises
  7. RAG-friendly summaries

2.2 Code Style
- All backend code MUST follow FastAPI standards.
- All React components must use functional components + hooks.
- Naming must be semantic and consistent.
- All API requests must include error handling.
- Every endpoint must be typed and documented.

2.3 AI-Native Rules
- All chapters must include embeddings-friendly chunking.
- Avoid unnecessary fluff.
- Maintain semantic density.
- Include “context blocks” for RAG indexers.

────────────────────────────────────────
SECTION 3 — CHATBOT & RAG RULES
────────────────────────────────────────

3.1 Selected Text Rule (VERY IMPORTANT)
If the user has selected text:
- The backend MUST answer **ONLY** from the selected text.
- If the answer is NOT in the selected text:
  → MUST return: “Not contained in the selected text.”

3.2 RAG Retrieval Rules
- Use OpenAI embeddings for all chunks.
- Use Qdrant for vector search.
- Use 3–5 relevant chunks per query.
- Provide fully grounded answers.

3.3 Prompting Rules
- Always include system guardrails.
- Always disclose if answer is drawn from selected text vs RAG.

────────────────────────────────────────
SECTION 4 — PERSONALIZATION RULES
────────────────────────────────────────

4.1 Personalization Button
Each chapter MUST support:
- Beginner mode
- Intermediate mode
- Advanced mode

4.2 User Profile Data (Neon)
When the user signs up, Better-Auth must capture:
- Hardware experience
- Software experience
- Robotics background
- Preferred language (Urdu/English)
- Learning goals

4.3 Personalized Content Rendering
- Frontend reads profile → displays proper difficulty blocks.
- Cache preferences in Neon.

────────────────────────────────────────
SECTION 5 — URDU TRANSLATION RULES
────────────────────────────────────────

- Every chapter must support a toggle for Urdu.
- Urdu text can be:
  • pre-generated and stored
  • or generated via LLM on demand
- On-demand translations MUST be stored in Neon cache.
- Toggle MUST persist per user.

────────────────────────────────────────
SECTION 6 — AUTHENTICATION RULES
────────────────────────────────────────

- Use Better-Auth.
- Use JWT/session cookies.
- On signup, ask the user skill questions.
- Store profile in Neon.
- Apply personalized content settings automatically after login.

────────────────────────────────────────
SECTION 7 — SUBAGENT RULES
────────────────────────────────────────

The system may generate Claude Code subagents:
- Chapter Summarizer Agent
- Quiz Generator Agent
- Hardware Advisor Agent
- Curriculum Planner Agent
- Troubleshooter Agent

All must use:
- JSON-based responses
- Dedicated API endpoints
- Semantic, deterministic outputs

────────────────────────────────────────
SECTION 8 — DEPLOYMENT RULES
────────────────────────────────────────

Frontend:
- Publish Docusaurus to GitHub Pages or Vercel.

Backend:
- Deploy FastAPI on:
  • Render
  • Railway
  • Fly.io
  • OR a Docker host

Databases:
- Qdrant Cloud (free tier)
- Neon Serverless (free tier)

────────────────────────────────────────
SECTION 9 — REQUIRED MODULES/CHAPTERS CLAUDE MUST GENERATE
────────────────────────────────────────

Claude MUST generate the following textbook chapters:

1. **Introduction to Physical AI & Humanoid Robotics**
2. **Linux + ROS2 Foundations**
3. **Gazebo / Ignition Simulation**
4. **NVIDIA Isaac Sim Robotics Simulation**
5. **Real Robot Control Architecture**
6. **Sensor Fusion + Localization (SLAM/IMU/LiDAR)**
7. **Kinematics & Dynamics (FK, IK, Trajectory Planning)**
8. **Control Systems (PID, MPC, Whole-Body Control)**
9. **Robot Perception (CV, LLM-Vision, Depth Estimation)**
10. **Vision-Language-Action Models (VLAs)**
11. **Reinforcement Learning for Robotics**
12. **Imitation Learning + Teleoperation**
13. **Building a Humanoid: Actuators, Joints, Hardware Choices**
14. **Autonomous Navigation for Humanoids**
15. **Safety, Fail-safes, Edge Computing**
16. **Capstone Project Guide**
17. **Appendix A: Embedded Systems for Robotics**
18. **Appendix B: RAG-Ready Book Summary for Search Indexing**
19. **Appendix C: Urdu Translations (Auto or Manual)**

Each chapter MUST follow:
- AI-native chunking
- Learning objectives
- Key concepts
- Diagrams
- Examples
- Code
- Exercises
- Final RAG summary

────────────────────────────────────────
SECTION 10 — FINAL ENFORCEMENT RULE
────────────────────────────────────────

All future steps (/sp.plan, /sp.tasks, /sp.implement) MUST obey this constitution strictly. No component, chapter, code file, or architecture output may violate any rule here.
```

## Response snapshot

```text
The project constitution has been updated to version 1.0.0. This is a major version bump as a complete and detailed constitution was provided, replacing the previous generic template.

Please review `plan-template.md`, `spec-template.md`, and `tasks-template.md` to ensure full alignment with the new constitution. No other documentation files were found that required updates.

Suggested commit message: `docs: amend constitution to v1.0.0 (initial detailed constitution)`
```

## Outcome

- ✅ Impact: Constitution updated, foundational project guidance established.
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Review other templates for full alignment, begin project specification.
- 🧠 Reflection: PHR creation script failed, fell back to agent-native tool usage.

## Evaluation notes (flywheel)

- Failure modes observed: PHR creation script not found.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment: Investigate existence/location of `create-phr.sh` script or update `.specify/scripts/bash/create-phr.sh` path.
