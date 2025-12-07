# Implementation Plan: Physical AI & Humanoid Robotics Textbook Platform

**Branch**: `001-ai-robotics-platform` | **Date**: 2025-12-05 | **Spec**: specs/001-ai-robotics-platform/spec.md
**Input**: Feature specification from `specs/001-ai-robotics-platform/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Produce a complete AI-Native Documentation System that includes a Docusaurus digital textbook, a fully functional RAG Chatbot embedded inside the textbook, and a backend architecture using FastAPI, Qdrant, OpenAI, and Neon Postgres, with AI-native features, user experience features, and optional bonus integrations, all optimized for hackathon scoring (minimum 80% score).

## Technical Context

**Language/Version**: Python 3.11 (FastAPI backend), Node.js (Docusaurus frontend), React 18+ (frontend components)
**Primary Dependencies**: FastAPI, Qdrant (vector DB), OpenAI (embeddings/ChatCompletion), Neon Postgres (relational DB), Docusaurus, React, Better-Auth (for authentication)
**Storage**: Qdrant (vector embeddings for RAG), Neon Postgres (user profiles, personalization settings, Urdu translation cache)
**Testing**: `pytest` for FastAPI backend, `Jest` / `React Testing Library` for Docusaurus/React frontend
**Target Platform**: Linux server (for FastAPI backend deployment), Web browser (for Docusaurus frontend)
**Project Type**: Web application (Frontend + Backend)
**Performance Goals**: RAG Chatbot provides a grounded answer within 5 seconds for 90% of queries. User signup/login processes complete within 30 seconds for 95% of users. Personalized chapter content rendered correctly for 99% of user requests.
**Constraints**: Must adhere to all rules in the project constitution. Must optimize for hackathon scoring rubric (min 80%).
**Scale/Scope**: All 19 specified textbook chapters with personalized content and Urdu translations. Fully integrated RAG chatbot. Secure authentication and user profiles. Optional Claude subagents.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **SECTION 1 — HIGH-LEVEL SYSTEM MISSION**: The plan aligns by covering all components (Docusaurus textbook, RAG Chatbot, FastAPI backend, etc.), AI-native features, user experience features, and optional bonus integrations. Optimization for clarity, modularity, readability, reusability, AI compatibility, and hackathon rubric is a core tenet of the plan.
- [x] **SECTION 2 — GLOBAL STYLE & QUALITY RULES**:
  - [x] *Documentation Style*: Plan includes tasks for creating chapters with learning objectives, core theory, practical examples, code blocks, diagrams, exercises, and RAG-friendly summaries.
  - [x] *Code Style*: Plan specifies FastAPI standards for backend, React functional components + hooks for frontend, semantic/consistent naming, error handling for API requests, and typed/documented endpoints.
  - [x] *AI-Native Rules*: Plan includes tasks for embeddings-friendly chunking, semantic density, and context blocks for RAG indexers.
- [x] **SECTION 3 — CHATBOT & RAG RULES**:
  - [x] *Selected Text Rule*: Plan accounts for implementing this logic in the backend.
  - [x] *RAG Retrieval Rules*: Plan specifies OpenAI embeddings, Qdrant, and 3-5 relevant chunks per query.
  - [x] *Prompting Rules*: Plan includes tasks for system guardrails and disclosure of answer source.
- [x] **SECTION 4 — PERSONALIZATION RULES**:
  - [x] *Personalization Button*: Plan includes tasks for supporting Beginner, Intermediate, Advanced modes per chapter.
  - [x] *User Profile Data (Neon)*: Plan includes tasks for capturing and storing hardware/software experience, robotics background, preferred language, and learning goals in Neon upon signup.
  - [x] *Personalized Content Rendering*: Plan includes tasks for frontend to read profile and display difficulty blocks, and caching preferences in Neon.
- [x] **SECTION 5 — URDU TRANSLATION RULES**:
  - [x] Plan includes tasks for Urdu toggle per chapter, pre-generated or on-demand LLM translations, storing on-demand translations in Neon cache, and persisting toggle per user.
- [x] **SECTION 6 — AUTHENTICATION RULES**:
  - [x] Plan includes tasks for Better-Auth, JWT/session cookies, custom signup fields, storing profile in Neon, and automatic application of personalized content settings post-login.
- [x] **SECTION 7 — SUBAGENT RULES**:
  - [x] Plan includes tasks for optional Claude Code subagents (Summarizer, Quiz Generator, Hardware Advisor, Curriculum Planner, Troubleshooter) with JSON responses, dedicated API endpoints, and semantic, deterministic outputs.
- [x] **SECTION 8 — DEPLOYMENT RULES**:
  - [x] Plan includes tasks for deploying Docusaurus to GitHub Pages/Vercel, FastAPI backend to Render/Railway/Fly.io/Docker, Qdrant Cloud, and Neon Serverless.
- [x] **SECTION 9 — REQUIRED MODULES/CHAPTERS CLAUDE MUST GENERATE**:
  - [x] Plan includes tasks for generating all 19 specified chapters, each following the detailed content requirements.
- [x] **SECTION 10 — FINAL ENFORCEMENT RULE**: The plan is designed to strictly obey all rules in the constitution.

## Project Structure

### Documentation (this feature)

```text
specs/001-ai-robotics-platform/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/             # FastAPI endpoints (query, embed, translate, personalize, auth, agents)
│   ├── models/          # Pydantic models for data (User, Chapter, PersonalizationSettings, TranslationCache)
│   ├── services/        # Business logic, RAG pipeline, Auth logic, Translation logic, Personalization logic
│   └── core/            # Config, middleware, logging, database connections (Neon, Qdrant)
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
├── src/
│   ├── components/      # React components (ChatWidget, PersonalizationToggle, UrduToggle, Auth forms)
│   ├── pages/           # Docusaurus pages / React views (Login, Signup, Profile, Chapter viewer)
│   ├── hooks/           # Custom React hooks for state/logic
│   ├── services/        # Frontend API calls to backend
│   └── context/         # React Context for user/personalization state
└── docs/                # Docusaurus markdown files for textbook chapters
    └── {chapter_name}/
        ├── index.md
        ├── content.md
        └── translations.md
```

**Structure Decision**: The "Web application" structure (Option 2 from template) is chosen due to the clear separation of Docusaurus frontend and FastAPI backend, aligning with the project's requirements. This provides a clean separation of concerns and facilitates independent development and deployment of frontend and backend services.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A       | N/A        | N/A                                 |
