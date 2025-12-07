# Feature Specification: Physical AI & Humanoid Robotics Textbook Platform

**Feature Branch**: `001-ai-robotics-platform`
**Created**: 2025-12-05
**Status**: Draft
**Input**: User description: "
- FastAPI backend + Qdrant + Neon
- Personalized learning
- Urdu translation toggle
- Better-Auth login
- Claude subagents (optional bonus)

All outputs must align with the hackathon rubric.

────────────────────────────────────────
SECTION 2 — GLOBAL STYLE & QUALITY RULES
────────────────────────────────────────

Documentation Style:
- AI-native, clean, semantic chapters
- Include learning objectives, code examples, exercises
- Maintain compact, RAG-friendly text

Code Style:
- FastAPI standards
- React functional components + hooks
- Robust typing & API docs

AI-Native Rules:
- Chunkable sections
- High semantic density
- Clean summaries for embedding

────────────────────────────────────────
SECTION 3 — CHATBOT & RAG RULES
──────────────/sp.constitution

BOOK TITLE:
Physical AI & Humanoid Robotics

PURPOSE OF THIS CONSTITUTION:
Define the global rules, standards, constraints, quality controls, and semantic structures that every module, chapter, component, backend service, frontend UI, and agent workflow must follow when generating the digital textbook and platform titled:

**“Physical AI & Humanoid Robotics”**

This constitution governs ALL /sp.specify, /sp.plan, /sp.tasks, and /sp.implement outputs.

────────────────────────────────────────
SECTION 1 — HIGH-LEVEL SYSTEM MISSION
────────────────────────────────────────
The project MUST produce a complete AI-Native Documentation System for the book titled:

**Physical AI & Humanoid Robotics**

The system includes:
- Docusaurus textbook for “Physical AI & Humanoid Robotics”
- Embedded RAG chatbot
- Selected-text answering mode──────────────────────────

Selected Text Rule:
If user selects text → MUST answer ONLY from selected text.
If answer not present → return: **"Not contained in the selected text."**

RAG Rules:
- Use OpenAI embeddings + Qdrant
- Retrieve 3–5 chunks
- Combine into grounded context

────────────────────────────────────────
SECTION 4 — PERSONALIZATION RULES
────────────────────────────────────────

Personalization:
- Beginner / Intermediate / Advanced modes
- Personalized chapter rendering

User Profile (Neon):
- Hardware experience
- Software experience
- Robotics background
- Preferred language (Urdu/English)
- Learning goals

────────────────────────────────────────
SECTION 5 — URDU TRANSLATION RULES
────────────────────────────────────────

Each chapter must support:
- Urdu/English toggle
- Cached translations in Neon

────────────────────────────────────────
SECTION 6 — AUTHENTICATION RULES
────────────────────────────────────────

Use Better-Auth:
- JWT/session cookies
- Custom signup fields (skill assessment)
- Store user profiles in Neon

────────────────────────────────────────
SECTION 7 — SUBAGENT RULES
────────────────────────────────────────

Optional Claude Code subagents:
- Chapter summarizer agent
- Quiz generator agent
- Curriculum planner agent
- Hardware recommendation agent
- Troubleshooter agent

All agents must:
- Use JSON responses
- Have dedicated API endpoints

────────────────────────────────────────
SECTION 8 — DEPLOYMENT RULES
────────────────────────────────────────

Frontend: GitHub Pages or Vercel
Backend: Render / Railway / Fly.io / Docker host
DB: Qdrant Cloud + Neon Serverless
Everything must support environment variables & CI/CD.

────────────────────────────────────────
SECTION 9 — REQUIRED CHAPTERS CLAUDE MUST GENERATE
────────────────────────────────────────

Claude MUST generate all chapters of the textbook titled:
**“Physical AI & Humanoid Robotics”**

Chapters include:

1. Introduction to Physical AI & Humanoid Robotics
2. Linux + ROS2 Foundations
3. Gazebo / Ignition Simulation
4. NVIDIA Isaac Sim Robotics Simulation
5. Real Robot Control Architecture
6. Sensor Fusion + Localization (SLAM/IMU/LiDAR)
7. Kinematics & Dynamics (FK, IK, Trajectory Planning)
8. Control Systems (PID, MPC, Whole-Body Control)
9. Robot Perception (CV, LLM-Vision, Depth Estimation)
10. Vision-Language-Action Models (VLAs)
11. Reinforcement Learning for Robotics
12. Imitation Learning + Teleoperation
13. Building a Humanoid (Actuators, Joints, Hardware Choices)
14. Autonomous Navigation for Humanoids
15. Safety, Fail-safes, Edge Computing
16. Capstone Project Guide
17. Appendix A: Embedded Systems for Robotics
18. Appendix B: RAG-Ready Book Summary for Indexing
19. Appendix C: Urdu Translations (Auto/Manual)

Each chapter MUST include:
- Learning objectives
- Theory
- Examples
- ROS2/Isaac code
- Diagrams
- Exercises
- RAG summary

────────────────────────────────────────
SECTION 10 — FINAL ENFORCEMENT RULE
────────────────────────────────────────

All actions under /sp.plan, /sp.tasks, and /sp.implement MUST obey this constitution without exception."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Read Personalized Textbook (Priority: P1)

A user wants to read a chapter from the "Physical AI & Humanoid Robotics" textbook, with the content personalized to their skill level (Beginner, Intermediate, Advanced) and preferred language (Urdu/English).

**Why this priority**: This is the core functionality of the platform - delivering personalized educational content. Without it, the platform's primary value proposition is not met.

**Independent Test**: A user can log in, select a chapter, choose a personalization mode and language, and view the correctly rendered content. This can be tested independently by mocking user profile data and chapter content.

**Acceptance Scenarios**:

1.  **Given** a logged-in user with a "Beginner" profile and "English" preference, **When** they navigate to a chapter, **Then** the chapter content is displayed in English at the Beginner difficulty level.
2.  **Given** a logged-in user with an "Advanced" profile and "Urdu" preference, **When** they navigate to a chapter and the Urdu translation is available/generated, **Then** the chapter content is displayed in Urdu at the Advanced difficulty level.
3.  **Given** a user toggles the language from English to Urdu, **When** the system has cached the Urdu translation, **Then** the chapter content immediately switches to Urdu.

---

### User Story 2 - Interact with RAG Chatbot (Priority: P1)

A user wants to use the embedded RAG Chatbot to ask questions about the textbook content, either generally or based on selected text within a chapter.

**Why this priority**: The RAG Chatbot is a critical AI-native feature providing interactive learning and deep semantic grounding within the textbook.

**Independent Test**: A user can ask a general question and receive a grounded answer from the textbook. They can also select text, ask a question, and receive an answer *only* from the selected text, or a "Not contained" message if irrelevant. This can be tested independently using mock RAG services and a fixed set of textbook content.

**Acceptance Scenarios**:

1.  **Given** a user is viewing a chapter, **When** they ask a general question via the chatbot, **Then** the chatbot provides a grounded answer using 3-5 relevant chunks from the textbook content.
2.  **Given** a user has selected text within a chapter, **When** they ask a question related to the selected text, **Then** the chatbot's answer is derived *only* from the selected text.
3.  **Given** a user has selected text, **When** they ask a question not contained in the selected text, **Then** the chatbot returns "Not contained in the selected text."

---

### User Story 3 - Manage User Account (Priority: P2)

A user wants to create an account, log in securely, and manage their personalized profile information.

**Why this priority**: User authentication and personalization are foundational for delivering tailored learning experiences. It enables persistent preferences and access control.

**Independent Test**: A new user can sign up, provide required profile data, log in successfully, and their profile data is correctly stored and applied for content rendering. This can be tested independently by simulating user registration and login flows against the authentication and user profile services.

**Acceptance Scenarios**:

1.  **Given** a new user wants to access the platform, **When** they sign up using Better-Auth, **Then** they are prompted to provide Hardware experience, Software experience, Robotics background, Preferred language, and Learning goals, and a new user profile is created.
2.  **Given** an existing user, **When** they log in using Better-Auth (JWT/session cookies), **Then** their personalized content settings are automatically applied.
3.  **Given** a logged-in user, **When** they update their profile preferences, **Then** the changes are persisted in Neon and reflected in personalized content rendering.

---

### User Story 4 - Utilize Claude Subagents (Priority: P3)

A user wants to leverage optional Claude Code subagents for tasks like chapter summarization, quiz generation, curriculum planning, hardware advice, or troubleshooting.

**Why this priority**: These are bonus integrations that enhance the learning experience but are not core to the initial MVP of content delivery or chatbot interaction.

**Independent Test**: A user can trigger a subagent (e.g., Chapter Summarizer) for a given chapter, and the subagent returns a JSON-based response via its dedicated API endpoint. This can be tested by calling the agent's API with specific inputs and verifying the JSON output.

**Acceptance Scenarios**:

1.  **Given** a user is viewing a chapter, **When** they request a chapter summary via the "Chapter Summarizer Agent", **Then** the agent processes the chapter content and returns a concise JSON-based summary.
2.  **Given** a user selects a topic, **When** they request a quiz via the "Quiz Generator Agent", **Then** the agent returns a JSON-based quiz with questions and answers.

### Edge Cases

- What happens when a chapter translation is requested but not available/generatable?
- How does the system handle an empty selected text for the chatbot?
- What are the rate limits or error behaviors for the LLM integrations (OpenAI, Claude)?
- How does the system handle invalid user profile data during signup?
- What happens if Qdrant or Neon are unavailable during RAG retrieval or profile updates?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a Docusaurus-based digital textbook titled "Physical AI & Humanoid Robotics".
- **FR-002**: System MUST embed a fully functional RAG Chatbot within the textbook.
- **FR-003**: Backend MUST be implemented using FastAPI, Qdrant vector database, OpenAI embeddings + ChatCompletion/Agents, and Neon Postgres database.
- **FR-004**: System MUST support a "Selected-text answering mode" for the chatbot, responding only from selected text if applicable.
- **FR-005**: System MUST support cross-chapter semantic grounding for the RAG chatbot.
- **FR-006**: System MUST enable personalized content delivery based on user profiles.
- **FR-007**: System MUST provide an Urdu/English translation toggle for all chapters.
- **FR-008**: System MUST implement user authentication via Better-Auth.
- **FR-009**: User signup MUST capture custom profile fields: Hardware experience, Software experience, Robotics background, Preferred language (Urdu/English), Learning goals.
- **FR-010**: Each chapter MUST include a per-chapter personalization button for Beginner, Intermediate, and Advanced modes.
- **FR-011**: Each chapter MUST include a per-chapter Urdu translation button.
- **FR-012**: System MAY include optional Claude Code subagents: Quiz generator, Chapter summarizer, Curriculum generator, Hardware advisor, Troubleshooter. (This is a MAY per the constitution)
- **FR-013**: All system outputs MUST be optimized for clarity, modularity, readability, reusability, AI compatibility, and hackathon scoring rubric.
- **FR-014**: Each chapter MUST include embeddings-friendly chunking, maintain semantic density, and include "context blocks" for RAG indexers.
- **FR-015**: RAG retrieval MUST use OpenAI embeddings for chunks and Qdrant for vector search, retrieving 3-5 relevant chunks per query.
- **FR-016**: Chatbot prompting MUST include system guardrails and disclose if the answer is drawn from selected text vs RAG.
- **FR-017**: Frontend MUST read user profile to display proper difficulty blocks.
- **FR-018**: User personalization preferences MUST be cached in Neon.
- **FR-019**: Urdu translations can be pre-generated or generated via LLM on demand, and on-demand translations MUST be stored in Neon cache.
- **FR-020**: The Urdu/English toggle MUST persist per user.
- **FR-021**: Authentication MUST use JWT/session cookies.
- **FR-022**: Personalized content settings MUST be applied automatically after login.
- **FR-023**: All generated Claude Code subagents MUST use JSON-based responses, dedicated API endpoints, and semantic, deterministic outputs.
- **FR-024**: Frontend (Docusaurus) MUST be deployable to GitHub Pages or Vercel.
- **FR-025**: Backend (FastAPI) MUST be deployable on Render, Railway, Fly.io, or a Docker host.
- **FR-026**: Databases MUST use Qdrant Cloud (free tier) and Neon Serverless (free tier).
- **FR-027**: System MUST generate 19 specific textbook chapters as listed in the constitution, each following documentation style rules.

### Key Entities *(include if feature involves data)*

-   **User**: Represents a platform user with attributes like `hardware_experience`, `software_experience`, `robotics_background`, `preferred_language`, `learning_goals`, and authentication credentials.
-   **Chapter**: Represents a textbook chapter with content, learning objectives, code examples, exercises, diagrams, RAG-friendly summaries, and potentially pre-generated Urdu translations.
-   **PersonalizationSettings**: Stores user-specific preferences for difficulty mode (Beginner, Intermediate, Advanced) and language.
-   **TranslationCache**: Stores on-demand generated Urdu translations for chapters.

## Success Criteria *(mandatory)*

### Measurable Outcomes

-   **SC-001**: 100% of required textbook chapters are generated according to documentation style and AI-native rules.
-   **SC-002**: User signup and login processes are completed successfully within 30 seconds for 95% of users.
-   **SC-003**: Personalized chapter content (difficulty and language) is rendered correctly for 99% of user requests.
-   **SC-004**: RAG Chatbot provides a grounded answer within 5 seconds for 90% of queries.
-   **SC-005**: Selected-text answering mode correctly identifies and answers from selected text, or states "Not contained" for 100% of relevant test cases.
-   **SC-006**: The platform achieves a minimum score of 80% on the hackathon scoring rubric.
-   **SC-007**: Optional Claude subagents (if implemented) return valid JSON responses within 10 seconds for 90% of requests.
