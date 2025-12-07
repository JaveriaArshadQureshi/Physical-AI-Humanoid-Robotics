# Data Model: Physical AI & Humanoid Robotics Textbook Platform

**Feature Branch**: `001-ai-robotics-platform` | **Date**: 2025-12-05

## Core Entities

### User
Represents a platform user. This entity is central to authentication, personalization, and tracking learning progress.

**Attributes**:
-   `user_id`: Unique identifier for the user.
-   `email`: User's email address (for authentication).
-   `hashed_password`: Securely stored password hash.
-   `hardware_experience`: User's experience level with hardware (e.g., Beginner, Intermediate, Advanced).
-   `software_experience`: User's experience level with software (e.g., Beginner, Intermediate, Advanced).
-   `robotics_background`: User's background in robotics (e.g., None, Academic, Professional).
-   `preferred_language`: User's preferred language for content (e.g., English, Urdu).
-   `learning_goals`: User's stated learning objectives for the platform.
-   `created_at`: Timestamp of user creation.
-   `updated_at`: Timestamp of last profile update.

### Chapter
Represents a single chapter within the "Physical AI & Humanoid Robotics" digital textbook.

**Attributes**:
-   `chapter_id`: Unique identifier for the chapter.
-   `title`: The title of the chapter.
-   `content_english`: The main English content of the chapter.
-   `content_urdu_pregenerated`: Pre-generated Urdu translation of the chapter content (optional).
-   `learning_objectives`: Key learning goals for the chapter.
-   `core_theory`: Theoretical concepts covered.
-   `practical_examples`: Practical demonstrations or use cases.
-   `code_blocks`: Code snippets (e.g., ROS2/Isaac Sim code).
-   `diagrams_ascii`: ASCII descriptions of diagrams.
-   `exercises`: Questions or tasks for practice.
-   `rag_summary`: A concise summary of the chapter, optimized for RAG indexing.
-   `chunking_metadata`: Information for embeddings-friendly chunking.
-   `personalization_variants`: Structure to hold content variations for different difficulty levels (Beginner, Intermediate, Advanced).

### PersonalizationSettings
Stores user-specific preferences for content personalization, persisting their chosen difficulty and language across sessions.

**Attributes**:
-   `setting_id`: Unique identifier for the personalization setting.
-   `user_id`: Foreign key linking to the User entity.
-   `difficulty_mode`: User's preferred difficulty level (Beginner, Intermediate, Advanced).
-   `display_language`: User's currently active display language (English, Urdu).

### TranslationCache
Caches on-demand generated Urdu translations to improve performance and reduce LLM calls for repeated requests.

**Attributes**:
-   `cache_id`: Unique identifier for the cache entry.
-   `chapter_id`: Foreign key linking to the Chapter entity.
-   `content_english_hash`: Hash of the original English content to check for updates.
-   `content_urdu_generated`: The LLM-generated Urdu translation.
-   `generated_at`: Timestamp of when the translation was generated.
