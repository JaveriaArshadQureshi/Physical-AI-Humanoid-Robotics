# Data Model: Introduction to Physical AI & Humanoid Robotics

## Book Chapter Entity

**Definition**: A self-contained unit of content focusing on a specific aspect of Physical AI & Humanoid Robotics

**Attributes**:
- `id`: Unique identifier for the chapter (e.g., "01-introduction-to-physical-ai")
- `title`: Display title of the chapter
- `subtitle`: Subtitle providing additional context
- `content`: Markdown content of the chapter
- `description`: Brief summary of chapter content
- `position`: Order in the book hierarchy (1-19)
- `category`: Main topic area (e.g., "foundations", "simulation", "control", "implementation")
- `prerequisites`: List of chapters or concepts needed before this chapter
- `learning_objectives`: List of skills/knowledge students will gain
- `technical_requirements`: Software/hardware needed for practical exercises
- `code_examples`: List of code files associated with the chapter
- `diagrams`: List of diagrams associated with the chapter
- `exercises`: List of practical exercises included in the chapter
- `cross_references`: List of related chapters for navigation
- `translations`: Available language versions

## Technical Concept Entity

**Definition**: A fundamental idea, algorithm, or methodology within the robotics domain

**Attributes**:
- `id`: Unique identifier for the concept
- `name`: Name of the concept
- `definition`: Clear definition of the concept
- `mathematical_foundation`: Mathematical representation where applicable
- `implementation`: How the concept is implemented in practice
- `related_concepts`: List of related technical concepts
- `chapter_origin`: Chapter where this concept is first introduced
- `use_cases`: Practical applications of the concept

## Implementation Guide Entity

**Definition**: Practical steps and code examples for implementing concepts in real systems

**Attributes**:
- `id`: Unique identifier for the guide
- `title`: Title of the implementation guide
- `concept`: The technical concept being implemented
- `prerequisites`: What is needed before starting this guide
- `steps`: Sequential steps for implementation
- `code_language`: Programming language used (Python, C++, etc.)
- `platform`: Target platform (ROS2, Isaac Sim, real hardware, etc.)
- `expected_outcome`: What the implementation should achieve
- `troubleshooting`: Common issues and solutions
- `extensions`: Ways to expand or modify the implementation

## Cross-Reference Entity

**Definition**: Linkage between related concepts across different chapters

**Attributes**:
- `source_chapter`: Chapter containing the reference
- `target_chapter`: Chapter being referenced
- `reference_type`: Type of relationship (prerequisite, related, extension, etc.)
- `description`: Brief description of the relationship
- `position`: Position in the source chapter where the reference appears

## Translation Unit Entity

**Definition**: Content that needs to be translated between different languages (English/Urdu)

**Attributes**:
- `id`: Unique identifier for the translation unit
- `source_language`: Language of the original content
- `target_language`: Language to translate to
- `content_type`: Type of content (text, code comment, diagram label, etc.)
- `original_content`: Original content to be translated
- `translated_content`: Translated content
- `validation_status`: Whether the translation has been validated
- `translator`: Who performed the translation