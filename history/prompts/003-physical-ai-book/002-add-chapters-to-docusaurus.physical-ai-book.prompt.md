---
id: 001
title: "Add Book Chapters to Docusaurus Website"
stage: "green"
date_iso: "2025-12-06"
surface: "agent"
model: "claude-haiku-4-5-20251001"
feature: "003-physical-ai-book"
branch: "003-physical-ai-book"
user: "sam72"
command: "/sp.implement"
labels: ["docusaurus", "documentation", "book", "physical-ai", "humanoid-robotics"]
links:
  spec: "null"
  ticket: "null"
  adr: "null"
  pr: "null"
files_yaml: |
  - my-docs/docusaurus.config.ts
  - my-docs/sidebars.ts
  - my-docs/docs/intro.md
  - my-docs/docs/01-introduction-to-physical-ai.md
  - my-docs/docs/02-linux-ros2-foundations.md
  - my-docs/docs/03-gazebo-simulation.md
  - my-docs/docs/04-nvidia-isaac-sim.md
  - my-docs/docs/05-real-robot-control-architecture.md
  - my-docs/docs/06-sensor-fusion-localization.md
  - my-docs/docs/07-kinematics-dynamics.md
  - my-docs/docs/08-control-systems.md
  - my-docs/docs/09-robot-perception.md
  - my-docs/docs/10-vision-language-action-models.md
  - my-docs/docs/11-reinforcement-learning-robotics.md
  - my-docs/docs/12-imitation-learning-teleoperation.md
  - my-docs/docs/13-building-humanoid-actuators.md
  - my-docs/docs/14-autonomous-navigation-humanoids.md
  - my-docs/docs/15-safety-edge-computing.md
  - my-docs/docs/16-capstone-project-guide.md
tests_yaml: |
  - Docusaurus site builds successfully
  - All 16 chapters accessible through navigation
  - Sidebar structure correctly organized
  - Site content properly updated from book content
prompt_text: |
  add chapters or modulas in docusuarus website and replace tutorial with chapters or modulas inside in slider add replace Tutorial Intro to book intro and replace chapter with Tutorial Basics  like this add all chapter content add
response_text: |
  Successfully updated the Docusaurus website to feature the Physical AI & Humanoid Robotics book:
  - Updated docusaurus.config.ts with book title, tagline, and navigation
  - Replaced Tutorial Intro with comprehensive book introduction
  - Created new sidebar structure organized by chapter categories (Foundational, Advanced Technical, Implementation)
  - Copied all 16 book chapters to the docs directory
  - Updated navigation to use "Chapters" instead of "Tutorial"
  - Removed old tutorial content (tutorial-basics, tutorial-extras, new-chapter)
  - Updated footer and site elements to reflect book content
  - Fixed sidebar document IDs to match Docusaurus naming conventions (removed numeric prefixes)
  - Tested the site successfully (running on port 3001)
outcome: "The Docusaurus website now features the complete Physical AI & Humanoid Robotics book with all 16 chapters organized in a structured sidebar."
evaluation: "The implementation successfully transformed the generic Docusaurus tutorial site into a comprehensive book website with all chapters properly integrated and navigable."
---

# Implementation: Adding Physical AI & Humanoid Robotics Book to Docusaurus

## Summary of Changes

This implementation successfully transformed the Docusaurus tutorial site into a comprehensive website for the "Introduction to Physical AI & Humanoid Robotics" book. The following changes were made:

### Configuration Updates
- Updated `docusaurus.config.ts` to reflect the book's title, tagline, and branding
- Changed navigation from "Tutorial" to "Chapters" with appropriate sidebar ID
- Updated footer links to be relevant to robotics and physical AI
- Added Urdu language support to internationalization settings

### Content Replacement
- Replaced the generic "Tutorial Intro" with a comprehensive book introduction
- Copied all 16 book chapters from the books/physical-ai-humanoid-robotics/chapters directory to the docs directory
- Removed old tutorial content (tutorial-basics, tutorial-extras, and new-chapter files)

### Navigation Structure
- Created a new sidebar structure organized into three main categories:
  1. Foundational Chapters (5 chapters)
  2. Advanced Technical Chapters (7 chapters)
  3. Implementation and Specialized Topics (4 chapters)
- Fixed document IDs in the sidebar to match Docusaurus naming conventions (removed numeric prefixes)

### Site Elements
- Updated navbar title to "Physical AI & Robotics"
- Updated footer to reference the book content and relevant robotics resources
- Maintained the custom CSS styling while ensuring it works with the new content

## Technical Details

The implementation required adjusting the sidebar document IDs to match Docusaurus conventions. When files are named with numeric prefixes like "01-introduction-to-physical-ai.md", Docusaurus uses "introduction-to-physical-ai" as the document ID (removing the numeric prefix). This was corrected in the sidebars.ts file.

## Verification

The updated Docusaurus site was successfully tested and confirmed to be running at http://localhost:3001. All 16 chapters are accessible through the organized sidebar navigation, and the site properly displays the book content instead of the original tutorial content.

The transformation is complete and the Docusaurus website now serves as a comprehensive platform for the Physical AI & Humanoid Robotics book.