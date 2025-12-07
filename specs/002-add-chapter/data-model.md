# Data Model: Add Chapter to Book of Docusaurus

## Chapter Entity

**Definition**: A documentation unit containing content in markdown format with metadata for navigation and organization

**Attributes**:
- `id`: Unique identifier for the chapter (auto-generated from filename)
- `title`: Display title of the chapter
- `content`: Markdown content of the chapter
- `description`: Brief summary of the chapter content
- `position`: Order in the navigation hierarchy
- `category`: Grouping category for the chapter (e.g., tutorial, reference, guide)
- `tags`: Array of keywords for search and categorization
- `authors`: List of content contributors
- `lastUpdated`: Timestamp of last modification

## Navigation Structure

**Definition**: The hierarchical organization of chapters that appears in the sidebar and top navigation

**Attributes**:
- `type`: Navigation element type (doc, category, link)
- `label`: Display name in navigation
- `items`: Array of child navigation items
- `link`: Associated document or external URL
- `collapsed`: Whether the category is collapsed by default in sidebar

## Content Metadata

**Definition**: Information about the chapter including title, description, and positioning in the documentation hierarchy

**Attributes**:
- `title`: Page title for SEO and browser tab
- `description`: Meta description for SEO
- `keywords`: SEO keywords
- `sidebar_label`: Label to appear in sidebar (may differ from page title)
- `sidebar_position`: Order position in sidebar
- `pagination_next`: Next page in documentation sequence
- `pagination_prev`: Previous page in documentation sequence