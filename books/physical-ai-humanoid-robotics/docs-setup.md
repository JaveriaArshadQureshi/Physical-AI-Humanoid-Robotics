# Documentation Toolchain Setup

## Overview

This document describes the setup for the documentation toolchain for the "Introduction to Physical AI & Humanoid Robotics" book. The toolchain enables building the book in multiple formats (HTML, PDF, ePub) and provides a web-based interface for online reading.

## Toolchain Components

### 1. Markdown-based Content

The book content is written in Markdown format for maximum portability and version control friendliness.

### 2. Build Tools

#### Pandoc
- Universal document converter
- Converts Markdown to various formats (PDF, ePub, HTML, etc.)
- Handles cross-references and mathematical notation

#### Sphinx (Recommended)
- Python documentation generator
- Excellent for technical documentation
- Supports mathematical notation via MathJax
- Extensible with plugins

#### Docusaurus (Alternative)
- React-based documentation platform
- Modern web interface
- Good search capabilities
- Versioning support

### 3. Mathematical Rendering

- MathJax for web-based mathematical notation
- LaTeX for PDF mathematical notation
- SVG diagrams for complex visualizations

## Setup Instructions

### Option 1: Basic Pandoc Setup

1. Install Pandoc: https://pandoc.org/installing.html
2. Install LaTeX (for PDF generation): https://www.tug.org/texlive/
3. Use the provided build script in `build/build-script.sh`

### Option 2: Sphinx Setup

1. Install Python 3.8+
2. Install Sphinx:
   ```bash
   pip install sphinx
   pip install sphinx-book-theme
   pip install sphinx-math-dollar
   ```

3. Create Sphinx configuration:
   ```bash
   cd books/physical-ai-humanoid-robotics
   mkdir docs
   cd docs
   sphinx-quickstart
   ```

4. Configure `conf.py`:
   ```python
   extensions = [
       'sphinx.ext.mathjax',
       'sphinx_math_dollar'
   ]

   html_theme = 'sphinx_book_theme'
   ```

### Option 3: Docusaurus Setup

1. Install Node.js and npm
2. Create Docusaurus project:
   ```bash
   npx create-docusaurus@latest books/physical-ai-humanoid-robotics/website classic
   cd books/physical-ai-humanoid-robotics/website
   ```

3. Configure `docusaurus.config.js` to include book content

## Directory Structure

```
books/physical-ai-humanoid-robotics/
├── docs/                 # Sphinx documentation source
│   ├── conf.py          # Sphinx configuration
│   ├── index.rst        # Main documentation entry point
│   └── ...              # Chapter files
├── website/             # Docusaurus documentation
│   ├── docs/            # Markdown documentation files
│   ├── src/             # Custom components
│   ├── docusaurus.config.js
│   └── package.json
├── build/               # Build outputs
│   ├── html/            # HTML output
│   ├── pdf/             # PDF output
│   └── epub/            # ePub output
├── chapters/            # Source Markdown chapters
├── code-examples/       # Code examples
└── diagrams/            # Diagrams and images
```

## Build Commands

### For Sphinx:
```bash
cd books/physical-ai-humanoid-robotics/docs
make html          # Build HTML
make latexpdf      # Build PDF
```

### For Docusaurus:
```bash
cd books/physical-ai-humanoid-robotics/website
npm run build      # Build static site
npm run serve      # Serve locally
```

### For Pandoc (using provided script):
```bash
cd books/physical-ai-humanoid-robotics
./build/build-script.sh
```

## Configuration Files

### Sphinx conf.py Example
```python
# Configuration file for the Sphinx documentation builder.

project = 'Introduction to Physical AI & Humanoid Robotics'
copyright = '2025, Author Name'
author = 'Author Name'

extensions = [
    'sphinx.ext.mathjax',
    'sphinx_math_dollar'
]

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

mathjax3_config = {
    'tex': {
        'inlineMath': [['\\(', '\\)']],
        'displayMath': [['\\[', '\\]']],
    }
}
```

### Docusaurus docusaurus.config.js Example
```javascript
// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.

import {themes as prismThemes} from '@docusaurus/prism';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Introduction to Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive guide to Physical AI and Humanoid Robotics',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-book-site.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/physical-ai-humanoid-robotics/',

  // GitHub pages deployment config.
  organizationName: 'your-username', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-robotics', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {
            href: 'https://github.com/your-username/physical-ai-humanoid-robotics',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Book Chapters',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/humanoid-robotics',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-username/physical-ai-humanoid-robotics',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Introduction to Physical AI & Humanoid Robotics. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
```

## Deployment

### GitHub Pages
1. Build the documentation
2. Push to `gh-pages` branch or enable GitHub Pages in repository settings

### Other Platforms
- Netlify: Connect GitHub repository and deploy automatically
- Vercel: Connect GitHub repository and deploy automatically
- AWS S3: Upload built files to S3 bucket with static website hosting

## Maintenance

The documentation toolchain should be maintained with:
- Regular updates to dependencies
- Testing of build process
- Verification of all links and cross-references
- Accessibility compliance checks
- Mobile responsiveness testing

This documentation toolchain provides a solid foundation for publishing the Physical AI & Humanoid Robotics book in multiple formats and making it accessible to a wide audience.