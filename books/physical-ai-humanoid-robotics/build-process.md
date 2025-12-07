# Book Build and Publishing Process

## Overview

This document describes the process for building and publishing the "Introduction to Physical AI & Humanoid Robotics" book in multiple formats.

## Build Environment Setup

### Required Tools
- Python 3.8+
- Node.js and npm (for documentation generators)
- Pandoc (for format conversion)
- LaTeX distribution (for PDF generation)
- Git

### Installation Steps
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (if using documentation generator)
npm install

# Install pandoc (for format conversion)
# Follow installation guide at https://pandoc.org/installing.html
```

## Building Different Formats

### HTML/Web Version
```bash
# Using documentation generator (e.g., Docusaurus, Sphinx)
npm run build
# or
make html
```

### PDF Version
```bash
# Using pandoc
pandoc index.md -o book.pdf --from markdown --template template.tex --listings

# Or using LaTeX directly
pdflatex book.tex
```

### ePub Version
```bash
# Using pandoc
pandoc index.md -o book.epub --from markdown --epub-cover-image=cover.png
```

## Quality Assurance Process

### Content Validation
- Verify all cross-references work correctly
- Check that all code examples are syntactically correct
- Ensure mathematical notation renders properly
- Validate all diagrams and images display correctly

### Testing Code Examples
```bash
# Test Python examples
python -m py_compile code-examples/python/foundations/*.py
python -m py_compile code-examples/python/advanced/*.py
python -m py_compile code-examples/python/implementation/*.py

# Test C++ examples (basic syntax check)
for file in code-examples/cpp/foundations/*.cpp; do
    g++ -c "$file" -o /dev/null
done
```

## Automated Build Script

Create a build script to automate the process:

```bash
#!/bin/bash
# build-book.sh

echo "Starting book build process..."

# Create build directory
mkdir -p build/html build/pdf build/epub

# Generate HTML version
echo "Building HTML version..."
if command -v sphinx-build &> /dev/null; then
    sphinx-build -b html . build/html
elif command -v docusaurus &> /dev/null; then
    docusaurus build
else
    echo "No documentation generator found, copying markdown files"
    cp -r chapters/ build/html/
fi

# Generate PDF version
echo "Building PDF version..."
pandoc index.md -o build/pdf/book.pdf --from markdown --pdf-engine=xelatex

# Generate ePub version
echo "Building ePub version..."
pandoc index.md -o build/epub/book.epub --from markdown --epub-cover-image=cover.png

echo "Build process completed. Output in build/ directory."
```

## Version Control and Publishing

### Git Workflow
```bash
# Create a release branch
git checkout -b release/v1.0

# Update version information
# Build the book in all formats
# Commit the build artifacts
git add build/
git commit -m "Build book v1.0 in all formats"

# Create a tag
git tag v1.0.0

# Push changes
git push origin release/v1.0
git push origin v1.0.0
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Build and Publish Book
on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'

    - name: Install dependencies
      run: |
        pip install pandoc
        # Install other dependencies

    - name: Build book
      run: |
        chmod +x build-book.sh
        ./build-book.sh

    - name: Publish artifacts
      uses: actions/upload-artifact@v2
      with:
        name: book-artifacts
        path: build/
```

## Quality Checks

### Before Publishing
- [ ] All chapters complete and reviewed
- [ ] Code examples tested and functional
- [ ] Diagrams render correctly in all formats
- [ ] Cross-references work properly
- [ ] Mathematical notation displays correctly
- [ ] Urdu translations complete and accurate
- [ ] Index comprehensive and accurate
- [ ] Table of contents matches content

### Automated Checks
```bash
# Check for broken links
find . -name "*.md" -exec grep -l "\[.*\](.*\.md)" {} \; | xargs -I {} sh -c 'grep -o "\[.*\](.*\.md)" "$1" | sed "s/.*\](\(.*\))$/\1/"' _ {} | while read link; do
    if [ ! -f "$link" ]; then
        echo "Broken link: $link"
    fi
done

# Check for undefined references
grep -r "{{.*}}" chapters/
```

## Deployment

### Web Hosting
```bash
# Deploy to GitHub Pages
gh-pages -d build/html

# Or deploy to custom server
rsync -avz build/html/ user@server:/path/to/web/
```

### Package Distribution
Create distribution packages for different formats:
```bash
# Create source package
tar -czf book-source-v1.0.0.tar.gz --exclude='.git' --exclude='build' .

# Create binary packages
zip -r book-html-v1.0.0.zip build/html/
tar -czf book-pdf-v1.0.0.tar.gz build/pdf/
```

## Maintenance

### Updating Content
1. Make changes to source markdown files
2. Update index and cross-references as needed
3. Run quality assurance checks
4. Rebuild all formats
5. Test updated content
6. Publish updated versions

### Versioning
Follow semantic versioning:
- Major version: Significant content additions or structural changes
- Minor version: New chapters or significant updates to existing content
- Patch version: Corrections, typo fixes, and minor improvements

This process ensures consistent, high-quality output across all formats and maintains the integrity of the book as a comprehensive educational resource.