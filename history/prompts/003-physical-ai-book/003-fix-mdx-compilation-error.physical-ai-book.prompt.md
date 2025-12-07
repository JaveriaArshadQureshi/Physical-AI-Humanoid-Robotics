---
id: 003
title: "Fix MDX Compilation Error in Imitation Learning Chapter"
stage: "green"
date_iso: "2025-12-06"
surface: "agent"
model: "claude-haiku-4-5-20251001"
feature: "003-physical-ai-book"
branch: "003-physical-ai-book"
user: "sam72"
command: "manual fix"
labels: ["docusaurus", "mdx", "compilation", "bug-fix", "physical-ai", "humanoid-robotics"]
links:
  spec: "null"
  ticket: "null"
  adr: "null"
  pr: "null"
files_yaml: |
  - my-docs/docs/12-imitation-learning-teleoperation.md
tests_yaml: |
  - Docusaurus site builds successfully
  - MDX compilation passes without errors
  - Chapter content displays correctly in the website
prompt_text: |
  Compiled with problems:
  ×
  ERROR in ./docs/12-imitation-learning-teleoperation.md
  Module build failed (from ./node_modules/@docusaurus/mdx-loader/lib/index.js):
  Error: MDX compilation failed for file "C:\Users\sam72\Desktop\hacbook\my-docs\docs\12-imitation-learning-teleoperation.md"
  Cause: Could not parse expression with acorn
  Details:
  {
    "cause": {
      "pos": 1586,
      "loc": {
        "line": 37,
        "column": 103
      },
      "raisedAt": 2
    },
    "column": 104,
    "file": "",
    "message": "Could not parse expression with acorn",
    "line": 37,
    "name": "37:104",
    "place": {
      "line": 37,
      "column": 104,
      "offset": 1586
    },
    "reason": "Could not parse expression with acorn",
    "ruleId": "acorn",
    "source": "micromark-extension-mdx-expression",
    "url": "https://github.com/micromark/micromark-extension-mdx-expression/tree/main/packages/micromark-extension-mdx-expression#could-not-parse-expression-with-acorn"
  }
response_text: |
  Fixed the MDX compilation error in the imitation learning chapter by replacing special mathematical characters with inline code formatting. The original issue was caused by special Unicode characters like π, subscripts (₁, ₂, ₙ), and other mathematical symbols that the MDX parser interpreted as potential expressions. The fix involved:
  - Changed mathematical expressions from using special characters to using inline code formatting
  - Replaced `π(a|s)` and similar expressions with `π(a|s)` using backticks
  - Maintained the mathematical meaning while avoiding parsing conflicts
  - Verified that the Docusaurus site now builds and runs without errors
outcome: "The MDX compilation error has been resolved and the Docusaurus site is running successfully."
evaluation: "The fix successfully resolved the parsing issue while preserving the mathematical content of the chapter."
---

# Fix: MDX Compilation Error in Imitation Learning Chapter

## Problem Description

The Docusaurus build was failing with an MDX compilation error in the file `12-imitation-learning-teleoperation.md` at line 37, column 104. The error was:

```
Error: MDX compilation failed for file "C:\Users\sam72\Desktop\hacbook\my-docs\docs\12-imitation-learning-teleoperation.md"
Cause: Could not parse expression with acorn
```

## Root Cause

The issue was caused by special Unicode characters in mathematical expressions:
- Greek letter π (pi)
- Subscript characters (₁, ₂, ₙ, etc.)
- Other mathematical symbols

These characters were being misinterpreted by the MDX parser as potential JavaScript expressions, causing the Acorn JavaScript parser to fail.

## Solution

The solution involved changing the mathematical expressions from using special characters to using inline code formatting:

**Before:**
```
Imitation learning addresses the problem of learning a policy π(a|s) from expert demonstrations D = {(s₁, a₁), (s₂, a₂), ..., (sₙ, aₙ)}.

The goal is to minimize the difference between the expert policy πₑ and the learned policy π:
min_π E[||π(a|s) - πₑ(a|s)||]
```

**After:**
```
Imitation learning addresses the problem of learning a policy `π(a|s)` from expert demonstrations `D = {(s_1, a_1), (s_2, a_2), ..., (s_n, a_n)}`.

The goal is to minimize the difference between the expert policy `π_e` and the learned policy `π`:
`min_π E[||π(a|s) - π_e(a|s)||]`
```

## Changes Made

1. Replaced special mathematical characters with standard ASCII equivalents
2. Wrapped mathematical expressions in inline code formatting (backticks)
3. Used underscore notation for subscripts (e.g., `s_1` instead of `s₁`)
4. Maintained the mathematical meaning and readability of the content

## Verification

- Docusaurus site builds successfully
- MDX compilation passes without errors
- Chapter content displays correctly in the website
- The site runs at http://localhost:3002

The fix preserves the mathematical content while avoiding parsing conflicts with the MDX parser.