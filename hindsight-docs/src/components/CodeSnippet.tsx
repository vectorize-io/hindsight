import React from 'react';
import CodeBlock from '@theme/CodeBlock';

interface CodeSnippetProps {
  /** Raw file content (use raw-loader to import) */
  code: string;
  /** Section marker name (e.g., "retain-basic" for [docs:retain-basic]) */
  section: string;
  /** Language for syntax highlighting */
  language: string;
  /** Optional title for the code block */
  title?: string;
}

/**
 * Extracts a marked section from source code.
 *
 * Markers are in the format:
 * - Start: `# [docs:section-name]` (Python/Bash) or `// [docs:section-name]` (JS/TS)
 * - End: `# [/docs:section-name]` (Python/Bash) or `// [/docs:section-name]` (JS/TS)
 */
function extractSection(code: string, section: string): string {
  // Match both Python/Bash (#) and JS/TS (//) comment styles
  const startPattern = new RegExp(`(?:#|//)\\s*\\[docs:${section}\\]`);
  const endPattern = new RegExp(`(?:#|//)\\s*\\[/docs:${section}\\]`);

  const lines = code.split('\n');
  let inSection = false;
  const sectionLines: string[] = [];

  for (const line of lines) {
    if (startPattern.test(line)) {
      inSection = true;
      continue;
    }
    if (endPattern.test(line)) {
      inSection = false;
      continue;
    }
    if (inSection) {
      sectionLines.push(line);
    }
  }

  if (sectionLines.length === 0) {
    console.warn(`CodeSnippet: Section "${section}" not found in code`);
    return `// Section "${section}" not found`;
  }

  // Trim leading/trailing empty lines and normalize indentation
  return trimAndNormalize(sectionLines);
}

/**
 * Trims leading/trailing empty lines and removes common leading indentation.
 */
function trimAndNormalize(lines: string[]): string {
  // Remove leading empty lines
  while (lines.length > 0 && lines[0].trim() === '') {
    lines.shift();
  }
  // Remove trailing empty lines
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
    lines.pop();
  }

  if (lines.length === 0) return '';

  // Find minimum indentation (ignoring empty lines)
  const nonEmptyLines = lines.filter(l => l.trim() !== '');
  if (nonEmptyLines.length === 0) return '';

  const minIndent = Math.min(
    ...nonEmptyLines.map(line => {
      const match = line.match(/^(\s*)/);
      return match ? match[1].length : 0;
    })
  );

  // Remove common indentation
  return lines
    .map(line => line.slice(minIndent))
    .join('\n');
}

/**
 * CodeSnippet component for embedding code from example files.
 *
 * Usage in MDX:
 * ```mdx
 * import CodeSnippet from '@site/src/components/CodeSnippet';
 * import retainPy from '!!raw-loader!@site/examples/api/retain.py';
 *
 * <CodeSnippet code={retainPy} section="retain-basic" language="python" />
 * ```
 */
export default function CodeSnippet({
  code,
  section,
  language,
  title
}: CodeSnippetProps): React.ReactElement {
  const extractedCode = extractSection(code, section);

  return (
    <CodeBlock language={language} title={title}>
      {extractedCode}
    </CodeBlock>
  );
}
