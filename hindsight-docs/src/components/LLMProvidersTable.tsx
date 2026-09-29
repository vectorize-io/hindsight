import React from 'react';
import {LLM_PROVIDERS} from '../data/llmProviders';

/**
 * Renders the "Provider Default Models" table from the single-source-of-truth
 * provider list in `src/data/llmProviders.tsx`. Skips entries without a default
 * model (e.g. the "OpenAI Compatible" pseudo-entry).
 */
export function LLMProvidersTable() {
  const rows = LLM_PROVIDERS.filter(p => p.id && p.defaultModel);
  return (
    <table>
      <thead>
        <tr>
          <th>Provider</th>
          <th>Default Model</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({id, defaultModel, defaultModelNote}) => (
          <tr key={id}>
            <td><code>{id}</code></td>
            <td>
              <code>{defaultModel}</code>
              {defaultModelNote && <> ({defaultModelNote})</>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
