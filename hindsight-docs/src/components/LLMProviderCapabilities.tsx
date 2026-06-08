import React from 'react';
import {LLM_PROVIDERS} from '../data/llmProviders';

/**
 * Renders the "Provider Capabilities" table from the single-source-of-truth
 * provider list in `src/data/llmProviders.tsx`. Adding a provider (or a
 * capability flag) is an edit to `llmProviders.json` only — this table, the
 * default-models table, and the icon grid all derive from it.
 *
 * Only real providers are listed (the empty-id "OpenAI Compatible" pseudo-entry
 * is skipped). A blank cell means the capability is not supported.
 */
export function LLMProviderCapabilities() {
  const rows = LLM_PROVIDERS.filter(p => p.id);
  const cell = (on?: boolean) => (on ? '✅' : '—');
  return (
    <table>
      <thead>
        <tr>
          <th>Provider</th>
          <th style={{textAlign: 'center'}}>Batch API</th>
          <th style={{textAlign: 'center'}}>Explicit prompt caching</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({id, label, batchApi, promptCaching}) => (
          <tr key={id}>
            <td>{label} (<code>{id}</code>)</td>
            <td style={{textAlign: 'center'}}>{cell(batchApi)}</td>
            <td style={{textAlign: 'center'}}>{cell(promptCaching)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
