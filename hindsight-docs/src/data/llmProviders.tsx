import type {IconType} from 'react-icons';
import React from 'react';
import {SiOpenai, SiAnthropic, SiGooglegemini, SiOllama} from 'react-icons/si';
import {LuTerminal, LuZap, LuBrainCog, LuSparkles, LuGlobe, LuLayers, LuCloud} from 'react-icons/lu';
import providersJson from './llmProviders.json';

const OpenAICompatibleIcon: IconType = ({size = 28, ...props}) => (
  <span style={{position: 'relative', display: 'inline-flex'}}>
    <SiOpenai size={size} {...props} />
    <span style={{
      position: 'absolute', bottom: -3, right: -6,
      fontSize: Math.round((size as number) * 0.5), fontWeight: 900, lineHeight: 1,
      color: 'currentColor',
    }}>+</span>
  </span>
);

const ICON_REGISTRY: Record<string, IconType> = {
  openai: SiOpenai,
  anthropic: SiAnthropic,
  gemini: SiGooglegemini,
  ollama: SiOllama,
  terminal: LuTerminal,
  zap: LuZap,
  brain: LuBrainCog,
  sparkles: LuSparkles,
  globe: LuGlobe,
  layers: LuLayers,
  cloud: LuCloud,
  'openai-compatible': OpenAICompatibleIcon,
};

export interface LLMProvider {
  /** HINDSIGHT_API_LLM_PROVIDER value, e.g. "deepseek". Empty string for the
   *  "OpenAI Compatible" pseudo-entry which is not a real provider id. */
  id: string;
  /** Display name shown in the grid tile and table. */
  label: string;
  /** Icon component rendered in the grid tile. */
  icon: IconType;
  /** Provider default model. Undefined = no entry in the default-models table. */
  defaultModel?: string;
  /** Optional note rendered in the default-models table. */
  defaultModelNote?: string;
  /** Supports the asynchronous Batch API (supports_batch_api in the engine). */
  batchApi?: boolean;
  /** Supports explicit prompt-prefix caching (supports_prompt_caching in the engine). */
  promptCaching?: boolean;
}

/**
 * Single source of truth for the supported LLM providers shown in the docs.
 *
 * Adding a provider means editing `llmProviders.json` ONLY. That file is
 * consumed by:
 *   - this module (resolves iconKey -> IconType for the icon grid)
 *   - LLMProvidersTable React component (renders the default-models table)
 *   - LLMProviderCapabilities React component (renders the capability table:
 *     batchApi / promptCaching)
 *   - scripts/generate-docs-skill.sh (renders <LLMProvidersTable />,
 *     <LLMProvidersGrid /> and <LLMProviderCapabilities /> as markdown when
 *     copying MDX docs into the agent-facing skill)
 *
 * Keep the default model aligned with PROVIDER_DEFAULT_MODELS, and the
 * capability flags with supports_batch_api() / supports_prompt_caching() on the
 * provider classes, in hindsight-api-slim/hindsight_api/.
 */
export const LLM_PROVIDERS: LLMProvider[] = (providersJson as Array<{
  id: string; label: string; iconKey: string; defaultModel?: string; defaultModelNote?: string;
  batchApi?: boolean; promptCaching?: boolean;
}>).map(({iconKey, ...rest}) => {
  const icon = ICON_REGISTRY[iconKey];
  if (!icon) throw new Error(`Unknown iconKey "${iconKey}" for provider "${rest.id || rest.label}"`);
  return {...rest, icon};
});
