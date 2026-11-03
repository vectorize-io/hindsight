import type {IconType} from 'react-icons';
import {SiAlibabacloud, SiCloudflare, SiGooglecloud, SiHuggingface, SiNvidia, SiOnnx} from 'react-icons/si';
import {LLM_PROVIDERS} from './llmProviders';

/**
 * Every provider on the Models page, for its search grid. LLM entries come from
 * `llmProviders.json` (the same list behind the LLM tables); embedding, reranker and
 * decision-model entries are listed here. `anchor` is the section on the Models page
 * that sets the provider up.
 */
export type ModelKind = 'llm' | 'embeddings' | 'reranker' | 'decision';

export interface ModelProvider {
  id: string;
  label: string;
  kind: ModelKind;
  anchor: string;
  icon?: IconType;
  imgSrc?: string;
  /** Dark-theme image, for a monochrome mark that would vanish on the dark surface. */
  imgSrcDark?: string;
  /** The model used when none is set. */
  defaultModel?: string;
  /** LLM only: Hindsight submits bulk retain through the provider's Batch API. */
  batchApi?: boolean;
  /** LLM only: Hindsight caches the fixed prompt prefix explicitly. */
  promptCaching?: boolean;
}

export const KIND_LABELS: Record<ModelKind, string> = {
  llm: 'LLM',
  embeddings: 'Embeddings',
  reranker: 'Reranker',
  decision: 'Decision model',
};

// LLM providers whose section predates the llm-<id> anchors; the rest are #llm-<id>.
const LLM_ANCHORS: Record<string, string> = {
  openai: 'openai-and-compatible-endpoints',
  'openai-responses': 'openai-responses-api',
  vertexai: 'vertex-ai-setup-google-cloud',
  bedrock: 'aws-bedrock',
  llamacpp: 'built-in-llamacpp',
  litellm: 'litellm-and-litellm-router',
  meta: 'meta-model-api-setup',
  nous: 'nous-portal-setup-hermes',
  'github-copilot': 'github-copilot-setup',
  chatgpt: 'chatgpt-setup-sign-in-with-chatgpt',
  'openai-codex': 'openai-codex-setup-chatgpt-pluspro',
  'claude-code': 'claude-code-setup-claude-promax',
  cursor: 'cursor-setup-cursor-subscription',
  'xai-oauth': 'supergrok-subscription-setup-device-code-oauth',
};

const HINDSIGHT_ICON = '/img/icons/hindsight.png';
const llmIcon = (id: string) => LLM_PROVIDERS.find((p) => p.id === id)?.icon;

// Brand marks by provider id, for any kind: an image path or a Simple Icons mark. Wins over
// the generic icon llmProviders.json gives a provider with no mark of its own.
const BRAND: Record<string, string | {light: string; dark: string} | IconType> = {
  groq: '/img/icons/providers/groq.png',
  deepseek: '/img/icons/providers/deepseek.webp',
  minimax: '/img/icons/providers/minimax.webp',
  zai: '/img/icons/providers/zai.webp',
  'azure-openai': '/img/icons/providers/microsoft.webp',
  local: '/img/icons/providers/sentence-transformers.webp',
  openrouter: '/img/icons/providers/openrouter.png',
  cohere: '/img/icons/providers/cohere.png',
  zeroentropy: '/img/icons/providers/zeroentropy.png',
  'jina-mlx': '/img/icons/providers/jina.png',
  typesafe: '/img/icons/providers/typesafe.webp',
  cursor: '/img/icons/cursor.svg',
  'github-copilot': '/img/icons/github-copilot.svg',
  'claude-code': '/img/icons/claude-code.png',
  chatgpt: '/img/icons/chatgpt.svg',
  'openai-codex': '/img/icons/codex.svg',
  nous: '/img/icons/hermes.png',
  litellm: '/img/icons/litellm.png',
  'opencode-go': '/img/icons/opencode.png',
  'xai-oauth': '/img/icons/grok-build.svg',
  volcano: '/img/icons/providers/volcengine-color.png',
  fireworks: '/img/icons/providers/fireworks-color.png',
  bedrock: '/img/icons/providers/bedrock-color.png',
  siliconflow: '/img/icons/providers/siliconcloud-color.png',
  lmstudio: {light: '/img/icons/providers/lmstudio.png', dark: '/img/icons/providers/lmstudio-dark.png'},
  atlas: {light: '/img/icons/providers/atlascloud.png', dark: '/img/icons/providers/atlascloud-dark.png'},
  vertexai: SiGooglecloud,
  nvidia: SiNvidia,
  alibaba: SiAlibabacloud,
  cloudflare: SiCloudflare,
  onnx: SiOnnx,
  tei: SiHuggingface,
};

const withBrand = (p: ModelProvider): ModelProvider => {
  const b = BRAND[p.id];
  if (!b) return p;
  if (typeof b === 'string') return {...p, imgSrc: b, icon: undefined};
  if ('light' in b) return {...p, imgSrc: b.light, imgSrcDark: b.dark, icon: undefined};
  return {...p, icon: b, imgSrc: undefined};
};

const llm: ModelProvider[] = [
  ...LLM_PROVIDERS.map((p) => ({
    id: p.id || 'openai-compatible',
    label: p.label,
    kind: 'llm' as const,
    anchor: p.id ? (LLM_ANCHORS[p.id] ?? `llm-${p.id}`) : 'openai-and-compatible-endpoints',
    icon: p.icon,
    defaultModel: p.defaultModel,
    batchApi: p.batchApi,
    promptCaching: p.promptCaching,
  })),
  {id: 'azure-openai', label: 'Azure OpenAI', kind: 'llm', anchor: 'azure-openai-setup', icon: llmIcon('openai')},
];

// [id, label, default model, icon source: an llmProviders id or an image path]. Each one's section
// on the Models page is #<kind>-<id>, except where ANCHOR_OVERRIDES says otherwise.
type Row = [string, string, string?, string?];

const ANCHOR_OVERRIDES: Record<string, string> = {'embeddings-bedrock': 'embeddings-litellm'};

const fromRows = (kind: ModelKind, rows: Row[]): ModelProvider[] =>
  rows.map(([id, label, defaultModel, icon]) => ({
    id,
    label,
    kind,
    anchor: ANCHOR_OVERRIDES[`${kind}-${id}`] ?? `${kind}-${id}`,
    defaultModel,
    ...(icon?.startsWith('/') ? {imgSrc: icon} : {icon: icon ? llmIcon(icon) : undefined}),
  }));

const embeddings = fromRows('embeddings', [
  ['hindsight', 'Hindsight Inference', 'hs-embed-1', HINDSIGHT_ICON],
  ['local', 'Local (SentenceTransformers)', 'BAAI/bge-small-en-v1.5'],
  ['onnx', 'ONNX Runtime', 'intfloat/multilingual-e5-small'],
  ['openai', 'OpenAI', 'text-embedding-3-small', 'openai'],
  ['openai-codex', 'OpenAI Codex', 'text-embedding-3-small', 'openai-codex'],
  ['openrouter', 'OpenRouter', 'perplexity/pplx-embed-v1-0.6b', 'openrouter'],
  ['requesty', 'Requesty', 'openai/text-embedding-3-small', 'requesty'],
  ['cohere', 'Cohere', 'embed-english-v3.0'],
  ['google', 'Google (Gemini, Vertex AI)', 'gemini-embedding-001', 'gemini'],
  ['zeroentropy', 'ZeroEntropy', 'zembed-1'],
  ['nvidia', 'NVIDIA NIM', 'nvidia/llama-nemotron-embed-vl-1b-v2'],
  ['tei', 'Text Embeddings Inference'],
  ['litellm', 'LiteLLM', 'text-embedding-3-small', 'litellm'],
  ['bedrock', 'AWS Bedrock', undefined, 'bedrock'],
]);

const reranker = fromRows('reranker', [
  ['hindsight', 'Hindsight Inference', 'hs-rerank-1', HINDSIGHT_ICON],
  ['local', 'Local (SentenceTransformers)', 'cross-encoder/ms-marco-MiniLM-L-6-v2'],
  ['cohere', 'Cohere and compatible endpoints', 'rerank-english-v3.0'],
  ['openrouter', 'OpenRouter', 'cohere/rerank-v3.5', 'openrouter'],
  ['zeroentropy', 'ZeroEntropy', 'zerank-2'],
  ['siliconflow', 'SiliconFlow', 'BAAI/bge-reranker-v2-m3'],
  ['nvidia', 'NVIDIA NIM', 'nvidia/llama-nemotron-rerank-vl-1b-v2'],
  ['alibaba', 'Alibaba Cloud (DashScope)', 'qwen3-rerank'],
  ['google', 'Google Discovery Engine', 'semantic-ranker-default-004', 'gemini'],
  ['tei', 'Text Embeddings Inference'],
  ['flashrank', 'FlashRank', 'ms-marco-MiniLM-L-12-v2'],
  ['onnx', 'ONNX Runtime', 'cross-encoder/ms-marco-MiniLM-L-6-v2'],
  ['jina-mlx', 'Jina (Apple Silicon MLX)', 'jinaai/jina-reranker-v3-mlx'],
  ['litellm', 'LiteLLM', 'cohere/rerank-english-v3.0', 'litellm'],
  ['rrf', 'RRF only (no model)'],
  ['typesafe', 'TypeSafe (deprecated)', 'jev-latest'],
]);

const decision = fromRows('decision', [
  ['hindsight', 'Hindsight Inference', 'hs-decide-1-flash', HINDSIGHT_ICON],
  ['typesafe', 'TypeSafe (Jev)', 'jev-latest'],
  ['cloudflare', 'Cloudflare Clef', 'clef'],
  ['openai', 'OpenAI Decisions', 'gpt-6-luna', 'openai'],
]);

export const MODEL_PROVIDERS: ModelProvider[] = [...llm, ...embeddings, ...reranker, ...decision].map(withBrand);
