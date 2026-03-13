import React from 'react';
import {IconGrid} from './IconGrid';
import {SiPython, SiGo, SiOpenai, SiAnthropic, SiGooglegemini, SiOllama, SiCrewai, SiPydantic, SiVercel} from 'react-icons/si';
import {LuTerminal, LuPlug, LuZap, LuBrainCog, LuSparkles} from 'react-icons/lu';

export function ClientsGrid() {
  return (
    <IconGrid items={[
      { label: 'Python',                icon: SiPython,  href: '/sdks/python' },
      { label: 'TypeScript', imgSrc: '/img/icons/typescript.png', href: '/sdks/nodejs' },
      { label: 'Go',                    icon: SiGo,      href: '/sdks/go' },
      { label: 'CLI',                   icon: LuTerminal, href: '/sdks/cli' },
    ]} />
  );
}

export function IntegrationsGrid() {
  return (
    <IconGrid items={[
      { label: 'Local MCP Server', imgSrc: '/img/icons/mcp.png',       href: '/sdks/integrations/local-mcp' },
      { label: 'LiteLLM',         imgSrc: '/img/icons/litellm.png',    href: '/sdks/integrations/litellm' },
      { label: 'OpenClaw',        imgSrc: '/img/icons/openclaw.png',    href: '/sdks/integrations/openclaw' },
      { label: 'Vercel AI SDK',   icon: SiVercel,                       href: '/sdks/integrations/ai-sdk' },
      { label: 'Vercel Chat SDK', icon: SiVercel,                       href: '/sdks/integrations/chat' },
      { label: 'CrewAI',          imgSrc: '/img/icons/crewai.png',      href: '/sdks/integrations/crewai' },
      { label: 'Pydantic AI',     imgSrc: '/img/icons/pydanticai.png',  href: '/sdks/integrations/pydantic-ai' },
      { label: 'Skills',          imgSrc: '/img/icons/skills.png',      href: '/sdks/integrations/skills' },
    ]} />
  );
}

export function LLMProvidersGrid() {
  return (
    <IconGrid items={[
      { label: 'OpenAI',        icon: SiOpenai },
      { label: 'Anthropic',     icon: SiAnthropic },
      { label: 'Google Gemini', icon: SiGooglegemini },
      { label: 'Groq',          icon: LuZap },
      { label: 'Ollama',        icon: SiOllama },
      { label: 'LM Studio',     icon: LuBrainCog },
      { label: 'MiniMax',       icon: LuSparkles },
    ]} />
  );
}
