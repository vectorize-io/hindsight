import React from 'react';
import type {IconType} from 'react-icons';
import {IconGrid} from './IconGrid';
import {SiPython, SiGo, SiOpenai, SiAnthropic, SiGooglegemini, SiOllama} from 'react-icons/si';
import {LuTerminal, LuPlug, LuZap, LuBrainCog, LuSparkles, LuGlobe, LuLayers, LuCloud} from 'react-icons/lu';

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

export function ClientsGrid() {
  return (
    <IconGrid items={[
      { label: 'Python',     icon: SiPython,   href: '/sdks/python' },
      { label: 'TypeScript', imgSrc: '/img/icons/typescript.png', href: '/sdks/nodejs' },
      { label: 'Go',         icon: SiGo,       href: '/sdks/go' },
      { label: 'CLI',        icon: LuTerminal, href: '/sdks/cli' },
      { label: 'HTTP',       icon: LuGlobe,    href: '/developer/api/quickstart' },
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
      { label: 'MiniMax',            icon: LuSparkles },
      { label: 'Volcano Engine',    icon: LuZap },
      { label: 'OpenAI Compatible', icon: OpenAICompatibleIcon },
      { label: 'AWS Bedrock', icon: LuCloud },
      { label: 'LiteLLM (100+)', icon: LuLayers },
    ]} />
  );
}
