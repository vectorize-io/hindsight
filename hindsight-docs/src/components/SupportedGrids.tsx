import React from 'react';
import {IconGrid} from './IconGrid';
import {SiPython, SiGo} from 'react-icons/si';
import {LuTerminal, LuGlobe} from 'react-icons/lu';
import {LLM_PROVIDERS} from '../data/llmProviders';

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
  return <IconGrid items={LLM_PROVIDERS.map(({label, icon, color, colorDark}) => ({label, icon, color, colorDark}))} />;
}
