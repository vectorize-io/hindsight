import React from 'react';
import type {IconType} from 'react-icons';
import {SiPython, SiGo, SiOpenai, SiAnthropic, SiGooglegemini, SiOllama, SiVercel} from 'react-icons/si';
import {LuTerminal, LuZap, LuBrainCog, LuSparkles, LuGlobe} from 'react-icons/lu';
import styles from './IntegrationsBanner.module.css';

interface BannerItem {
  label: string;
  icon?: IconType;
  imgSrc?: string;
  href?: string;
}

const OpenAICompatibleIcon: IconType = ({size = 18, ...props}) => (
  <span style={{position: 'relative', display: 'inline-flex'}}>
    <SiOpenai size={size} {...props} />
    <span style={{
      position: 'absolute', bottom: -2, right: -5,
      fontSize: Math.round((size as number) * 0.5), fontWeight: 900, lineHeight: 1,
      color: 'currentColor',
    }}>+</span>
  </span>
);

const ITEMS: BannerItem[] = [
  // Clients
  {label: 'Python',     icon: SiPython,   href: '/sdks/python'},
  {label: 'TypeScript', imgSrc: '/img/icons/typescript.png', href: '/sdks/nodejs'},
  {label: 'Go',         icon: SiGo,       href: '/sdks/go'},
  {label: 'CLI',        icon: LuTerminal, href: '/sdks/cli'},
  {label: 'HTTP',       icon: LuGlobe,    href: '/developer/api/quickstart'},
  // Integrations
  {label: 'MCP Server',    imgSrc: '/img/icons/mcp.png',       href: '/sdks/integrations/local-mcp'},
  {label: 'LiteLLM',      imgSrc: '/img/icons/litellm.png',    href: '/sdks/integrations/litellm'},
  {label: 'Claude Code',  imgSrc: '/img/icons/claudecode.svg', href: '/sdks/integrations/claude-code'},
  {label: 'Grok Build',   imgSrc: '/img/icons/grok-build.png', href: '/sdks/integrations/grok-build'},
  {label: 'OpenClaw',     imgSrc: '/img/icons/openclaw.png',   href: '/sdks/integrations/openclaw'},
  {label: 'Vercel AI',    icon: SiVercel,                      href: '/sdks/integrations/ai-sdk'},
  {label: 'Vercel Chat',  icon: SiVercel,                      href: '/sdks/integrations/chat'},
  {label: 'CrewAI',       imgSrc: '/img/icons/crewai.png',     href: '/sdks/integrations/crewai'},
  {label: 'Pydantic AI',  imgSrc: '/img/icons/pydanticai.png', href: '/sdks/integrations/pydantic-ai'},
  {label: 'Skills',       imgSrc: '/img/icons/skills.png',     href: '/sdks/integrations/skills'},
  {label: 'Agno',         imgSrc: '/img/icons/agno.png',       href: '/sdks/integrations/agno'},
  {label: 'Hermes',       imgSrc: '/img/icons/hermes.png',     href: '/sdks/integrations/hermes'},
  // LLM Providers
  {label: 'OpenAI',        icon: SiOpenai},
  {label: 'Anthropic',     icon: SiAnthropic},
  {label: 'Gemini',        icon: SiGooglegemini},
  {label: 'Groq',          icon: LuZap},
  {label: 'Ollama',        icon: SiOllama},
  {label: 'LM Studio',     icon: LuBrainCog},
  {label: 'MiniMax',       icon: LuSparkles},
  {label: 'OpenAI Compat', icon: OpenAICompatibleIcon},
];

function BannerItemComponent({item}: {item: BannerItem}) {
  const content = (
    <span className={styles.item}>
      <span className={styles.itemIcon}>
        {item.icon && <item.icon size={18} />}
        {item.imgSrc && <img src={item.imgSrc} alt={item.label} width={18} height={18} style={{objectFit: 'contain'}} />}
      </span>
      <span className={styles.itemLabel}>{item.label}</span>
    </span>
  );
  return item.href
    ? <a href={item.href} className={styles.itemLink}>{content}</a>
    : <span>{content}</span>;
}

export default function IntegrationsBanner(): JSX.Element {
  const doubled = [...ITEMS, ...ITEMS];
  return (
    <div className={styles.banner}>
      <div className={styles.track}>
        {doubled.map((item, i) => (
          <BannerItemComponent key={`${item.label}-${i}`} item={item} />
        ))}
      </div>
    </div>
  );
}
