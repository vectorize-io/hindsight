import React from 'react';
import Link from '@theme-original/DocSidebarItem/Link';
import type LinkType from '@theme/DocSidebarItem/Link';
import type {WrapperProps} from '@docusaurus/types';
import type {IconType} from 'react-icons';

import {
  LuBrain, LuRefreshCw, LuSearch, LuMessageSquare, LuLanguages,
  LuZap, LuDatabase, LuGitCompare, LuRocket, LuMemoryStick,
  LuWebhook, LuFileText, LuServer, LuSettings, LuTerminal,
  LuActivity, LuPlug, LuShield, LuPackage, LuBook,
  LuNetwork, LuCode2, LuLayers, LuCpu,
} from 'react-icons/lu';
import {SiGo, SiPython} from 'react-icons/si';

const ICON_MAP: Record<string, IconType> = {
  'lu-brain':       LuBrain,
  'lu-refresh':     LuRefreshCw,
  'lu-search':      LuSearch,
  'lu-message':     LuMessageSquare,
  'lu-languages':   LuLanguages,
  'lu-zap':         LuZap,
  'lu-database':    LuDatabase,
  'lu-compare':     LuGitCompare,
  'lu-rocket':      LuRocket,
  'lu-memory':      LuMemoryStick,
  'lu-webhook':     LuWebhook,
  'lu-file':        LuFileText,
  'lu-server':      LuServer,
  'lu-settings':    LuSettings,
  'lu-terminal':    LuTerminal,
  'lu-activity':    LuActivity,
  'lu-plug':        LuPlug,
  'lu-shield':      LuShield,
  'lu-package':     LuPackage,
  'lu-book':        LuBook,
  'lu-network':     LuNetwork,
  'lu-code':        LuCode2,
  'lu-layers':      LuLayers,
  'lu-cpu':         LuCpu,
  'si-go':          SiGo,
  'si-python':      SiPython,
};

type Props = WrapperProps<typeof LinkType>;

export default function LinkWrapper(props: Props): JSX.Element {
  const {item} = props;
  const icon = item.customProps?.icon as string | undefined;

  if (!icon) {
    return <Link {...props} />;
  }

  const IconComponent = ICON_MAP[icon];

  const iconNode = IconComponent
    ? <IconComponent size={16} style={{flexShrink: 0, opacity: 0.65}} />
    : <img src={icon} alt="" style={{width: '16px', height: '16px', flexShrink: 0, objectFit: 'contain'}} />;

  const modifiedItem = {
    ...item,
    label: (
      <span style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
        {iconNode}
        <span>{item.label}</span>
      </span>
    ),
  };

  return <Link {...props} item={modifiedItem} />;
}
