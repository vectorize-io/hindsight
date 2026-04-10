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
  LuNetwork, LuCode, LuLayers, LuCpu, LuHardDrive,
  LuArrowUpRight, LuBookOpen, LuRss, LuCloud, LuMessageCircle,
  LuChartBar, LuChartColumn, LuStar, LuCircleHelp,
  LuLayoutTemplate, LuFileJson,
} from 'react-icons/lu';
import {SiGo, SiPython, SiGithub, SiSlack, SiDocker, SiKubernetes, SiNodedotjs} from 'react-icons/si';

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
  'lu-code':        LuCode,
  'lu-layers':      LuLayers,
  'lu-cpu':         LuCpu,
  'lu-hard-drive':  LuHardDrive,
  'si-go':          SiGo,
  'si-python':      SiPython,
  'si-github':      SiGithub,
  'si-slack':       SiSlack,
  'si-docker':      SiDocker,
  'si-kubernetes':  SiKubernetes,
  'si-nodedotjs':   SiNodedotjs,
  'lu-chart-bar':   LuChartBar,
  'lu-chart-column': LuChartColumn,
  'lu-arrow-up-right': LuArrowUpRight,
  'lu-book-open':   LuBookOpen,
  'lu-rss':         LuRss,
  'lu-cloud':       LuCloud,
  'lu-message-circle': LuMessageCircle,
  'lu-star':        LuStar,
  'lu-circle-help': LuCircleHelp,
  'lu-file-text':   LuFileText,
  'lu-layout-template': LuLayoutTemplate,
  'lu-file-json':   LuFileJson,
};

type Props = WrapperProps<typeof LinkType>;

export default function LinkWrapper(props: Props): JSX.Element {
  const {item} = props;
  const icon = item.customProps?.icon as string | undefined;
  const iconAfter = item.customProps?.iconAfter as string | undefined;

  if (!icon && !iconAfter) {
    return <Link {...props} />;
  }

  const IconComponent = icon ? ICON_MAP[icon] : undefined;
  const IconAfterComponent = iconAfter ? ICON_MAP[iconAfter] : undefined;

  const iconNode = icon
    ? IconComponent
      ? <IconComponent size={16} style={{flexShrink: 0, opacity: 0.65}} />
      : <img src={icon} alt="" style={{width: '16px', height: '16px', flexShrink: 0, objectFit: 'contain'}} />
    : null;

  const iconAfterNode = IconAfterComponent
    ? <IconAfterComponent size={13} style={{flexShrink: 0, opacity: 0.45}} />
    : null;

  const modifiedItem = {
    ...item,
    label: (
      <span style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
        {iconNode}
        <span style={{flex: 1}}>{item.label}</span>
        {iconAfterNode}
      </span>
    ),
  };

  return <Link {...props} item={modifiedItem} />;
}
