import React from 'react';
import DefaultNavbarItem from '@theme-original/NavbarItem/DefaultNavbarItem';
import type DefaultNavbarItemType from '@theme/NavbarItem/DefaultNavbarItem';
import type {WrapperProps} from '@docusaurus/types';
import type {IconType} from 'react-icons';
import {
  LuArrowUpRight, LuCode, LuCircleHelp, LuScrollText,
  LuLayoutGrid, LuLayoutTemplate, LuCloud, LuBook, LuRss, LuBookOpen,
  LuChartBar, LuCpu, LuFileText, LuStar,
} from 'react-icons/lu';
import {SiGithub, SiSlack} from 'react-icons/si';

const ICON_MAP: Record<string, IconType> = {
  'lu-code':         LuCode,
  'lu-circle-help':  LuCircleHelp,
  'lu-scroll-text':  LuScrollText,
  'lu-layout-grid':     LuLayoutGrid,
  'lu-layout-template': LuLayoutTemplate,
  'lu-cloud':        LuCloud,
  'lu-book':         LuBook,
  'lu-rss':          LuRss,
  'lu-book-open':    LuBookOpen,
  'lu-chart-bar':    LuChartBar,
  'lu-cpu':          LuCpu,
  'lu-file-text':    LuFileText,
  'lu-star':         LuStar,
  'si-github':       SiGithub,
  'si-slack':        SiSlack,
};

type Props = WrapperProps<typeof DefaultNavbarItemType>;

export default function DefaultNavbarItemWrapper(props: Props): JSX.Element {
  const isExternal = typeof props.href === 'string' && props.href.startsWith('http');
  const isGithub = (props.className as string | undefined)?.includes('header-github-link');
  const iconKey = props.customProps?.icon as string | undefined;
  const IconComponent = iconKey ? ICON_MAP[iconKey] : undefined;

  if (isGithub) {
    return <DefaultNavbarItem {...props} label={<SiGithub size={18} style={{display: 'block'}} />} />;
  }

  if (!IconComponent && !isExternal) {
    return <DefaultNavbarItem {...props} />;
  }

  const modifiedProps = {
    ...props,
    label: (
      <span style={{display: 'inline-flex', alignItems: 'center', gap: '5px'}}>
        {IconComponent && <IconComponent size={14} style={{flexShrink: 0, opacity: 0.8}} />}
        {props.label}
        {isExternal && <LuArrowUpRight size={11} style={{flexShrink: 0, opacity: 0.45}} />}
      </span>
    ),
  };

  return <DefaultNavbarItem {...modifiedProps} />;
}
