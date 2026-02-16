import React from 'react';
import Link from '@theme-original/DocSidebarItem/Link';
import type LinkType from '@theme/DocSidebarItem/Link';
import type {WrapperProps} from '@docusaurus/types';

type Props = WrapperProps<typeof LinkType>;

export default function LinkWrapper(props: Props): JSX.Element {
  const {item} = props;
  const icon = item.customProps?.icon as string | undefined;

  if (!icon) {
    return <Link {...props} />;
  }

  // Create a modified item with icon in the label
  const modifiedItem = {
    ...item,
    label: (
      <span style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
        <img
          src={icon}
          alt=""
          style={{
            width: '16px',
            height: '16px',
            flexShrink: 0
          }}
        />
        <span>{item.label}</span>
      </span>
    ),
  };

  return <Link {...props} item={modifiedItem} />;
}
