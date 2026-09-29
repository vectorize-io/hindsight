import React from 'react';
import DocItemContent from '@theme-original/DocItem/Content';
import type DocItemContentType from '@theme/DocItem/Content';
import type { WrapperProps } from '@docusaurus/types';

type Props = WrapperProps<typeof DocItemContentType>;

export default function DocItemContentWrapper(props: Props): JSX.Element {
  return <DocItemContent {...props} />;
}
