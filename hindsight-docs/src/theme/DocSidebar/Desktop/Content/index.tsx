import React from 'react';
import Content from '@theme-original/DocSidebar/Desktop/Content';
import type ContentType from '@theme/DocSidebar/Desktop/Content';
import type {WrapperProps} from '@docusaurus/types';

type Props = WrapperProps<typeof ContentType>;

export default function ContentWrapper(props: Props): JSX.Element {
  return <Content {...props} />;
}
