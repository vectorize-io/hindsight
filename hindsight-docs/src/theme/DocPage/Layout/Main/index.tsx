import React from 'react';
import Main from '@theme-original/DocPage/Layout/Main';
import type MainType from '@theme/DocPage/Layout/Main';
import type {WrapperProps} from '@docusaurus/types';
import {useLocation} from '@docusaurus/router';

type Props = WrapperProps<typeof MainType>;

export default function MainWrapper(props: Props): JSX.Element {
  const location = useLocation();
  const isCookbook = location.pathname.includes('/cookbook');

  return (
    <div style={isCookbook ? {maxWidth: '100%', width: '100%'} : undefined}>
      <Main {...props} />
    </div>
  );
}
