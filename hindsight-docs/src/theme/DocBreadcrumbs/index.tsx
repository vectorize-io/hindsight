import React from 'react';
import Breadcrumbs from '@theme-original/DocBreadcrumbs';
import type BreadcrumbsType from '@theme/DocBreadcrumbs';
import type {WrapperProps} from '@docusaurus/types';
import SkillBanner from '@site/src/components/SkillBanner';

type Props = WrapperProps<typeof BreadcrumbsType>;

export default function BreadcrumbsWrapper(props: Props): JSX.Element {
  return (
    <>
      <Breadcrumbs {...props} />
      <SkillBanner />
    </>
  );
}
