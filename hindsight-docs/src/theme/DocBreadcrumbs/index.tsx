import React from 'react';
import Breadcrumbs from '@theme-original/DocBreadcrumbs';
import type BreadcrumbsType from '@theme/DocBreadcrumbs';
import type {WrapperProps} from '@docusaurus/types';
import {useLocation} from '@docusaurus/router';
import SkillBanner from '@site/src/components/SkillBanner';

type Props = WrapperProps<typeof BreadcrumbsType>;

export default function BreadcrumbsWrapper(props: Props): JSX.Element {
  // The site root opens on the hero, which has to be the first thing on the
  // page — a bordered strip above it puts a box between the navbar and the
  // band and undoes the full-bleed. The homepage renders its own <SkillBanner />
  // just below the hero instead (docs/developer/index.mdx).
  const isHome = useLocation().pathname.replace(/\/$/, '') === '';

  return (
    <>
      <Breadcrumbs {...props} />
      {!isHome && <SkillBanner />}
    </>
  );
}
