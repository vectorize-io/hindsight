import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import type {Props} from '@theme/Navbar/Search';

import styles from './styles.module.css';

export default function NavbarSearch({children, className}: Props): ReactNode {
  return (
    <div className={clsx(className, styles.navbarSearchContainer)}>
      {children}
    </div>
  );
}
