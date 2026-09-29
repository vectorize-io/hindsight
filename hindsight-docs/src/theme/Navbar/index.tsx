import React, {type ReactNode} from 'react';
import NavbarLayout from '@theme/Navbar/Layout';
import NavbarContent from '@theme/Navbar/Content';

export default function Navbar(): ReactNode {
  return (
    <NavbarLayout>
      <NavbarContent />
    </NavbarLayout>
  );
}
