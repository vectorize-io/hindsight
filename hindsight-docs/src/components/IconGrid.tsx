import React from 'react';
import type {IconType} from 'react-icons';
import styles from './IconGrid.module.css';

export interface IconGridItem {
  label: string;
  href?: string;
  icon?: IconType;
  imgSrc?: string;
  /** Brand colour for the mark. Omitted = the neutral inherited colour. */
  color?: string;
  /** Dark-theme override, for brand colours that are near-black. */
  colorDark?: string;
}

export function IconGrid({items}: {items: IconGridItem[]}) {
  return (
    <div className={styles.grid}>
      {items.map(({label, href, icon: Icon, imgSrc, color, colorDark}) => {
        const card = (
          <div className={`${styles.card} ${href ? styles.cardLink : ''}`}>
            <div
              className={styles.icon}
              style={{
                ...(color ? {['--icon-color' as string]: color} : {}),
                ...(colorDark ? {['--icon-color-dark' as string]: colorDark} : {}),
              }}
            >
              {Icon && <Icon size={28} />}
              {imgSrc && <img src={imgSrc} alt={label} style={{width: 28, height: 28, objectFit: 'contain'}} />}
            </div>
            <span className={styles.label} style={{color: 'var(--ifm-font-color-base)', WebkitTextFillColor: 'var(--ifm-font-color-base)'}}>{label}</span>
          </div>
        );
        return href
          ? <a key={label} href={href} className={styles.anchor}>{card}</a>
          : <div key={label}>{card}</div>;
      })}
    </div>
  );
}
