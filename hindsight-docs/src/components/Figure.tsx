import 'giotto/player';

declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      'giotto-player': {doc?: string; src?: string; autoplay?: string; speed?: string};
    }
  }
}

/** An animated Giotto figure from hindsight-docs/figures/*.json. */
export default function Figure({doc}: {doc: object}) {
  // A custom element gets React props as attributes, which are strings: the player parses this one.
  return <giotto-player doc={JSON.stringify(doc)} />;
}
