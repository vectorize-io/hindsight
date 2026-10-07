import 'giotto/player';

declare module 'react' {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace JSX {
    interface IntrinsicElements {
      'giotto-player': {doc?: string; src?: string; autoplay?: string; speed?: string};
    }
  }
}

/** An animated Giotto figure from hindsight-docs/figures/*.json. */
export default function Figure({doc}: {doc: object}) {
  return <giotto-player doc={JSON.stringify(doc)} />;
}
