// A figure imports `../src`, which resolves to the JSX React entry — fine for vite, unparseable for
// plain node. Point that one specifier at the model-only shim so `node` can render a figure to SVG
// without a build step. Everything else resolves normally.
const shim = new URL('../src/node-figures.ts', import.meta.url).href;

export function resolve(specifier, context, next) {
  if (/^\.\.?\/src$/.test(specifier) && context.parentURL?.includes('/figures/')) {
    return { url: shim, shortCircuit: true, format: 'module-typescript' };
  }
  return next(specifier, context);
}
