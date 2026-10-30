const {lazyCompilationMiddleware} = require('@rspack/core');

/**
 * Light local dev server (DOCS_LIGHT=true, set by scripts/dev/start-docs.sh).
 *
 * `docusaurus start` compiles every page of every docs version, blog post and
 * guide up front and keeps all of it in memory — about 3.6 GB. This plugin makes
 * the bundler compile a page only when a browser first opens it, and drops
 * source maps. Never active in `docusaurus build`.
 */
module.exports = function devLightPlugin() {
  return {
    name: 'dev-light',
    configureWebpack(config, isServer) {
      if (isServer || process.env.NODE_ENV === 'production') return {};
      return {
        devtool: false,
        // Docusaurus merges all CSS into one styles.css. The browser has already
        // loaded it when a page compiles on first visit, so that page renders
        // unstyled. Per-page CSS loads together with the page instead.
        optimization: {splitChunks: {cacheGroups: {styles: false}}},
        module: {
          rules: [
            {
              test: /[\\/]\.docusaurus[\\/]registry\.js$/,
              enforce: 'pre',
              use: [require.resolve('./drop-weak-refs-loader')],
            },
          ],
        },
        lazyCompilation: {entries: false, imports: true},
        // Serves the requests that trigger a lazy compile. This replaces
        // Docusaurus's own setupMiddlewares, which only adds source contents to
        // the error overlay — moot here, since source maps are off.
        devServer: {
          setupMiddlewares: (middlewares, devServer) => {
            middlewares.unshift(lazyCompilationMiddleware(devServer.compiler));
            return middlewares;
          },
        },
      };
    },
  };
};
