/**
 * Pluggable logger interface.
 *
 * This package does not own any logging infrastructure — consumers inject
 * whatever they want (console, pino, openclaw's logger, a no-op). The default
 * is silent so embedding this package never adds noise to an unrelated app.
 */
export interface Logger {
  debug(msg: string): void;
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
}

/** Logger that drops every call. Used when no logger is passed. */
export const silentLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};

/** Logger that writes to the standard console. Handy for CLIs and tests. */
export const consoleLogger: Logger = {
  debug: (msg) => console.debug(msg),
  info: (msg) => console.log(msg),
  warn: (msg) => console.warn(msg),
  error: (msg) => console.error(msg),
};
