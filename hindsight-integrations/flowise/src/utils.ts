/**
 * Local shim for `packages/components/src/utils` from the upstream
 * Flowise tree. Tool nodes import these helpers via deep relative paths
 * (e.g. `from '../../../src/utils'`); we re-export the shim
 * implementation so those imports resolve naturally from this directory
 * without altering the source files.
 *
 * The actual implementations live in `./Interface.ts` to avoid
 * duplication; this file exists purely so the upstream relative path
 * (`<components>/src/utils`) resolves locally.
 */
export { getBaseClasses, getCredentialData, getCredentialParam } from "./Interface";
