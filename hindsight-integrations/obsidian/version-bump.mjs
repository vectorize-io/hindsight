// Runs as the npm `version` lifecycle script, which `scripts/release-integration.sh` triggers via
// `npm version`. Obsidian, BRAT and the community store read manifest.json + versions.json, not
// package.json; before this hook, v0.3.1 shipped a manifest still saying 0.3.0 (#5969).
import { readFileSync, writeFileSync } from "node:fs";

const version = process.env.npm_package_version;
if (!version) throw new Error("npm_package_version is unset: run via `npm version <v>`");

const manifest = JSON.parse(readFileSync("manifest.json", "utf8"));
manifest.version = version;
writeFileSync("manifest.json", JSON.stringify(manifest, null, 2) + "\n");

const versions = JSON.parse(readFileSync("versions.json", "utf8"));
versions[version] = manifest.minAppVersion;
writeFileSync("versions.json", JSON.stringify(versions, null, 2) + "\n");
