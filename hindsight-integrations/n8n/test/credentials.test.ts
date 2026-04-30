import { describe, expect, it } from 'vitest';

import { HindsightApi } from '../credentials/HindsightApi.credentials';

describe('HindsightApi credentials', () => {
	const cred = new HindsightApi();

	it('declares the expected name and displayName', () => {
		expect(cred.name).toBe('hindsightApi');
		expect(cred.displayName).toBe('Hindsight API');
	});

	it('exposes apiUrl and apiKey properties', () => {
		const propNames = cred.properties.map((p) => p.name);
		expect(propNames).toContain('apiUrl');
		expect(propNames).toContain('apiKey');
	});

	it('defaults apiUrl to Hindsight Cloud', () => {
		const apiUrlProp = cred.properties.find((p) => p.name === 'apiUrl');
		expect(apiUrlProp?.default).toBe('https://api.hindsight.vectorize.io');
	});

	it('marks apiKey as a password field', () => {
		const apiKeyProp = cred.properties.find((p) => p.name === 'apiKey');
		expect(apiKeyProp?.typeOptions?.password).toBe(true);
	});

	it('uses a generic Bearer-token auth scheme', () => {
		expect(cred.authenticate.type).toBe('generic');
		const props = cred.authenticate.properties as { headers?: Record<string, string> };
		expect(props.headers?.Authorization).toContain('Bearer');
	});

	it('tests against /health', () => {
		expect(cred.test.request.url).toBe('/health');
		expect(cred.test.request.method).toBe('GET');
	});
});
