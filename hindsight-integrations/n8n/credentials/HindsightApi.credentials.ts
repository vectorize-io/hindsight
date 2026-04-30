import type {
	IAuthenticateGeneric,
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

/**
 * Credentials for the Hindsight memory API.
 *
 * Defaults to Hindsight Cloud. For self-hosted instances, change the API URL
 * to your deployment (e.g. http://localhost:8888).
 */
export class HindsightApi implements ICredentialType {
	name = 'hindsightApi';
	displayName = 'Hindsight API';
	documentationUrl = 'https://hindsight.vectorize.io/developer/api/quickstart';
	properties: INodeProperties[] = [
		{
			displayName: 'API URL',
			name: 'apiUrl',
			type: 'string',
			default: 'https://api.hindsight.vectorize.io',
			description:
				'Base URL of the Hindsight API. Defaults to Hindsight Cloud; change for self-hosted instances.',
			required: true,
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description:
				'API key for Hindsight Cloud (begins with "hsk_"). Leave blank for unauthenticated self-hosted instances.',
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				Authorization: '={{ $credentials.apiKey ? "Bearer " + $credentials.apiKey : undefined }}',
			},
		},
	};

	// Test against /health — works for both Cloud and self-hosted
	test: ICredentialTestRequest = {
		request: {
			baseURL: '={{ $credentials.apiUrl }}',
			url: '/health',
			method: 'GET',
		},
	};
}
