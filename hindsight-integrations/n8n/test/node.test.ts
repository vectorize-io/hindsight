import { describe, expect, it } from 'vitest';

import { Hindsight } from '../nodes/Hindsight/Hindsight.node';

describe('Hindsight node', () => {
	const node = new Hindsight();

	it('declares the expected node metadata', () => {
		expect(node.description.name).toBe('hindsight');
		expect(node.description.displayName).toBe('Hindsight');
		expect(node.description.version).toBe(1);
	});

	it('requires the hindsightApi credential', () => {
		const cred = node.description.credentials?.[0];
		expect(cred?.name).toBe('hindsightApi');
		expect(cred?.required).toBe(true);
	});

	it('exposes retain / recall / reflect operations', () => {
		const operationProp = node.description.properties.find((p) => p.name === 'operation');
		expect(operationProp).toBeDefined();
		const values = (operationProp?.options as Array<{ value: string }>).map((o) => o.value);
		expect(values).toEqual(['retain', 'recall', 'reflect']);
	});

	it('requires a bankId for every operation', () => {
		const bankIdProp = node.description.properties.find((p) => p.name === 'bankId');
		expect(bankIdProp).toBeDefined();
		expect(bankIdProp?.required).toBe(true);
		// No displayOptions = visible for every operation
		expect(bankIdProp?.displayOptions).toBeUndefined();
	});

	it('shows content only for retain', () => {
		const contentProp = node.description.properties.find((p) => p.name === 'content');
		expect(contentProp?.displayOptions?.show?.operation).toEqual(['retain']);
	});

	it('shows recall query only for recall', () => {
		const queryProp = node.description.properties.find((p) => p.name === 'recallQuery');
		expect(queryProp?.displayOptions?.show?.operation).toEqual(['recall']);
	});

	it('shows reflect query only for reflect', () => {
		const queryProp = node.description.properties.find((p) => p.name === 'reflectQuery');
		expect(queryProp?.displayOptions?.show?.operation).toEqual(['reflect']);
	});

	it('exposes budget options low/mid/high for recall and reflect', () => {
		for (const propName of ['recallBudget', 'reflectBudget']) {
			const prop = node.description.properties.find((p) => p.name === propName);
			expect(prop).toBeDefined();
			const values = (prop?.options as Array<{ value: string }>).map((o) => o.value);
			expect(values).toEqual(['low', 'mid', 'high']);
		}
	});
});
