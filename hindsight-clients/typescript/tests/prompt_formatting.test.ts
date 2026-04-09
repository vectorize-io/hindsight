/**
 * Tests for recallResponseToPromptString.
 */

import { recallResponseToPromptString } from '../src';

describe('recallResponseToPromptString', () => {
    test('facts only', () => {
        const response = {
            results: [
                {
                    id: '1',
                    text: 'Alice works at Google',
                    context: 'work',
                    occurred_start: '2024-01-15T10:00:00Z',
                    occurred_end: '2024-06-15T10:00:00Z',
                    mentioned_at: '2024-03-01T09:00:00Z',
                },
                { id: '2', text: 'The sky is blue' },
            ],
        };
        const prompt = recallResponseToPromptString(response);
        expect(prompt.startsWith('FACTS:\n')).toBe(true);
        const facts = JSON.parse(prompt.slice('FACTS:\n'.length));
        expect(facts).toEqual([
            {
                text: 'Alice works at Google',
                context: 'work',
                occurred_start: '2024-01-15T10:00:00Z',
                occurred_end: '2024-06-15T10:00:00Z',
                mentioned_at: '2024-03-01T09:00:00Z',
            },
            { text: 'The sky is blue' },
        ]);
    });

    test('with chunks', () => {
        const response = {
            results: [
                { id: '1', text: 'Alice works at Google', chunk_id: 'chunk_1' },
            ],
            chunks: {
                chunk_1: { id: 'chunk_1', text: 'Alice works at Google on the AI team since 2020.', chunk_index: 0 },
            },
        };
        const prompt = recallResponseToPromptString(response);
        const facts = JSON.parse(prompt.slice('FACTS:\n'.length));
        expect(facts[0].source_chunk).toBe('Alice works at Google on the AI team since 2020.');
    });

    test('with entities', () => {
        const response = {
            results: [
                { id: '1', text: 'Alice works at Google' },
            ],
            entities: {
                Alice: {
                    entity_id: 'e1',
                    canonical_name: 'Alice',
                    observations: [{ text: 'Alice is a senior engineer at Google working on AI.' }],
                },
            },
        };
        const prompt = recallResponseToPromptString(response);
        expect(prompt).toContain('ENTITIES:');
        expect(prompt).toContain('## Alice');
        expect(prompt).toContain('Alice is a senior engineer at Google working on AI.');
    });

    test('with chunks and entities', () => {
        const response = {
            results: [
                { id: '1', text: 'Alice works at Google', chunk_id: 'c1' },
            ],
            chunks: {
                c1: { id: 'c1', text: 'Full conversation about Alice at Google.', chunk_index: 0 },
            },
            entities: {
                Alice: {
                    entity_id: 'e1',
                    canonical_name: 'Alice',
                    observations: [{ text: 'Alice is a senior engineer.' }],
                },
            },
        };
        const prompt = recallResponseToPromptString(response);
        const factsSection = prompt.split('ENTITIES:')[0].trim();
        const facts = JSON.parse(factsSection.slice('FACTS:\n'.length));
        expect(facts[0].source_chunk).toBe('Full conversation about Alice at Google.');
        expect(prompt).toContain('## Alice\nAlice is a senior engineer.');
    });

    test('empty results', () => {
        const response = { results: [] };
        expect(recallResponseToPromptString(response)).toBe('FACTS:\n[]');
    });

    test('chunk_id not in chunks is ignored', () => {
        const response = {
            results: [
                { id: '1', text: 'Some fact', chunk_id: 'missing_chunk' },
            ],
        };
        const prompt = recallResponseToPromptString(response);
        const facts = JSON.parse(prompt.slice('FACTS:\n'.length));
        expect(facts[0]).not.toHaveProperty('source_chunk');
    });
});
