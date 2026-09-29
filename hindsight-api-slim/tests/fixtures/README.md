# Test Fixtures

## locomo_conversation_sample.json

Sample conversation from the LoComo benchmark (conv-26) used for performance tuning tests.

**Stats:**
- Sample ID: conv-26
- Sessions: 19
- Total dialogues: 419
- Questions: 199

**Usage:**
Used by `test_performance_tuning.py` to measure:
- Batch ingestion performance
- Search performance
- Entity resolution performance

This is a realistic long-form conversation for stress testing the memory system.
