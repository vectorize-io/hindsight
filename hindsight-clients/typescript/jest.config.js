module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/tests/**/*.test.ts"],
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json", "node"],
  transform: {
    "^.+\\.tsx?$": "ts-jest",
  },
  // 300s, not 120s: CI's shared inference endpoint answers HTTP 429 under load, and the
  // server's LLM retries (8, backoff from 2s) can spend minutes on one call (#6391).
  // Same budget as the Python client tests.
  testTimeout: 300000,
};
