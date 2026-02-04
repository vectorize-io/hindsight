// Client-side localStorage utilities for username persistence

const USERNAME_KEY = 'taste-ai-username';

export function getStoredUsername(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(USERNAME_KEY);
}

export function setStoredUsername(username: string): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(USERNAME_KEY, username);
}

export function clearStoredUsername(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(USERNAME_KEY);
}
