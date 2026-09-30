import {usePluginData} from '@docusaurus/useGlobalData';

/**
 * Star count from the build-time `github-stars` plugin, pre-formatted for
 * display ("12.4k"). Shared by the navbar GitHub item and the homepage hero so
 * the two can never disagree.
 */
export function useGitHubStars(): string {
  const data = usePluginData('github-stars') as {stars?: number} | undefined;
  const stars = data?.stars;
  if (typeof stars !== 'number') return '';
  return stars >= 1000 ? `${(stars / 1000).toFixed(1)}k` : String(stars);
}
