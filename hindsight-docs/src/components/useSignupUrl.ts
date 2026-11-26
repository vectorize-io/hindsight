import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/** The hosted app's sign-up page (customFields.appUrl in docusaurus.config.ts). */
export function useSignupUrl(): string {
  const {siteConfig} = useDocusaurusContext();
  return `${siteConfig.customFields?.appUrl as string}/signup`;
}
