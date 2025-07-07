// Utility to handle base URL for internal links
export function getUrl(path: string): string {
  const baseUrl = import.meta.env.BASE_URL;
  
  if (path === '/') return baseUrl;
  
  // Ensure we don't double up on slashes
  const base = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  const pathPart = path.startsWith('/') ? path : `/${path}`;
  
  return `${base}${pathPart}`;
}