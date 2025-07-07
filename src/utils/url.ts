// Utility to handle base URL for internal links
export function getUrl(path: string): string {
  const baseUrl = import.meta.env.BASE_URL;
  if (path.startsWith('/')) {
    return `${baseUrl}${path.slice(1)}`;
  }
  return path;
}