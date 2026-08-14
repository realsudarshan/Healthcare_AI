export type Geo = {
  city?: string;
  country?: string;
  latitude?: string;
  longitude?: string;
};

function header(request: Request, name: string) {
  return request.headers.get(name)?.trim() || undefined;
}

/**
 * Client IP as seen through a reverse proxy (nginx, Caddy, Traefik, a load
 * balancer). Returns undefined when the app is hit directly.
 */
export function getClientIp(request: Request) {
  const forwardedFor = header(request, "x-forwarded-for");

  if (forwardedFor) {
    return forwardedFor.split(",")[0].trim() || undefined;
  }

  return header(request, "x-real-ip");
}

/**
 * Optional geo hints. A reverse proxy with a GeoIP module can set these; when
 * nothing sets them we return an empty object and the system prompt omits the
 * location section entirely.
 */
export function getGeo(request: Request): Geo {
  return {
    city: header(request, "x-geo-city"),
    country: header(request, "x-geo-country"),
    latitude: header(request, "x-geo-latitude"),
    longitude: header(request, "x-geo-longitude"),
  };
}
