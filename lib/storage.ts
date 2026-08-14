import {
  GetObjectCommand,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";

const DEFAULT_REGION = "us-east-1";

/**
 * S3-compatible object storage (MinIO, Ceph, Backblaze B2, AWS S3, ...).
 * Everything is env-driven so the app stays portable across hosts.
 */
const config = {
  accessKeyId: process.env.S3_ACCESS_KEY_ID,
  bucket: process.env.S3_BUCKET,
  endpoint: process.env.S3_ENDPOINT,
  forcePathStyle: process.env.S3_FORCE_PATH_STYLE === "true",
  publicUrl: process.env.S3_PUBLIC_URL?.replace(/\/$/, "") || undefined,
  region: process.env.S3_REGION ?? DEFAULT_REGION,
  secretAccessKey: process.env.S3_SECRET_ACCESS_KEY,
};

export function isStorageConfigured() {
  return Boolean(config.bucket && config.accessKeyId && config.secretAccessKey);
}

/** Absolute base URL for objects, or null when they must be proxied by the app. */
export function getPublicBaseUrl() {
  return config.publicUrl ?? null;
}

let client: S3Client | null = null;

function getClient() {
  if (!isStorageConfigured()) {
    throw new Error(
      "Object storage is not configured. Set S3_BUCKET, S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY."
    );
  }

  if (!client) {
    client = new S3Client({
      credentials: {
        accessKeyId: config.accessKeyId as string,
        secretAccessKey: config.secretAccessKey as string,
      },
      // Only set for non-AWS endpoints; the SDK derives the AWS one from region.
      ...(config.endpoint ? { endpoint: config.endpoint } : {}),
      forcePathStyle: config.forcePathStyle,
      region: config.region,
    });
  }

  return client;
}

export async function putObject(
  key: string,
  body: Uint8Array,
  contentType: string
) {
  await getClient().send(
    new PutObjectCommand({
      Body: body,
      Bucket: config.bucket,
      ContentType: contentType,
      Key: key,
    })
  );

  return { key };
}

export async function getObject(key: string) {
  const result = await getClient().send(
    new GetObjectCommand({
      Bucket: config.bucket,
      Key: key,
    })
  );

  if (!result.Body) {
    return null;
  }

  const body = await result.Body.transformToByteArray();

  return {
    body,
    contentType: result.ContentType ?? "application/octet-stream",
  };
}
