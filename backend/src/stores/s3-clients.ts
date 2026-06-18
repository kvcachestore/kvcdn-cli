import { S3Client } from "@aws-sdk/client-s3";

export interface S3Connection {
  endpoint: string;
  accessKeyId: string;
  secretAccessKey: string;
}

export function createS3Client(conn: S3Connection): S3Client {
  return new S3Client({
    region: "us-east-1",
    endpoint: conn.endpoint,
    forcePathStyle: true,
    credentials: { accessKeyId: conn.accessKeyId, secretAccessKey: conn.secretAccessKey },
  });
}
